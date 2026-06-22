//! Adversarial runtime-robustness suite for the two untrusted-input paths.
//!
//! Goal 022 (input-robustness). This is a permanent regression asset. It
//! throws a battery of malformed / hostile inputs at BOTH parsers that ingest
//! untrusted data and asserts the production code degrades gracefully — it
//! returns `Err` / `None` / an `Error` response, or silently ignores the bad
//! bit — but NEVER panics, hangs, infinite-loops, or OOMs.
//!
//! TARGET 1 — STEP file parser (`echomap::io::load_step_file`): parses
//! untrusted `.STEP` CAD files chosen by the user via the import dialog.
//!
//! TARGET 2 — agent JSON protocol (`ClientMessage` deserialization +
//! `AgentSession::handle_message`): parses untrusted JSON arriving over
//! TCP/WS from a network client, then dispatches it.
//!
//! Two real bugs were found and fixed at root cause while building this suite
//! (see the dedicated regression tests near the end of each target section):
//!   * STEP: `truncate_for_message` sliced a warning string at a fixed BYTE
//!     offset (80) that could land mid-codepoint on a multibyte char →
//!     "byte index is not a char boundary" panic. Fixed to snap to a UTF-8
//!     boundary (src/io/step_parser.rs).
//!
//! Everything else here is a *confirmation* that the parsers already shrug off
//! hostile input — equally valuable as a guarantee that future edits keep that
//! property.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use echomap::agent::bridge::create_bridge;
use echomap::agent::protocol::{ClientMessage, ServerMessage};
use echomap::agent::session::AgentSession;
use echomap::io::load_step_file;
use echomap::robot::definition::RobotDefinition;
use echomap::robot::state::RobotAction;
use echomap::robot::RobotManager;
use glam::Mat4;

// ===========================================================================
// TARGET 1 — STEP parser
// ===========================================================================
//
// `load_step_file` is file-based, so each adversarial input is written to a
// unique temp file first. The contract (documented on `StepLoadResult`) is:
// the parser NEVER panics — anything it cannot interpret becomes a warning and
// parsing continues. A file lacking the `ISO-10303-21` magic is the one case
// that returns `Err(InvalidFormat)`, which is still graceful (not a panic).

/// Monotonic counter so concurrently-running test threads never collide on a
/// fixture filename.
static STEP_FIXTURE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Write `body` to a uniquely-named temp file and return its path. Bytes are
/// written raw (not as UTF-8 text) so non-UTF-8 fuzz payloads survive intact.
fn step_fixture(tag: &str, body: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join("echomap_adversarial_step");
    std::fs::create_dir_all(&dir).expect("create temp fixture dir");
    let seq = STEP_FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("{tag}_{seq}.step"));
    std::fs::write(&path, body).expect("write fixture");
    path
}

/// Drive `load_step_file` against a byte payload and assert it returns (either
/// Ok or Err) without panicking. `std::panic::catch_unwind` is not needed —
/// the test harness already turns a panic into a failed test — but returning
/// the result lets individual cases make extra assertions.
fn load_bytes(tag: &str, body: &[u8]) -> Result<echomap::io::StepLoadResult, String> {
    let path = step_fixture(tag, body);
    load_step_file(&path).map_err(|e| e.to_string())
}

/// Convenience: a minimal valid STEP envelope wrapping a custom DATA body.
fn step_envelope(data_body: &str) -> String {
    format!("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{data_body}\nENDSEC;\nEND-ISO-10303-21;\n")
}

#[test]
fn step_empty_input() {
    // Completely empty file: lacks the magic → graceful InvalidFormat Err.
    let res = load_bytes("empty", b"");
    assert!(
        res.is_err(),
        "empty file should be a graceful Err, not a panic"
    );
}

#[test]
fn step_only_whitespace() {
    let res = load_bytes("whitespace", b"   \n\t  \r\n   ");
    assert!(
        res.is_err(),
        "whitespace-only file should be Err (no magic)"
    );
}

#[test]
fn step_magic_only_no_sections() {
    // Has the magic but nothing else: no DATA;, no ENDSEC;. Must not panic;
    // returns Ok with empty geometry + a warning.
    let res = load_bytes("magic_only", b"ISO-10303-21;\n").expect("magic-only must parse Ok");
    assert!(res.objects.is_empty());
    assert!(
        !res.warnings.is_empty(),
        "expected an empty-geometry warning"
    );
}

#[test]
fn step_truncated_mid_token() {
    // File is cut off mid-entity: open paren, open quote, no terminators.
    let body = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = CARTESIAN_POINT('p1',(0.0,0.0";
    load_bytes("truncated", body.as_bytes()).expect("truncated file must not panic");
}

#[test]
fn step_truncated_mid_header() {
    let body = "ISO-10303-21;\nHEADER;\nFILE_NAME('x";
    load_bytes("trunc_header", body.as_bytes()).expect("truncated header must not panic");
}

#[test]
fn step_missing_data_section() {
    // Magic + HEADER but no DATA section at all.
    let body = "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('X'));\nENDSEC;\nEND-ISO-10303-21;\n";
    let res = load_bytes("no_data", body.as_bytes()).expect("missing DATA must not panic");
    assert!(res.objects.is_empty());
}

#[test]
fn step_missing_endsec() {
    // DATA; present, ENDSEC; absent → parser slices to end-of-string. No panic.
    let body = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = CARTESIAN_POINT('p',(0.0,0.0,0.0));\n";
    load_bytes("no_endsec", body.as_bytes()).expect("missing ENDSEC must not panic");
}

#[test]
fn step_data_marker_but_no_entities() {
    let body = step_envelope("");
    let res = load_bytes("empty_data", body.as_bytes()).expect("empty DATA must not panic");
    assert!(res.objects.is_empty());
}

#[test]
fn step_huge_declared_count_no_data() {
    // A BREP claims a shell ref that doesn't exist, plus a CLOSED_SHELL that
    // references thousands of faces none of which are defined. The parser must
    // walk the dangling refs, warn, and finish — not allocate per declared id.
    let mut body = String::from("#1 = MANIFOLD_SOLID_BREP('Box',#2);\n");
    body.push_str("#2 = CLOSED_SHELL('S',(");
    for i in 0..5000 {
        if i > 0 {
            body.push(',');
        }
        // references #1000000..#1004999, none defined
        body.push_str(&format!("#{}", 1_000_000 + i));
    }
    body.push_str("));\n");
    let env = step_envelope(&body);
    let res = load_bytes("huge_count", env.as_bytes()).expect("huge ref list must not panic");
    // No geometry can be built from dangling refs.
    assert!(res.objects.is_empty());
}

#[test]
fn step_deeply_nested_parens() {
    // 20k nested parens in an entity arg list. `split_top_args` tracks depth
    // with an i32 counter (no recursion) so this must not stack-overflow.
    let depth = 20_000;
    let mut arg = String::with_capacity(depth * 2 + 8);
    arg.push_str("#1 = WEIRD('n',");
    for _ in 0..depth {
        arg.push('(');
    }
    for _ in 0..depth {
        arg.push(')');
    }
    arg.push_str(");");
    let env = step_envelope(&arg);
    load_bytes("nested_parens", env.as_bytes()).expect("deep nesting must not panic/overflow");
}

#[test]
fn step_unbalanced_parens() {
    // More opens than closes, and vice versa. depth counter goes negative /
    // stays positive; neither path indexes out of bounds.
    let env1 = step_envelope("#1 = CLOSED_SHELL('S',((((((#2);");
    load_bytes("unbalanced_open", env1.as_bytes()).expect("unbalanced opens must not panic");

    let env2 = step_envelope("#1 = CLOSED_SHELL('S',))))))#2));");
    load_bytes("unbalanced_close", env2.as_bytes()).expect("unbalanced closes must not panic");
}

#[test]
fn step_unterminated_string_literal() {
    // An apostrophe opens a string that never closes → `in_string` stays true
    // for the rest of the arg scan. Must not loop or panic.
    let env = step_envelope("#1 = CARTESIAN_POINT('never closes,(1.0,2.0,3.0));");
    load_bytes("unterminated_str", env.as_bytes()).expect("unterminated string must not panic");
}

#[test]
fn step_non_utf8_bytes() {
    // `load_step_file` uses `read_to_string`, which rejects non-UTF-8 with an
    // Io error. The point: it must be a graceful Err, never a panic.
    let mut body = b"ISO-10303-21;\nDATA;\n#1 = X(".to_vec();
    body.extend_from_slice(&[0xFF, 0xFE, 0x80, 0x00, 0xC0, 0xC1]);
    body.extend_from_slice(b");\nENDSEC;\n");
    let res = load_bytes("non_utf8", &body);
    assert!(
        res.is_err(),
        "non-UTF-8 bytes should yield a graceful Io Err"
    );
}

#[test]
fn step_non_utf8_in_otherwise_valid_file() {
    // Invalid UTF-8 buried mid-file after the magic. Still an Io Err, no panic.
    let mut body = step_envelope("#1 = CARTESIAN_POINT('p',(0.0,0.0,0.0));").into_bytes();
    // splice a lone continuation byte in the middle
    let mid = body.len() / 2;
    body.insert(mid, 0x80);
    let res = load_bytes("non_utf8_mid", &body);
    assert!(
        res.is_err(),
        "mid-file invalid UTF-8 should be a graceful Err"
    );
}

#[test]
fn step_nan_inf_coordinates() {
    // NaN / inf / -inf coordinates parse as f32 (Rust accepts "NaN","inf").
    // Geometry built from them goes through `normalize_or_zero`, which is
    // NaN-safe, so no panic — at worst the face is dropped.
    let body = step_envelope(
        "#1 = CARTESIAN_POINT('a',(NaN,inf,-inf));\n\
         #2 = CARTESIAN_POINT('b',(1.0,NaN,0.0));\n\
         #3 = CARTESIAN_POINT('c',(inf,inf,inf));\n\
         #4 = VERTEX_POINT('v1',#1);\n\
         #5 = VERTEX_POINT('v2',#2);\n\
         #6 = VERTEX_POINT('v3',#3);",
    );
    load_bytes("nan_coords", body.as_bytes()).expect("NaN/inf coords must not panic");
}

#[test]
fn step_gigantic_coordinate_values() {
    // Values far beyond f32 range parse to inf; subtraction/cross/normalize
    // stay finite-or-zero. No panic, no overflow trap (floats saturate).
    let body = step_envelope(
        "#1 = CARTESIAN_POINT('a',(1e300,1e300,1e300));\n\
         #2 = CARTESIAN_POINT('b',(-1e308,9.9e307,1e38));\n\
         #3 = CARTESIAN_POINT('c',(1e9999,1e9999,1e9999));",
    );
    load_bytes("giant_coords", body.as_bytes()).expect("gigantic coords must not panic");
}

#[test]
fn step_duplicate_entity_ids() {
    // Same id defined twice — HashMap insert just overwrites. No panic.
    let body = step_envelope(
        "#1 = CARTESIAN_POINT('a',(0.0,0.0,0.0));\n\
         #1 = CARTESIAN_POINT('b',(1.0,1.0,1.0));\n\
         #1 = CLOSED_SHELL('dup',(#1));",
    );
    load_bytes("dup_ids", body.as_bytes()).expect("duplicate ids must not panic");
}

#[test]
fn step_forward_referencing_ids() {
    // Entities reference ids that appear LATER in the file. The two-pass design
    // (parse all, then resolve) means forward refs resolve fine; but even a
    // forward ref to a never-defined id must only warn, not panic.
    let body = step_envelope(
        "#1 = MANIFOLD_SOLID_BREP('Box',#2);\n\
         #2 = CLOSED_SHELL('S',(#3));\n\
         #3 = ADVANCED_FACE('f',(#4),#99,.T.);\n\
         #4 = FACE_OUTER_BOUND('b',#5,.T.);",
    );
    load_bytes("forward_refs", body.as_bytes()).expect("forward refs must not panic");
}

#[test]
fn step_self_referencing_chain() {
    // A pile of mutually- and self-referencing entities. The resolver has a
    // visited-set + depth cap, so cycles terminate. Bounded time asserts no
    // infinite loop.
    let body = step_envelope(
        "#1 = MANIFOLD_SOLID_BREP('a',#1);\n\
         #2 = CLOSED_SHELL('b',(#2,#3));\n\
         #3 = ADVANCED_FACE('c',(#3),#3,.T.);\n\
         #4 = VERTEX_POINT('d',#4);\n\
         #5 = VERTEX_POINT('e',#6);\n\
         #6 = VERTEX_POINT('f',#5);",
    );
    let start = std::time::Instant::now();
    load_bytes("self_ref_chain", body.as_bytes()).expect("self-ref chain must not panic");
    assert!(
        start.elapsed().as_millis() < 1000,
        "self-ref chain must terminate quickly (no infinite loop)"
    );
}

#[test]
fn step_entity_id_overflow() {
    // An id far beyond u32::MAX fails the u32 parse → entity skipped (None),
    // no panic. An id just over MAX_ENTITY_ID is warned-and-skipped.
    let body = step_envelope(
        "#99999999999999999999 = CARTESIAN_POINT('overflow',(0.0,0.0,0.0));\n\
         #20000000 = CARTESIAN_POINT('over_cap',(1.0,1.0,1.0));\n\
         #1 = CARTESIAN_POINT('ok',(2.0,2.0,2.0));",
    );
    let res = load_bytes("id_overflow", body.as_bytes()).expect("id overflow must not panic");
    assert!(
        res.warnings.iter().any(|w| w.contains("MAX_ENTITY_ID")),
        "over-cap id should warn, got {:?}",
        res.warnings
    );
}

#[test]
fn step_negative_and_garbage_ids() {
    let body = step_envelope(
        "#-5 = CARTESIAN_POINT('neg',(0.0,0.0,0.0));\n\
         #abc = CARTESIAN_POINT('alpha',(1.0,1.0,1.0));\n\
         # = CARTESIAN_POINT('empty',(2.0,2.0,2.0));",
    );
    load_bytes("garbage_ids", body.as_bytes()).expect("garbage ids must not panic");
}

#[test]
fn step_ten_megabyte_junk_blob() {
    // A 10 MB blob of random-looking junk that still carries the magic so the
    // parser actually walks it. Must finish well under any hang threshold.
    let mut blob = String::with_capacity(10 * 1024 * 1024 + 64);
    blob.push_str("ISO-10303-21;\nDATA;\n");
    // Fill with a repeating non-entity pattern peppered with stray '#' and '('
    // to exercise the ref-scanner and arg-splitter on garbage.
    let chunk = "garbage #12 ((( 'unterminated ,,, ))) ### nonsense %%% \n";
    while blob.len() < 10 * 1024 * 1024 {
        blob.push_str(chunk);
    }
    blob.push_str("\nENDSEC;\n");
    let start = std::time::Instant::now();
    load_bytes("ten_mb_junk", blob.as_bytes()).expect("10MB junk must not panic");
    assert!(
        start.elapsed().as_secs() < 10,
        "10MB junk parse must not hang (took {:?})",
        start.elapsed()
    );
}

#[test]
fn step_ten_megabyte_long_single_line() {
    // A single 10MB "line" (no newlines) of one unterminated entity. Exercises
    // the line-accumulator and arg scanners on a pathologically long token.
    let mut blob = String::with_capacity(10 * 1024 * 1024 + 64);
    blob.push_str("ISO-10303-21;\nDATA;\n#1 = CLOSED_SHELL('s',(");
    while blob.len() < 10 * 1024 * 1024 {
        blob.push_str("#7,");
    }
    // deliberately leave it unterminated (no `));`)
    let start = std::time::Instant::now();
    load_bytes("ten_mb_line", blob.as_bytes()).expect("10MB single line must not panic");
    assert!(
        start.elapsed().as_secs() < 10,
        "10MB single-line parse must not hang (took {:?})",
        start.elapsed()
    );
}

#[test]
fn step_degenerate_faces() {
    // Faces with collinear / coincident / <3 vertices. `triangulate_face` and
    // `compute_face_normal` both early-return on <3 verts and use
    // normalize_or_zero, so degenerate geometry is dropped, never panics.
    // Build a full valid-ish shell whose single face collapses to a line.
    let body = step_envelope(
        "#1 = MANIFOLD_SOLID_BREP('Degen',#2);\n\
         #2 = CLOSED_SHELL('S',(#3));\n\
         #3 = ADVANCED_FACE('f',(#4),#20,.T.);\n\
         #4 = FACE_OUTER_BOUND('b',#5,.T.);\n\
         #5 = EDGE_LOOP('l',(#6,#7,#8));\n\
         #6 = ORIENTED_EDGE('',*,*,#9,.T.);\n\
         #7 = ORIENTED_EDGE('',*,*,#10,.T.);\n\
         #8 = ORIENTED_EDGE('',*,*,#11,.T.);\n\
         #9 = EDGE_CURVE('',#12,#13,#21,.T.);\n\
         #10 = EDGE_CURVE('',#13,#14,#21,.T.);\n\
         #11 = EDGE_CURVE('',#14,#12,#21,.T.);\n\
         #12 = VERTEX_POINT('',#15);\n\
         #13 = VERTEX_POINT('',#16);\n\
         #14 = VERTEX_POINT('',#17);\n\
         #15 = CARTESIAN_POINT('',(0.0,0.0,0.0));\n\
         #16 = CARTESIAN_POINT('',(1.0,0.0,0.0));\n\
         #17 = CARTESIAN_POINT('',(2.0,0.0,0.0));",
    );
    load_bytes("degenerate_faces", body.as_bytes()).expect("degenerate faces must not panic");
}

#[test]
fn step_zero_area_coincident_vertices() {
    // All three vertices identical → zero-area face → normalize_or_zero → Vec3::ZERO
    // normal, triangle still emitted but no panic / NaN propagation crash.
    let body = step_envelope(
        "#1 = MANIFOLD_SOLID_BREP('Coincident',#2);\n\
         #2 = CLOSED_SHELL('S',(#3));\n\
         #3 = ADVANCED_FACE('f',(#4),#20,.T.);\n\
         #4 = FACE_OUTER_BOUND('b',#5,.T.);\n\
         #5 = EDGE_LOOP('l',(#6,#7,#8));\n\
         #6 = ORIENTED_EDGE('',*,*,#9,.T.);\n\
         #7 = ORIENTED_EDGE('',*,*,#10,.T.);\n\
         #8 = ORIENTED_EDGE('',*,*,#11,.T.);\n\
         #9 = EDGE_CURVE('',#12,#13,#21,.T.);\n\
         #10 = EDGE_CURVE('',#13,#14,#21,.T.);\n\
         #11 = EDGE_CURVE('',#14,#12,#21,.T.);\n\
         #12 = VERTEX_POINT('',#15);\n\
         #13 = VERTEX_POINT('',#15);\n\
         #14 = VERTEX_POINT('',#15);\n\
         #15 = CARTESIAN_POINT('',(3.0,3.0,3.0));",
    );
    load_bytes("coincident_verts", body.as_bytes()).expect("coincident verts must not panic");
}

#[test]
fn step_null_bytes_interspersed() {
    // Embedded NUL bytes are valid UTF-8 (U+0000), so read_to_string accepts
    // them. The parser must treat them as ordinary chars, not panic.
    let body = step_envelope("#1 = CARTESIAN_POINT('p',(0.0,0.0,0.0));");
    let mut bytes = body.into_bytes();
    bytes.insert(40, 0x00);
    bytes.insert(10, 0x00);
    load_bytes("null_bytes", &bytes).expect("embedded NUL must not panic");
}

/// REGRESSION (root-cause fix): a malformed entity record longer than 80
/// BYTES whose 80th byte falls inside a multibyte UTF-8 character used to
/// panic in `truncate_for_message` ("byte index 80 is not a char boundary").
/// The line must (a) contain `#`, (b) be >4 chars, and (c) fail
/// `parse_entity_line` (here: no `(` after `=`) so the truncating warning path
/// fires. Fixed to snap the cut to a char boundary.
#[test]
fn step_regression_multibyte_warning_truncation_no_panic() {
    // 78 ASCII filler + a 4-byte emoji that straddles byte index 80, no '(' so
    // `parse_entity_line` returns None and the line is routed to
    // `truncate_for_message`. The trailing ';' is required so the line
    // accumulator actually flushes and processes the record.
    let filler = "A".repeat(78);
    let malformed = format!("#1 = {filler}\u{1F980}TAIL;");
    let body = step_envelope(&malformed);
    let res = load_bytes("regr_mb_trunc", body.as_bytes())
        .expect("multibyte malformed line must NOT panic");
    assert!(
        res.warnings
            .iter()
            .any(|w| w.contains("malformed entity record")),
        "expected the malformed-record warning to fire, got {:?}",
        res.warnings
    );
}

/// Several multibyte chars at varied offsets around the 80-byte cut, to make
/// sure the boundary walk-back is robust regardless of where the codepoint
/// lands.
#[test]
fn step_regression_multibyte_at_varied_offsets_no_panic() {
    for pad in 77..=82 {
        let filler = "B".repeat(pad);
        // 4-byte emoji right after the filler so the cut can land anywhere
        // across its bytes depending on `pad`. Trailing ';', no '(' → the
        // record flushes and routes through the truncating warning path.
        let malformed = format!("#9 = {filler}\u{1F4A9}{};", "z".repeat(20));
        let body = step_envelope(&malformed);
        load_bytes("regr_mb_offset", body.as_bytes())
            .unwrap_or_else(|_| panic!("multibyte truncation at pad={pad} must not panic"));
    }
}

// ===========================================================================
// TARGET 2 — agent JSON protocol
// ===========================================================================
//
// Two stages, mirroring the real network path in tcp_server.rs / ws_server.rs:
//   STAGE A: raw bytes/string -> `serde_json::from_str::<ClientMessage>` must
//            never panic (graceful Err on anything malformed).
//   STAGE B: a deserialized `ClientMessage` -> `AgentSession::handle_message`
//            must return a graceful `ServerMessage` (Error, not panic), even
//            for out-of-range robot ids and shape-mismatched actions.

/// STAGE A helper: deserialization of an arbitrary string must not panic.
/// Returns whether it parsed (for cases that care).
fn try_parse_client(raw: &str) -> bool {
    serde_json::from_str::<ClientMessage>(raw).is_ok()
}

#[test]
fn json_malformed_garbage() {
    for raw in [
        "",
        "   ",
        "\0\0\0",
        "not json at all",
        "{",
        "}",
        "{\"type\":",
        "{\"type\":\"connect\",}",
        "[",
        "[[[[[[",
        "{{{{{{",
        "\"unterminated string",
        "{'single':'quotes'}",
        "{type:connect}",
        "tru",
        "nul",
        "0x1F",
        "{\"type\":\"connect\" \"robot_id\":0}", // missing comma
        "\u{FEFF}{\"type\":\"reset\"}",          // BOM prefix
    ] {
        // Must not panic; almost all are Err. Whatever the verdict, no crash.
        let _ = try_parse_client(raw);
    }
}

#[test]
fn json_valid_but_wrong_schema() {
    for raw in [
        r#"{"not_type":"connect"}"#,
        r#"{"type":42}"#,
        r#"{"type":["connect"]}"#,
        r#"{"type":{"nested":"object"}}"#,
        r#"{"type":null}"#,
        r#"42"#,
        r#"true"#,
        r#"null"#,
        r#""just a string""#,
        r#"[1,2,3]"#,
        r#"[{"type":"reset"}]"#, // array, not object
        r#"{}"#,
    ] {
        assert!(
            !try_parse_client(raw),
            "wrong-schema input should fail to parse: {raw}"
        );
    }
}

#[test]
fn json_unknown_message_type() {
    for raw in [
        r#"{"type":"explode"}"#,
        r#"{"type":"connect "}"#, // trailing space → unknown variant
        r#"{"type":"CONNECT"}"#,  // wrong case (snake_case expected)
        r#"{"type":"drop_table"}"#,
        r#"{"type":"step ","action":{}}"#,
    ] {
        assert!(
            !try_parse_client(raw),
            "unknown message type should be rejected: {raw}"
        );
    }
}

#[test]
fn json_missing_required_fields() {
    for raw in [
        r#"{"type":"connect"}"#,                                  // needs robot_id
        r#"{"type":"step"}"#,                                     // needs action
        r#"{"type":"step","action":{}}"#, // action needs motor_velocities + gripper_commands
        r#"{"type":"step","action":{"motor_velocities":[1.0]}}"#, // needs gripper_commands
        r#"{"type":"send_message","content":"hi"}"#, // needs to_robot_id
        r#"{"type":"send_message","to_robot_id":1}"#, // needs content
        r#"{"type":"bind_target"}"#,      // needs target_id
    ] {
        assert!(
            !try_parse_client(raw),
            "missing-required-field input should fail: {raw}"
        );
    }
}

#[test]
fn json_null_where_struct_expected() {
    for raw in [
        r#"{"type":"step","action":null}"#,
        r#"{"type":"connect","robot_id":null}"#,
        r#"{"type":"step","action":{"motor_velocities":null,"gripper_commands":[]}}"#,
        r#"{"type":"send_message","to_robot_id":null,"content":null}"#,
    ] {
        assert!(
            !try_parse_client(raw),
            "null-where-value-expected should fail: {raw}"
        );
    }
}

#[test]
fn json_wrong_value_types() {
    for raw in [
        r#"{"type":"connect","robot_id":-1}"#, // usize can't be negative
        r#"{"type":"connect","robot_id":1.5}"#, // usize can't be float
        r#"{"type":"connect","robot_id":"zero"}"#, // usize can't be string
        r#"{"type":"step","action":{"motor_velocities":[true],"gripper_commands":[]}}"#, // f32 != bool
        r#"{"type":"step","action":{"motor_velocities":["x"],"gripper_commands":[]}}"#, // f32 != string
        r#"{"type":"step","action":{"motor_velocities":[],"gripper_commands":[1.0]}}"#, // bool != float
    ] {
        assert!(
            !try_parse_client(raw),
            "wrong value type should fail: {raw}"
        );
    }
}

#[test]
fn json_nan_and_infinity_literals() {
    // JSON has no NaN/Infinity; serde_json must reject the literals (not panic).
    for raw in [
        r#"{"type":"step","action":{"motor_velocities":[NaN],"gripper_commands":[]}}"#,
        r#"{"type":"step","action":{"motor_velocities":[Infinity],"gripper_commands":[]}}"#,
        r#"{"type":"step","action":{"motor_velocities":[-Infinity],"gripper_commands":[]}}"#,
        r#"{"type":"step","action":{"motor_velocities":[1e],"gripper_commands":[]}}"#, // malformed float
    ] {
        assert!(
            !try_parse_client(raw),
            "NaN/Inf/bad-float literal should fail: {raw}"
        );
    }
}

#[test]
fn json_integer_overflow_values() {
    // robot_id is usize. Values beyond u64::MAX overflow the integer parse and
    // must produce an Err, not a panic.
    for raw in [
        r#"{"type":"connect","robot_id":99999999999999999999999999999999}"#,
        r#"{"type":"send_message","to_robot_id":99999999999999999999999999999999,"content":"x"}"#,
    ] {
        assert!(
            !try_parse_client(raw),
            "integer overflow should fail: {raw}"
        );
    }
    // usize::MAX itself must parse fine (boundary — graceful Ok).
    let ok = format!(r#"{{"type":"connect","robot_id":{}}}"#, usize::MAX);
    assert!(try_parse_client(&ok), "usize::MAX robot_id should parse");
}

#[test]
fn json_huge_string_does_not_oom() {
    // A 5 MB string content field. Must parse (or fail) bounded — no OOM, no
    // hang. serde allocates O(n); 5 MB is fine and proves no quadratic blowup.
    let huge = "x".repeat(5 * 1024 * 1024);
    let raw = format!(r#"{{"type":"send_message","to_robot_id":1,"content":"{huge}"}}"#);
    let start = std::time::Instant::now();
    let parsed = serde_json::from_str::<ClientMessage>(&raw).is_ok();
    assert!(parsed, "huge content string should parse");
    assert!(
        start.elapsed().as_secs() < 10,
        "huge string parse must not hang ({:?})",
        start.elapsed()
    );
}

#[test]
fn json_huge_array_does_not_oom() {
    // A motor_velocities array with 1M entries. Bounded allocation, no panic.
    let mut raw = String::from(r#"{"type":"step","action":{"motor_velocities":["#);
    for i in 0..1_000_000 {
        if i > 0 {
            raw.push(',');
        }
        raw.push_str("0.0");
    }
    raw.push_str(r#"],"gripper_commands":[]}}"#);
    let start = std::time::Instant::now();
    let _ = serde_json::from_str::<ClientMessage>(&raw); // Ok expected, but no-panic is the point
    assert!(
        start.elapsed().as_secs() < 15,
        "huge array parse must not hang ({:?})",
        start.elapsed()
    );
}

#[test]
fn json_deeply_nested_does_not_stack_overflow() {
    // serde_json has a built-in recursion limit (128) that returns an Err
    // rather than overflowing the stack. Feed 100k levels of nested arrays in
    // the `type` position; it must Err, not crash.
    let depth = 100_000;
    let mut raw = String::with_capacity(depth * 2 + 32);
    raw.push_str(r#"{"type":"#);
    for _ in 0..depth {
        raw.push('[');
    }
    for _ in 0..depth {
        raw.push(']');
    }
    raw.push('}');
    let res = serde_json::from_str::<ClientMessage>(&raw);
    assert!(
        res.is_err(),
        "deeply nested JSON must be rejected (recursion limit), not stack-overflow"
    );
}

#[test]
fn json_deeply_nested_object_does_not_stack_overflow() {
    // Same idea but nested OBJECTS via a recursive value position.
    let depth = 100_000;
    let mut raw = String::with_capacity(depth * 8 + 32);
    for _ in 0..depth {
        raw.push_str(r#"{"a":"#);
    }
    raw.push('1');
    for _ in 0..depth {
        raw.push('}');
    }
    let res = serde_json::from_str::<ClientMessage>(&raw);
    assert!(
        res.is_err(),
        "deeply nested objects must be rejected, not overflow the stack"
    );
}

#[test]
fn json_duplicate_keys_and_injection() {
    // Duplicate keys (serde takes last) and attempted JSON injection in the
    // type discriminator. Neither may panic.
    let _ = try_parse_client(r#"{"type":"reset","type":"connect","robot_id":0}"#);
    let _ = try_parse_client(r#"{"type":"connect\",\"robot_id\":999","robot_id":0}"#);
    let _ = try_parse_client(r#"{"robot_id":0,"robot_id":1,"type":"connect"}"#);
}

#[test]
fn json_unicode_and_escapes_in_strings() {
    // Exotic-but-valid strings must parse cleanly (these are well-formed).
    for raw in [
        r#"{"type":"send_message","to_robot_id":1,"content":" 🦀"}"#,
        r#"{"type":"bind_target","target_id":"robot/ "}"#,
        r#"{"type":"send_message","to_robot_id":1,"content":"tab\there\nnewline"}"#,
    ] {
        // Well-formed → should parse; the assertion is really "no panic".
        let _ = try_parse_client(raw);
    }
    // A lone unpaired surrogate is INVALID JSON → must Err, not panic.
    assert!(
        !try_parse_client(r#"{"type":"send_message","to_robot_id":1,"content":"\ud83e"}"#),
        "lone surrogate is invalid JSON"
    );
}

// ---- STAGE B: dispatch of valid-but-hostile messages through a live session ----

/// Spin up a RobotManager with a single simple_arm(2) robot, a bridge pair,
/// and a background task draining bridge commands. Mirrors the in-crate test
/// harness so dispatch hits real production code (`handle_message` → bridge).
fn spawn_session() -> (AgentSession, tokio::task::JoinHandle<()>) {
    let mut manager = RobotManager::new();
    manager.add_robot(RobotDefinition::simple_arm(2), Mat4::IDENTITY);

    let (server, mut client) = create_bridge();
    let session = AgentSession::new(server);

    let handle = tokio::spawn(async move {
        loop {
            client.process_pending(&mut manager, &[]);
            tokio::task::yield_now().await;
        }
    });
    (session, handle)
}

#[tokio::test]
async fn dispatch_out_of_range_robot_id_returns_error_not_panic() {
    let (mut session, handle) = spawn_session();
    // Only robot 0 exists. Connect to wildly out-of-range ids.
    for bad in [1usize, 99, 1_000_000, usize::MAX] {
        let resp = session
            .handle_message(ClientMessage::Connect { robot_id: bad })
            .await;
        match resp {
            ServerMessage::Error { message, .. } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "expected invalid robot_id error for {bad}, got: {message}"
                );
            }
            other => panic!("expected graceful Error for robot_id {bad}, got {other:?}"),
        }
    }
    handle.abort();
}

#[tokio::test]
async fn dispatch_bind_target_out_of_range_and_garbage_returns_error() {
    // Out-of-range numeric target, then unparseable target strings.
    for target in [
        "robot/99999",
        "robot/18446744073709551615", // usize::MAX
        "robot/-1",
        "robot/abc",
        "robot/",
        "",
        "   ",
        "not_a_robot",
        "robot/0/extra",
    ] {
        // Fresh session each time so the "already bound" guard doesn't mask the
        // resolution path.
        let (mut session, handle) = spawn_session();
        let resp = session
            .handle_message(ClientMessage::BindTarget {
                target_id: target.to_string(),
                agent_type: None,
                domain: None,
                observe_only: false,
            })
            .await;
        assert!(
            matches!(resp, ServerMessage::Error { .. }),
            "bind_target '{target}' should yield Error, got {resp:?}"
        );
        handle.abort();
    }
}

#[tokio::test]
async fn dispatch_step_before_connect_returns_error() {
    let (mut session, handle) = spawn_session();
    let action = RobotAction {
        motor_velocities: vec![1.0, 2.0],
        gripper_commands: vec![],
        base_velocity: [0.0, 0.0],
    };
    let resp = session.handle_message(ClientMessage::Step { action }).await;
    assert!(
        matches!(resp, ServerMessage::Error { .. }),
        "step before connect must Error, got {resp:?}"
    );
    handle.abort();
}

#[tokio::test]
async fn dispatch_wrong_shape_action_returns_error_not_panic() {
    let (mut session, handle) = spawn_session();
    session
        .handle_message(ClientMessage::Connect { robot_id: 0 })
        .await;

    // simple_arm(2) expects 2 motors. Try 0, 1, 5, and a huge vector.
    for n in [0usize, 1, 5, 100_000] {
        let action = RobotAction {
            motor_velocities: vec![0.5; n],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        let resp = session.handle_message(ClientMessage::Step { action }).await;
        // n==2 would be valid, but we never send 2 here; all of these are
        // wrong-shape and must produce an Error (never a panic / OOB index).
        assert!(
            matches!(resp, ServerMessage::Error { .. }),
            "wrong motor count {n} must Error, got {resp:?}"
        );
    }
    handle.abort();
}

#[tokio::test]
async fn dispatch_nan_inf_action_components_return_error() {
    let (mut session, handle) = spawn_session();
    session
        .handle_message(ClientMessage::Connect { robot_id: 0 })
        .await;

    // Non-finite motor velocity (correct length so it reaches the finite check).
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let action = RobotAction {
            motor_velocities: vec![1.0, bad],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        let resp = session.handle_message(ClientMessage::Step { action }).await;
        assert!(
            matches!(resp, ServerMessage::Error { .. }),
            "non-finite motor velocity {bad} must Error, got {resp:?}"
        );

        // Non-finite base_velocity component is checked regardless of motors.
        let action2 = RobotAction {
            motor_velocities: vec![1.0, 1.0],
            gripper_commands: vec![],
            base_velocity: [bad, 0.0],
        };
        let resp2 = session
            .handle_message(ClientMessage::Step { action: action2 })
            .await;
        assert!(
            matches!(resp2, ServerMessage::Error { .. }),
            "non-finite base_velocity {bad} must Error, got {resp2:?}"
        );
    }
    handle.abort();
}

/// REGRESSION (root-cause fix): a Step with a NaN/inf `base_velocity` but an
/// otherwise-valid action used to slip past validation and reach the bridge,
/// where `f32::NAN.clamp(..) == NAN` propagated into `base_pose` and
/// permanently poisoned every subsequent observation with NaN. Fixed in two
/// layers: session-level validation (descriptive Error to the client) and a
/// bridge-level sanitize (defense-in-depth for any caller that skipped it).
/// This test proves a malicious base_velocity neither corrupts state nor
/// breaks future steps.
#[tokio::test]
async fn dispatch_nan_base_velocity_does_not_corrupt_future_observations() {
    let (mut session, handle) = spawn_session();
    session
        .handle_message(ClientMessage::Connect { robot_id: 0 })
        .await;

    // Hostile step: valid motors, poisoned base_velocity. Must be rejected.
    let bad = session
        .handle_message(ClientMessage::Step {
            action: RobotAction {
                motor_velocities: vec![0.0, 0.0],
                gripper_commands: vec![],
                base_velocity: [f32::NAN, f32::INFINITY],
            },
        })
        .await;
    assert!(
        matches!(bad, ServerMessage::Error { .. }),
        "poisoned base_velocity must be rejected, got {bad:?}"
    );

    // A subsequent legitimate step must produce a fully-finite observation —
    // proof that no NaN leaked into the robot's persistent state.
    let good = session
        .handle_message(ClientMessage::Step {
            action: RobotAction {
                motor_velocities: vec![0.5, -0.5],
                gripper_commands: vec![],
                base_velocity: [0.1, 0.1],
            },
        })
        .await;
    match good {
        ServerMessage::Observation { state, reward, .. } => {
            assert!(
                state.joint_positions.iter().all(|v| v.is_finite()),
                "joint positions corrupted by earlier NaN base_velocity: {:?}",
                state.joint_positions
            );
            assert!(
                state.joint_velocities.iter().all(|v| v.is_finite()),
                "joint velocities corrupted: {:?}",
                state.joint_velocities
            );
            assert!(reward.is_finite(), "reward became non-finite: {reward}");
        }
        other => panic!("expected a finite Observation, got {other:?}"),
    }
    handle.abort();
}

#[tokio::test]
async fn dispatch_send_message_to_out_of_range_robot_is_graceful() {
    let (mut session, handle) = spawn_session();
    session
        .handle_message(ClientMessage::Connect { robot_id: 0 })
        .await;

    // Sending to a non-existent robot must not panic; it returns either
    // MessageSent (queued for a robot that never reads) or an Error — both are
    // graceful, non-panicking outcomes.
    let resp = session
        .handle_message(ClientMessage::SendMessage {
            to_robot_id: usize::MAX,
            content: "hello void".to_string(),
        })
        .await;
    assert!(
        matches!(
            resp,
            ServerMessage::MessageSent | ServerMessage::Error { .. }
        ),
        "send to out-of-range robot must be graceful, got {resp:?}"
    );
    handle.abort();
}

#[tokio::test]
async fn dispatch_huge_message_content_does_not_panic() {
    let (mut session, handle) = spawn_session();
    session
        .handle_message(ClientMessage::Connect { robot_id: 0 })
        .await;

    // 2 MB of trash talk. The dispatch path must shuttle it through the bridge
    // without panicking or hanging.
    let content = "z".repeat(2 * 1024 * 1024);
    let resp = session
        .handle_message(ClientMessage::SendMessage {
            to_robot_id: 0,
            content,
        })
        .await;
    assert!(
        matches!(
            resp,
            ServerMessage::MessageSent | ServerMessage::Error { .. }
        ),
        "huge message content must be graceful, got {resp:?}"
    );
    handle.abort();
}

#[tokio::test]
async fn dispatch_full_hostile_sequence_keeps_session_alive() {
    // Fire a barrage of out-of-order / invalid messages and confirm the session
    // never panics and remains usable for a valid request afterward.
    let (mut session, handle) = spawn_session();

    // Close before connect, reset before connect, observe before connect,
    // cancel before connect, double messages — all must return gracefully.
    let _ = session.handle_message(ClientMessage::Close).await;
    let _ = session.handle_message(ClientMessage::Reset).await;
    let _ = session.handle_message(ClientMessage::Observe).await;
    let _ = session.handle_message(ClientMessage::Cancel).await;
    let _ = session
        .handle_message(ClientMessage::Step {
            action: RobotAction {
                motor_velocities: vec![],
                gripper_commands: vec![],
                base_velocity: [f32::NAN, f32::NAN],
            },
        })
        .await;

    // Session must still work for a legitimate connect + step.
    let resp = session
        .handle_message(ClientMessage::Connect { robot_id: 0 })
        .await;
    assert!(
        matches!(resp, ServerMessage::Connected { .. }),
        "session must recover and accept a valid connect, got {resp:?}"
    );

    let resp = session
        .handle_message(ClientMessage::Step {
            action: RobotAction {
                motor_velocities: vec![0.0, 0.0],
                gripper_commands: vec![],
                base_velocity: [0.0, 0.0],
            },
        })
        .await;
    assert!(
        matches!(resp, ServerMessage::Observation { .. }),
        "session must accept a valid step after the hostile barrage, got {resp:?}"
    );

    handle.abort();
}
