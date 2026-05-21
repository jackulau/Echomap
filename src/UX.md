# EchoMap UX — Shortcut Reference

This document is the source of truth for EchoMap's keyboard / mouse model.
It pairs with goal 012 "industry-grade UX parity" and tracks the equivalents
shipped for Cinema 4D, Blender, and SolidWorks idioms.

## Modes

| Key | Mode |
|---|---|
| `1` | Select |
| `2` | Place Source |
| `3` | Place Listener |
| `Tab` | Toggle fly camera (WASD + right-drag look) |
| `Ctrl+T` | Toggle tele-op (drives `robot/0` from WASD/QE) |
| `Ctrl+Alt+Q` | Toggle quad-view (Top / Front / Side / Perspective) |
| `Esc` | Cancel modal / clear selection |

## Command Palette

| Key | Action |
|---|---|
| `Cmd/Ctrl+K` | Open command palette (fuzzy search across all actions) |
| `↑ / ↓` | Navigate results |
| `Enter` | Run the highlighted action |
| `Esc` | Close palette |

The palette registers 30+ actions across categories: `view:`, `display:`,
`mode:`, `add:`, `edit:`, `sim:`, `robot:`, `misc:`.

## Transform Gizmos (Blender-style modal)

| Key | Action |
|---|---|
| `G` | Translate gizmo |
| `R` | Rotate gizmo (fallback: reset camera if no selection) |
| `S` | Scale gizmo |
| `X / Y / Z` | Constrain to axis (while gizmo active) |
| `Shift` (hold) | Snap during gizmo confirm |
| `0-9 . -` (type) | Numeric type-in for exact delta |
| `Enter` | Apply gizmo |
| `Esc` | Cancel gizmo |

## Selection

| Key | Action |
|---|---|
| LMB | Select item under cursor |
| `Cmd/Ctrl+LMB` | Toggle item in multi-select set |
| `Shift+LMB` | Range select (when both endpoints are same category) |
| `A` | Select all (sources + listeners + objects) |
| `Alt+A` | Deselect all |
| `H` | Hide selection |
| `Alt+H` | Unhide all |
| `/` | Toggle isolate mode |
| `B` | Arm box-select (drag-rectangle on next press) |
| `Esc` | Clear selection |

## History / Undo

| Key | Action |
|---|---|
| `Cmd/Ctrl+Z` | Undo |
| `Cmd/Ctrl+Shift+Z` | Redo |
| `Cmd/Ctrl+Y` | Redo (alternate) |

The history ring buffer is bounded; the status bar surfaces the last
operation with an undo affordance.

## Snap

`Shift` toggles snap mode while moving with a gizmo. The `SnapConfig` field
selects between `Grid` (configurable size, default 0.25 m), `Surface`
(raycast onto mesh), and `Angle` (rotation in 15° increments).

## Camera Views

| Key | View |
|---|---|
| `Num 0` | Perspective |
| `Num 7` | Top |
| `Num 5` | Isometric |
| `[` | Ringside A preset |
| `]` | Ringside B preset |
| `Ctrl+1` | Front |
| `Ctrl+3` | Side |
| `Home` | Frame scene |

Trackpad: pinch zooms via `egui::InputState::zoom_delta()`; two-finger
scroll pans via `smooth_scroll_delta`.

## Outliner

Each row in the outliner panel exposes per-row controls (toggleable
globally via `vp.show_visibility_icons`):

- Eye icon — toggle row visibility
- Lock icon — block accidental click-select
- Locked rows render with `weak()` styling

Multi-row selection routes through the same `SelectionSet` used by viewport
multi-select, so `H` / `/` keys honour outliner-picked items.

## Properties Polish

Numeric DragValue fields accept arithmetic expressions via
`ui::expr::evaluate_expression`. Supported syntax: `+`, `-`, `*`, `/`, `^`,
unary `±`, parens, scientific notation, constants (`pi`, `e`, `tau`), and
functions (`sin`, `cos`, `tan`, `abs`, `sqrt`, `ln`, `log`, `floor`, `ceil`,
`round`). Example: `2*pi`, `sqrt(9)`, `1+sin(0)`.

## Right-Click Context Menus

Right-clicking the viewport opens a selection-aware menu:

- No selection: Add Source / Listener / Partition Wall / Platform · Reset Camera
- Source / Listener / Object: Focus · Delete
- Robot / RobotLink: Focus

## Pie Menu

Hold `Q` to open the radial pie menu at the cursor. Eight wedges (clockwise
from North): Frame Selected → Reset Camera → Toggle Grid → Toggle Shaded →
Top View → Place Source → Place Listener → Run Simulation. Release `Q` on a
wedge to commit; release on the centre dead-zone to cancel.

## Onboarding

A first-run modal walks new users through Load Model → Place Source → Place
Listener → Run Sim → View Results. `F1` reopens the cheat sheet at any time.
The "Don't show again" choice persists across launches.

## Status Bar

The viewport status bar always shows:

- Current mode label (`Select`, `Place Source`, `Place Listener`)
- A `next_step_hint` for the current mode + selection
- Held modifier glyphs (⇧ ⌃ ⌥ ⌘)
- An `action_hint` showing the last history-affecting operation + undo affordance

## Sensible Defaults

- `auto_fit_camera(aabb)` runs after each scene load so models always appear
  in-frame
- Default ambient lighting (Lambert + drop-shadow) on shaded objects
- Numeric inspector fields show units (m, °, kg) next to values
- "Welcome" empty-state hint in the viewport when no scene is loaded

## Industry Audit Cross-Reference

| Pattern | Cinema 4D | Blender | SolidWorks | EchoMap (today) |
|---|---|---|---|---|
| Undo/redo | ✓ | ✓ | ✓ | ✓ (named ops, ring buffer) |
| Command palette | ✓ (Cmd+E) | ✓ (F3) | ✓ (S, command search) | ✓ (Cmd/Ctrl+K, fuzzy) |
| Transform gizmos | ✓ | ✓ (G/R/S) | ✓ (triad) | ✓ (G/R/S + X/Y/Z lock) |
| Snap system | ✓ | ✓ | ✓ | ✓ (grid / surface / angle) |
| Customizable hotkeys | ✓ | ✓ | ✓ | ✓ (JSON keymap, XDG_CONFIG_HOME) |
| Right-click context menus | ✓ | ✓ | ✓ | ✓ (selection-aware) |
| Multi-select + box-select | ✓ | ✓ | ✓ | ✓ (Ctrl-click + B + A/Alt+A) |
| Outliner drag/reparent | ✓ | ✓ | ✓ | ✓ (lock + eye icons) |
| Properties polish | Drag + expressions | Drag + drivers | Equations field | ✓ (drag + arithmetic eval) |
| Quad-view | ✓ | ✓ (Ctrl+Alt+Q) | ✓ | ✓ (Ctrl+Alt+Q) |
| Trackpad gestures | ✓ | ✓ | Limited | ✓ (pinch zoom + scroll pan) |
| Onboarding overlay | ✓ | ✓ | ✓ | ✓ (first-run tour + F1) |
| Sensible defaults | Auto-frame | Auto-frame | Fit view | ✓ (auto-fit camera) |
| Status bar hints | ✓ | ✓ | ✓ | ✓ (mode + next-step + modifiers) |
| Pie menu | — | ✓ | — | ✓ (hold-Q radial) |
