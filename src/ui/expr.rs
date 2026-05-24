//! Tiny arithmetic expression evaluator for DragValue / numeric input
//! fields.
//!
//! Lets the user type things like `2 * 3.14`, `1 + sin(0)`, or `(3 + 5)/2`
//! directly into a position / frequency / power field instead of pulling
//! out a calculator. Returns `Ok(f64)` for a valid expression, `Err` with
//! a one-line diagnostic for bad input.
//!
//! Grammar (recursive descent, left-associative):
//!
//! ```text
//! expr   := term  (('+' | '-') term)*
//! term   := factor (('*' | '/') factor)*
//! factor := unary ('^' factor)?               // right-assoc
//! unary  := ('+' | '-')? atom
//! atom   := number | ident '(' expr ')' | ident | '(' expr ')'
//! ```
//!
//! Supported identifiers: `pi`, `e`, plus single-arg functions
//! `sin`, `cos`, `tan`, `abs`, `sqrt`, `ln`, `log` (base-10).

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprError {
    pub msg: String,
    pub pos: usize,
}

impl std::fmt::Display for ExprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expr error at {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for ExprError {}

/// Evaluate a numeric expression. Whitespace is ignored. Empty / whitespace
/// input is an error so the caller can treat it like "no change".
pub fn evaluate_expression(input: &str) -> Result<f64, ExprError> {
    let mut p = Parser::new(input);
    let v = p.parse_expr()?;
    p.skip_ws();
    if p.pos < p.bytes.len() {
        return Err(p.err("unexpected trailing input"));
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn err(&self, msg: &str) -> ExprError {
        ExprError {
            msg: msg.to_string(),
            pos: self.pos,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn eat(&mut self, ch: u8) -> bool {
        self.skip_ws();
        if self.peek() == Some(ch) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> Result<f64, ExprError> {
        let mut lhs = self.parse_term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs += rhs;
                }
                Some(b'-') => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs -= rhs;
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<f64, ExprError> {
        let mut lhs = self.parse_factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    lhs *= rhs;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    if rhs == 0.0 {
                        return Err(self.err("division by zero"));
                    }
                    lhs /= rhs;
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_factor(&mut self) -> Result<f64, ExprError> {
        let lhs = self.parse_unary()?;
        self.skip_ws();
        if self.peek() == Some(b'^') {
            self.pos += 1;
            let rhs = self.parse_factor()?; // right-assoc
            Ok(lhs.powf(rhs))
        } else {
            Ok(lhs)
        }
    }

    fn parse_unary(&mut self) -> Result<f64, ExprError> {
        self.skip_ws();
        match self.peek() {
            Some(b'+') => {
                self.pos += 1;
                self.parse_unary()
            }
            Some(b'-') => {
                self.pos += 1;
                let v = self.parse_unary()?;
                Ok(-v)
            }
            _ => self.parse_atom(),
        }
    }

    fn parse_atom(&mut self) -> Result<f64, ExprError> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let v = self.parse_expr()?;
                if !self.eat(b')') {
                    return Err(self.err("expected ')'"));
                }
                Ok(v)
            }
            Some(b) if b.is_ascii_digit() || b == b'.' => self.parse_number(),
            Some(b) if b.is_ascii_alphabetic() || b == b'_' => self.parse_ident_or_call(),
            Some(_) => Err(self.err("unexpected character")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn parse_number(&mut self) -> Result<f64, ExprError> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || b == b'.' {
                self.pos += 1;
            } else {
                break;
            }
        }
        // Scientific notation: optional e+/-digits.
        if let Some(b) = self.peek() {
            if b == b'e' || b == b'E' {
                let save = self.pos;
                self.pos += 1;
                if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                    self.pos += 1;
                }
                if matches!(self.peek(), Some(b) if b.is_ascii_digit()) {
                    while let Some(b) = self.peek() {
                        if b.is_ascii_digit() {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                } else {
                    // Not a valid exponent — roll back.
                    self.pos = save;
                }
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("invalid utf8 in number"))?;
        s.parse::<f64>().map_err(|_| self.err("invalid number"))
    }

    fn parse_ident_or_call(&mut self) -> Result<f64, ExprError> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let name = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("invalid utf8 in identifier"))?
            .to_ascii_lowercase();
        self.skip_ws();
        if self.peek() == Some(b'(') {
            self.pos += 1;
            let arg = self.parse_expr()?;
            if !self.eat(b')') {
                return Err(self.err("expected ')'"));
            }
            apply_fn(&name, arg).ok_or_else(|| ExprError {
                msg: format!("unknown function: {name}"),
                pos: start,
            })
        } else {
            constant(&name).ok_or_else(|| ExprError {
                msg: format!("unknown identifier: {name}"),
                pos: start,
            })
        }
    }
}

fn constant(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        "e" => Some(std::f64::consts::E),
        "tau" => Some(std::f64::consts::TAU),
        _ => None,
    }
}

fn apply_fn(name: &str, x: f64) -> Option<f64> {
    Some(match name {
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "abs" => x.abs(),
        "sqrt" => x.sqrt(),
        "ln" => x.ln(),
        "log" => x.log10(),
        "floor" => x.floor(),
        "ceil" => x.ceil(),
        "round" => x.round(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str, expected: f64) {
        let got = evaluate_expression(s).expect(s);
        assert!((got - expected).abs() < 1e-9, "{s} → {got} ≠ {expected}");
    }

    fn err(s: &str) {
        let r = evaluate_expression(s);
        assert!(r.is_err(), "expected error for {s:?}, got {r:?}");
    }

    #[test]
    #[allow(clippy::approx_constant)] // literal "3.14" is the parser input we're testing
    fn expr_eval_plain_number() {
        ok("42", 42.0);
        ok("3.14", 3.14);
        ok(".5", 0.5);
        ok("1e3", 1000.0);
        ok("2.5e-2", 0.025);
    }

    #[test]
    fn expr_eval_addition_subtraction() {
        ok("1+2", 3.0);
        ok("10 - 3", 7.0);
        ok("1 + 2 - 3", 0.0);
    }

    #[test]
    fn expr_eval_multiplication_division() {
        ok("2*3", 6.0);
        ok("6/2", 3.0);
        ok("2 * 3 + 4", 10.0);
        ok("2 + 3 * 4", 14.0);
    }

    #[test]
    fn expr_eval_unary_minus() {
        ok("-3", -3.0);
        ok("--3", 3.0);
        ok("-(2+3)", -5.0);
    }

    #[test]
    fn expr_eval_parens() {
        ok("(2 + 3) * 4", 20.0);
        ok("((1+2)*(3+4))", 21.0);
    }

    #[test]
    fn expr_eval_pow_right_associative() {
        // 2^3^2 = 2^(3^2) = 2^9 = 512
        ok("2^3^2", 512.0);
        ok("2^3", 8.0);
    }

    #[test]
    fn expr_eval_constants() {
        ok("pi", std::f64::consts::PI);
        ok("e", std::f64::consts::E);
        ok("2*pi", std::f64::consts::TAU);
    }

    #[test]
    fn expr_eval_functions() {
        ok("sin(0)", 0.0);
        ok("cos(0)", 1.0);
        ok("abs(-5)", 5.0);
        ok("sqrt(9)", 3.0);
    }

    #[test]
    fn expr_eval_functions_nested() {
        ok("sqrt(abs(-16))", 4.0);
        ok("1 + sin(0)", 1.0);
    }

    #[test]
    fn expr_eval_whitespace_tolerated() {
        ok("  1  +  2  ", 3.0);
        ok("\t3\n*\n4", 12.0);
    }

    #[test]
    fn expr_eval_rejects_empty() {
        err("");
        err("   ");
    }

    #[test]
    fn expr_eval_rejects_garbage() {
        err("@");
        err("1 +");
        err("(1 + 2");
        err("foo(1)");
    }

    #[test]
    fn expr_eval_rejects_division_by_zero() {
        err("1/0");
        err("5/(2-2)");
    }

    #[test]
    fn expr_eval_case_insensitive_idents() {
        ok("PI", std::f64::consts::PI);
        ok("SIN(0)", 0.0);
    }

    #[test]
    fn expr_eval_trailing_input_rejected() {
        err("1 2");
        err("1+2 foo");
    }
}
