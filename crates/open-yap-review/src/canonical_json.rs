//! Canonical JSON that matches the Python review store byte for byte
//!
//! Python writes `json.dumps(value, sort_keys=True, separators=(",", ":"),
//! ensure_ascii=True) + "\n"` and hashes the same text. Every hashed identity
//! in the session (event hashes, overlay hash, proposal content hash) depends
//! on this exact encoding, including the `repr` spelling of floats

use std::fmt::Write as _;

use serde_json::{Number, Value};
use sha2::{Digest, Sha256};

/// Encode a JSON value as canonical text with a trailing newline
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out.push('\n');
    out
}

/// Return the lowercase SHA-256 hex digest of the canonical encoding
pub fn sha256_json(value: &Value) -> String {
    sha256_hex(canonical_json(value).as_bytes())
}

/// Return the lowercase SHA-256 hex digest of raw bytes
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(out, number),
        Value::String(text) => write_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // sort explicitly because feature unification can turn on preserve_order
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            out.push('{');
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_string(out, key);
                out.push(':');
                write_value(out, item);
            }
            out.push('}');
        }
    }
}

fn write_number(out: &mut String, number: &Number) {
    if let Some(integer) = number.as_u64() {
        let _ = write!(out, "{integer}");
        return;
    }
    if let Some(integer) = number.as_i64() {
        let _ = write!(out, "{integer}");
        return;
    }
    let float = number.as_f64().unwrap_or(f64::NAN);
    out.push_str(&python_float_repr(float));
}

/// Spell a float the way Python `repr` does
///
/// Both Rust and Python pick the shortest digit string that round trips, so
/// only the layout differs: Python switches to scientific notation when the
/// decimal exponent is below -4 or at least 16 and always keeps a fractional
/// part on integral values
pub fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_owned();
    }

    let scientific = shortest_scientific(value);
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();

    let mut out = String::from(sign);
    if !(-4..16).contains(&exponent) {
        let (first, rest) = digits.split_at(1);
        out.push_str(first);
        if !rest.is_empty() {
            out.push('.');
            out.push_str(rest);
        }
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        let _ = write!(out, "e{exponent_sign}{:02}", exponent.unsigned_abs());
        return out;
    }

    if exponent < 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n(
            '0',
            exponent.unsigned_abs() as usize - 1,
        ));
        out.push_str(&digits);
        return out;
    }

    let integer_len = exponent as usize + 1;
    if digits.len() <= integer_len {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', integer_len - digits.len()));
        out.push_str(".0");
        return out;
    }
    let (integer, fraction) = digits.split_at(integer_len);
    out.push_str(integer);
    out.push('.');
    out.push_str(fraction);
    out
}

/// Return the shortest round-trip digits in scientific form, ties rounded to even
///
/// Rust's shortest mode rounds an exact decimal tie up, while Python's dtoa
/// rounds it to the even digit. The correctly rounded string with the same
/// digit count uses round-half-even, so prefer it whenever it still round trips
fn shortest_scientific(value: f64) -> String {
    let shortest = format!("{value:e}");
    let mantissa = shortest
        .split_once('e')
        .map_or(shortest.as_str(), |(mantissa, _)| mantissa);
    let digit_count = mantissa.chars().filter(char::is_ascii_digit).count();
    let exact = format!(
        "{value:.precision$e}",
        precision = digit_count.saturating_sub(1)
    );
    if exact.parse::<f64>() == Ok(value) {
        exact
    } else {
        shortest
    }
}

fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(character),
            _ => {
                let mut units = [0u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{canonical_json, python_float_repr, sha256_json};

    #[test]
    fn escapes_match_python_ensure_ascii() {
        let value = json!({"s": "é \u{7f}/\u{1f}\"\\\n\r\t\u{08}\u{0c}😀"});

        assert_eq!(
            canonical_json(&value),
            "{\"s\":\"\\u00e9 \\u007f/\\u001f\\\"\\\\\\n\\r\\t\\b\\f\\ud83d\\ude00\"}\n"
        );
    }

    #[test]
    fn keys_sort_recursively_without_spaces() {
        let value = json!({"b": [1, {"z": true, "a": null}], "a": "x"});

        assert_eq!(
            canonical_json(&value),
            "{\"a\":\"x\",\"b\":[1,{\"a\":null,\"z\":true}]}\n"
        );
    }

    #[test]
    fn floats_match_python_json_dumps() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"a":1e-05,"b":1e+16,"c":30.0,"d":0.30000000000000004,"f":0.0,"i":7}"#,
        )
        .unwrap();

        assert_eq!(
            canonical_json(&value),
            "{\"a\":1e-05,\"b\":1e+16,\"c\":30.0,\"d\":0.30000000000000004,\"f\":0.0,\"i\":7}\n"
        );
    }

    #[test]
    fn float_repr_table_matches_python() {
        // expected strings come from python3 -c 'print(repr(x))'
        let cases: &[(f64, &str)] = &[
            (1e16, "1e+16"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (5e-324, "5e-324"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (0.02, "0.02"),
            (30.0, "30.0"),
            (4808.04, "4808.04"),
            (0.5500000000001819, "0.5500000000001819"),
            (1.0, "1.0"),
            (-1.5, "-1.5"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e15, "1000000000000000.0"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (2.2250738585072014e-308, "2.2250738585072014e-308"),
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (100.0, "100.0"),
            (1e22, "1e+22"),
            (1.5e-7, "1.5e-07"),
            (-2.5e-5, "-2.5e-05"),
            (0.00012345, "0.00012345"),
            (12345.678, "12345.678"),
            (4.35, "4.35"),
            (6.94, "6.94"),
            (2.0f64.powi(53), "9007199254740992.0"),
            (1e100, "1e+100"),
            (1059438285926254.2, "1059438285926254.2"),
            (-1e-100, "-1e-100"),
        ];

        for (value, expected) in cases {
            assert_eq!(python_float_repr(*value), *expected, "value {value:e}");
        }
    }

    #[test]
    fn integers_and_floats_stay_distinct() {
        let integer: serde_json::Value =
            serde_json::from_str("[5, 5.0, -3, 18446744073709551615]").unwrap();

        assert_eq!(
            canonical_json(&integer),
            "[5,5.0,-3,18446744073709551615]\n"
        );
    }

    #[test]
    fn hash_includes_trailing_newline() {
        assert_eq!(
            sha256_json(&json!({})),
            "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
        );
    }
}
