//! Share codes: bytes deflated and written in URL-safe base64, so a robot
//! fits in a link (a champion's 32 KB of JSON makes about 4 KB of code).

use miniz_oxide::deflate::compress_to_vec;
use miniz_oxide::inflate::decompress_to_vec_with_limit;

/// The most bytes a code from elsewhere may inflate to (see [`decode`]). A
/// code can come from anywhere, and a short one could otherwise inflate to
/// fill memory.
pub const MAX_INFLATED: usize = 1 << 20;

/// miniz's strongest deflate level.
const LEVEL: u8 = 10;

/// URL-safe base64 (RFC 4648 section 5): nothing in it needs escaping in a
/// link.
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// `bytes` as a code: deflated, in URL-safe base64 without padding.
pub fn encode(bytes: &[u8]) -> String {
    let deflated = compress_to_vec(bytes, LEVEL);
    let mut code = String::with_capacity(deflated.len().div_ceil(3) * 4);
    for chunk in deflated.chunks(3) {
        let group = chunk.iter().enumerate().fold(0u32, |group, (i, &byte)| {
            group | u32::from(byte) << (16 - 8 * i)
        });
        // One byte makes two characters, two make three, three make four.
        for i in 0..=chunk.len() {
            code.push(char::from(ALPHABET[(group >> (18 - 6 * i) & 63) as usize]));
        }
    }
    code
}

/// The bytes a code holds, refusing more than [`MAX_INFLATED`].
pub fn decode(code: &str) -> Result<Vec<u8>, String> {
    decode_within(code, MAX_INFLATED)
}

/// The bytes a code holds, refusing more than `limit`. Whitespace (a code
/// broken over lines) is skipped, padding ends the code, and the standard
/// base64 alphabet reads as well as the URL-safe one.
pub fn decode_within(code: &str, limit: usize) -> Result<Vec<u8>, String> {
    let mut deflated = Vec::with_capacity(code.len() / 4 * 3 + 2);
    let mut bits = 0u32;
    let mut held = 0;
    for c in code.chars() {
        let value = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '-' | '+' => 62,
            '_' | '/' => 63,
            '=' => break,
            c if c.is_whitespace() => continue,
            c => return Err(format!("'{c}' is not part of a share code")),
        };
        bits = (bits << 6 | value) & 0xFFFF;
        held += 6;
        if held >= 8 {
            held -= 8;
            deflated.push((bits >> held) as u8);
        }
    }
    decompress_to_vec_with_limit(&deflated, limit).map_err(|err| match err.status {
        miniz_oxide::inflate::TINFLStatus::HasMoreOutput => {
            format!("the code holds more than {limit} bytes")
        }
        status => format!("the code is cut short or damaged ({status:?})"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_and_use_only_link_safe_characters() {
        for text in ["", "a", "ab", "abc", "abcd", "{\"robot\": [1, 2, 3]}"] {
            let code = encode(text.as_bytes());
            assert!(
                code.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
                "{code}"
            );
            assert_eq!(decode(&code).unwrap(), text.as_bytes());
        }
        // Repetitive JSON shrinks a lot.
        let json = "{\"weight\": 0.5, \"enabled\": true}, ".repeat(400);
        let code = encode(json.as_bytes());
        assert!(
            code.len() * 10 < json.len(),
            "{} of {}",
            code.len(),
            json.len()
        );
        assert_eq!(decode(&code).unwrap(), json.as_bytes());
    }

    #[test]
    fn a_code_reads_through_line_breaks_padding_and_the_standard_alphabet() {
        let bytes: Vec<u8> = (0..=255).collect();
        let code = encode(&bytes);
        let broken: String = code
            .chars()
            .enumerate()
            .flat_map(|(i, c)| (i % 40 == 39).then_some('\n').into_iter().chain([c]))
            .collect();
        assert_eq!(decode(&broken).unwrap(), bytes);
        let standard = code.replace('-', "+").replace('_', "/") + "==";
        assert_eq!(decode(&standard).unwrap(), bytes);
    }

    #[test]
    fn damaged_codes_and_bombs_are_refused() {
        let code = encode(b"a robot that walks");
        assert!(decode(&code[..code.len() / 2]).is_err());
        assert!(decode("not a code!").is_err());
        assert!(decode("").is_err());
        // Two megabytes of zeros deflate to a few kilobytes but inflate past
        // the limit.
        let bomb = encode(&vec![0; 2 * MAX_INFLATED]);
        assert!(bomb.len() < 8 * 1024, "{}", bomb.len());
        assert!(decode(&bomb).unwrap_err().contains("more than"));
        // A caller that trusts the code more can raise the limit.
        assert_eq!(
            decode_within(&bomb, 4 * MAX_INFLATED).unwrap().len(),
            2 * MAX_INFLATED
        );
    }
}
