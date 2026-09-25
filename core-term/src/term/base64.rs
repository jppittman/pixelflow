// src/term/base64.rs

//! Standard base64 (RFC 4648 §4), the encoding OSC 52 carries text in.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes `data` as padded base64.
pub(crate) fn encode(data: &[u8]) -> String {
    let mut encoded = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let mut bytes = [0u8; 4];
        bytes[1..=chunk.len()].copy_from_slice(chunk);
        let bits = u32::from_be_bytes(bytes);
        // A chunk of n bytes fills n + 1 digits; the rest of the quad is padding.
        for digit in 0..4 {
            let sextet = (bits >> (18 - 6 * digit)) & 0x3f;
            encoded.push(match digit <= chunk.len() {
                true => char::from(ALPHABET[sextet as usize]),
                false => '=',
            });
        }
    }
    encoded
}

/// Decodes standard base64 (RFC 4648 §4), padding optional. `None` for any
/// character outside the alphabet.
pub(crate) fn decode(encoded: &str) -> Option<Vec<u8>> {
    let sextet = |c: u8| -> Option<u32> {
        Some(u32::from(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        }))
    };
    let digits = encoded.trim_end_matches('=').as_bytes();
    let mut decoded = Vec::with_capacity(digits.len() * 3 / 4);
    for chunk in digits.chunks(4) {
        let mut bits = 0u32;
        for &digit in chunk {
            bits = (bits << 6) | sextet(digit)?;
        }
        // A chunk of n digits carries 6n bits, the top 8(n-1) of them data.
        let data_bytes = chunk.len().saturating_sub(1);
        bits <<= 6 * (4 - chunk.len());
        decoded.extend_from_slice(&bits.to_be_bytes()[1..=data_bytes]);
    }
    Some(decoded)
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};

    #[test]
    fn encoding_matches_the_rfc_4648_test_vectors() {
        let vectors = [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];
        for (plain, encoded) in vectors {
            assert_eq!(encode(plain.as_bytes()), encoded, "encoding {plain:?}");
            assert_eq!(
                decode(encoded).as_deref(),
                Some(plain.as_bytes()),
                "decoding {encoded:?}"
            );
        }
    }

    #[test]
    fn decoding_refuses_characters_outside_the_alphabet() {
        assert_eq!(decode("Zm9v!"), None);
    }
}
