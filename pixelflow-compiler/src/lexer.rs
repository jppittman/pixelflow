#![allow(dead_code)]
#![allow(clippy::all)]
//! # Lexer
//!
//! Tokenization for the kernel DSL.
//!
//! ## Design Note
//!
//! We delegate actual tokenization to `syn` and the Rust lexer - there's no
//! benefit to re-implementing tokenization for a DSL embedded in Rust syntax.
//!
//! This module exists as a conceptual placeholder and for potential future
//! preprocessing (e.g., custom operators, string interpolation, etc.).
//!
//! ## Token Classification
//!
//! The kernel DSL recognizes these token classes:
//!
//! | Class      | Examples              | Notes                           |
//! |------------|-----------------------|---------------------------------|
//! | Intrinsic  | X, Y, Z, W            | Coordinate variables            |
//! | Ident      | cx, radius, dx        | User identifiers                |
//! | Literal    | 1.0, 2.5f32, 0        | Numeric constants               |
//! | Operator   | +, -, *, /, %         | Arithmetic operators            |
//! | Punct      | (, ), {, }, ;, :      | Delimiters and separators       |
//! | Keyword    | let                   | Binding introduction            |

use proc_macro2::TokenStream;

/// Token classification for semantic purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenClass {
    /// Intrinsic coordinate (X, Y, Z, W).
    Intrinsic,
    /// User-defined identifier.
    Ident,
    /// Numeric literal.
    Literal,
    /// Operator (+, -, etc.).
    Operator,
    /// Punctuation.
    Punct,
    /// Keyword (let).
    Keyword,
}

/// Classify an identifier token.
pub fn classify_ident(name: &str) -> TokenClass {
    match name {
        "X" | "Y" | "Z" | "W" => TokenClass::Intrinsic,
        "let" => TokenClass::Keyword,
        _ => TokenClass::Ident,
    }
}

/// The "lexer" is effectively a pass-through to syn.
/// This function exists for API consistency and future expansion.
pub fn lex(input: TokenStream) -> TokenStream {
    // Currently a no-op - syn handles tokenization
    // Future: could add preprocessing here
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_intrinsics() {
        assert_eq!(classify_ident("X"), TokenClass::Intrinsic);
        assert_eq!(classify_ident("Y"), TokenClass::Intrinsic);
        assert_eq!(classify_ident("Z"), TokenClass::Intrinsic);
        assert_eq!(classify_ident("W"), TokenClass::Intrinsic);
    }

    #[test]
    fn classify_user_idents() {
        assert_eq!(classify_ident("cx"), TokenClass::Ident);
        assert_eq!(classify_ident("radius"), TokenClass::Ident);
        assert_eq!(classify_ident("my_var"), TokenClass::Ident);
    }

    #[test]
    fn classify_keywords() {
        assert_eq!(classify_ident("let"), TokenClass::Keyword);
    }
}
