import re

filepath = 'pixelflow-core/src/algebra.rs'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace assert_eq!(..., true/false) with idiomatic assert!(...)
# We can do this manually to be safe since there's only a few lines.
replacements = [
    ("assert_eq!(bool::zero(), false);", 'assert!(!bool::zero(), "bool::zero() should be false");'),
    ("assert_eq!(bool::one(), true);", 'assert!(bool::one(), "bool::one() should be true");'),
    ("assert_eq!(false.add(false), false);", 'assert!(!false.add(false), "false.add(false) should be false");'),
    ("assert_eq!(false.add(true), true);", 'assert!(false.add(true), "false.add(true) should be true");'),
    ("assert_eq!(true.add(false), true);", 'assert!(true.add(false), "true.add(false) should be true");'),
    ("assert_eq!(true.add(true), true);", 'assert!(true.add(true), "true.add(true) should be true");'),
    ("assert_eq!(false.mul(false), false);", 'assert!(!false.mul(false), "false.mul(false) should be false");'),
    ("assert_eq!(false.mul(true), false);", 'assert!(!false.mul(true), "false.mul(true) should be false");'),
    ("assert_eq!(true.mul(true), true);", 'assert!(true.mul(true), "true.mul(true) should be true");'),
    ("assert_eq!(true.neg(), false);", 'assert!(!true.neg(), "true.neg() should be false");'),
    ("assert_eq!(false.neg(), true);", 'assert!(false.neg(), "false.neg() should be true");'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Asserts updated in {filepath}")
