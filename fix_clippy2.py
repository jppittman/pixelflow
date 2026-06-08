import sys

# Replace constants with built-ins to fix excessive precision clippy errors

with open('pixelflow-ir/src/backend/x86.rs', 'r') as f:
    text = f.read()

text = text.replace("let c1 = _mm_set1_ps(1.671_997_1);", "let c1 = _mm_set1_ps(1.6719971);")
text = text.replace("let c3 = _mm_set1_ps(-0.645_963_55);", "let c3 = _mm_set1_ps(-0.64596355);")
text = text.replace("let c5 = _mm_set1_ps(0.079_689_45);", "let c5 = _mm_set1_ps(0.07968945);")
text = text.replace("let c7 = _mm_set1_ps(-0.004_681_754);", "let c7 = _mm_set1_ps(-0.004681754);")

text = text.replace("let c3 = _mm_set1_ps(-0.333_333_34);", "let c3 = _mm_set1_ps(-0.33333334);")
text = text.replace("let c7 = _mm_set1_ps(-0.142_857_15);", "let c7 = _mm_set1_ps(-0.14285715);")

with open('pixelflow-ir/src/backend/x86.rs', 'w') as f:
    f.write(text)

print("Updated x86.rs")
