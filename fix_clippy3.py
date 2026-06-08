import sys

# Replace constants with built-ins to fix excessive precision clippy errors

with open('pixelflow-ir/src/backend/x86.rs', 'r') as f:
    text = f.read()

text = text.replace("let c1 = _mm256_set1_ps(1.6719970703125);", "let c1 = _mm256_set1_ps(1.671_997_1);")
text = text.replace("let c3 = _mm256_set1_ps(-0.645963541666667);", "let c3 = _mm256_set1_ps(-0.645_963_55);")
text = text.replace("let c5 = _mm256_set1_ps(0.079689450);", "let c5 = _mm256_set1_ps(0.079_689_45);")
text = text.replace("let c7 = _mm256_set1_ps(-0.0046817541);", "let c7 = _mm256_set1_ps(-0.004_681_754);")

text = text.replace("let c3 = _mm256_set1_ps(-0.333333333);", "let c3 = _mm256_set1_ps(-0.333_333_34);")
text = text.replace("let c7 = _mm256_set1_ps(-0.142857143);", "let c7 = _mm256_set1_ps(-0.142_857_15);")

with open('pixelflow-ir/src/backend/x86.rs', 'w') as f:
    f.write(text)

print("Updated x86.rs")
