import os
import re

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return

    original_content = content
    content = re.sub(r'assert_eq!\(([^,]+),\s*true\)', r'assert!(\1)', content)
    content = re.sub(r'assert_eq!\(true,\s*([^,]+)\)', r'assert!(\1)', content)
    content = re.sub(r'assert_eq!\(([^,]+),\s*false\)', r'assert!(!(\1))', content)
    content = re.sub(r'assert_eq!\(false,\s*([^,]+)\)', r'assert!(!(\1))', content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed booleans in {filepath}")

for root, _, files in os.walk('.'):
    if 'target' in root: continue
    for f in files:
        if f.endswith('.rs'):
            process_file(os.path.join(root, f))
