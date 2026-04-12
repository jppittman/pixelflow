import os
import re

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return

    original_content = content
    test_pattern = re.compile(r'#\[test\]\s*(?:#\[.*?\]\s*)*fn\s+[a-zA-Z0-9_]+\s*\(\)\s*\{', re.DOTALL)

    new_content = ""
    last_end = 0

    for match in test_pattern.finditer(content):
        start_idx = match.end()
        brace_count = 1
        i = start_idx
        while i < len(content) and brace_count > 0:
            if content[i] == '{': brace_count += 1
            elif content[i] == '}': brace_count -= 1
            i += 1

        test_body = content[start_idx:i-1]

        new_test_body = test_body.replace('.unwrap()', '.expect("Expected value but got None/Err")')

        new_content += content[last_end:start_idx] + new_test_body
        last_end = i - 1

    new_content += content[last_end:]

    if new_content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed unwraps in {filepath}")

for root, _, files in os.walk('.'):
    if 'target' in root: continue
    for f in files:
        if f.endswith('.rs'):
            process_file(os.path.join(root, f))
