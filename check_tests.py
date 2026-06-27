import os
import re

def check_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Simple regex to find test functions
    tests = re.finditer(r'#\[test\]\s*(#\[.*?\]\s*)*fn\s+(\w+)\s*\(\)\s*\{([^}]*)\}', content, re.MULTILINE)

    for t in tests:
        body = t.group(3)
        if 'assert' not in body and 'panic' not in body and 'unwrap' not in body and 'expect' not in body and 'match' not in body:
            print(f"{path}: {t.group(2)} has no checks.")

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.rs'):
            check_file(os.path.join(root, file))

for root, _, files in os.walk('.'):
    if 'src' in root and '/target/' not in root:
        for file in files:
            if file.endswith('.rs'):
                check_file(os.path.join(root, file))
