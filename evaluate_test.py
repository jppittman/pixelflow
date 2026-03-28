import re
import sys
import glob

def evaluate(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    tests = re.findall(r'#\[test\]\s*(?:#\[.*\]\s*)*fn\s+([a-zA-Z0-9_]+)\s*\(\)\s*\{(.*?)\n\}', content, re.DOTALL)

    violations = []

    for test_name, test_body in tests:
        if test_name.startswith('test_'):
            violations.append(f"{test_name} has generic test_ prefix")

        if '.unwrap()' in test_body and '.unwrap_err()' not in test_body:
             violations.append(f"{test_name} uses .unwrap()")

    if violations:
        print(f"File: {filepath}")
        for v in violations:
            print(f"  - {v}")
        return True
    return False

files = glob.glob('core-term/src/**/*.rs', recursive=True)
found = False
for f in files:
    if 'test' in f:
        if evaluate(f):
            found = True
if not found:
    print("No violations found!")
