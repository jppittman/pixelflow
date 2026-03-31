import os
import re

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.rs'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()

            # Very basic parsing to find test functions
            # This handles single-level braces well enough for simple tests
            matches = re.finditer(r'#\[test\]\s*(?:#\[.*?\]\s*)*fn\s+(\w+)\s*\(\)\s*\{([^}]*)\}', content, re.DOTALL)
            for m in matches:
                name = m.group(1)
                body = m.group(2)
                if 'assert' not in body and 'expect' not in body and 'unwrap' not in body:
                    print(f"Empty test: {path} -> {name}")
