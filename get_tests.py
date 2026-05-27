import re

with open('core-term/src/term/tests.rs', 'r', encoding='utf-8') as f:
    content = f.read()

test_pattern = re.compile(r'^\s*(#\[test\]\n\s*fn\s+([a-zA-Z0-9_]+)\(\)\s*\{)', re.MULTILINE)
matches = list(test_pattern.finditer(content))
for match in matches:
    print(match.group(2))
