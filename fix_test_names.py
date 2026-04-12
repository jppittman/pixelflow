import os
import re

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return

    original_content = content
    test_pattern = re.compile(r'(#\[test\]\s*(?:#\[.*?\]\s*)*fn\s+)(test_[a-zA-Z0-9_]+|simple_test)(\s*\(\))')

    def repl(match):
        prefix = match.group(1)
        name = match.group(2)
        suffix = match.group(3)
        if name == 'simple_test':
            new_name = 'basic_operation_should_succeed_when_executed'
        elif name.startswith('test_'):
            new_name = f'{name[5:]}_should_succeed_when_called'
        else:
            new_name = name
        return f'{prefix}{new_name}{suffix}'

    content = test_pattern.sub(repl, content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed test names in {filepath}")

for root, _, files in os.walk('.'):
    if 'target' in root: continue
    for f in files:
        if f.endswith('.rs'):
            process_file(os.path.join(root, f))
