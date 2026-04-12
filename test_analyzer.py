import os
import re

def parse_tests(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    test_pattern = re.compile(r'#\[test\]\s*(?:#\[.*?\]\s*)*fn\s+([a-zA-Z0-9_]+)\s*\(\)\s*\{', re.DOTALL)

    tests = []
    for match in test_pattern.finditer(content):
        func_name = match.group(1)
        start_idx = match.end()
        brace_count = 1
        i = start_idx
        while i < len(content) and brace_count > 0:
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
            i += 1

        body = content[start_idx:i-1]
        full_match_start = match.start()
        tests.append({
            'name': func_name,
            'body': body,
            'full_start': full_match_start,
            'full_end': i,
            'content': content[full_match_start:i]
        })

    return tests

def collect_modifications(filepath):
    try:
        tests = parse_tests(filepath)
    except Exception:
        return None

    if not tests:
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()

    deletions = []

    for test in tests:
        name = test['name']
        body = test['body']

        has_assert = 'assert' in body or 'expect' in body or 'panic' in body or 'unwrap' in body
        only_asserts_true = bool(re.match(r'^\s*assert!\(\s*true\s*\)\s*;\s*$', body.strip()))
        empty_body = not body.strip()

        if not has_assert or only_asserts_true or empty_body:
            is_noise = empty_body or only_asserts_true
            if is_noise:
                print(f"Deleting pure noise test {name} in {filepath}")
                deletions.append((test['full_start'], test['full_end']))
                continue

            lines = [l.strip() for l in body.split('\n') if l.strip()]
            is_just_variables = all(l.startswith('let ') for l in lines)
            if not lines or is_just_variables:
                print(f"Deleting var-only test {name} in {filepath}")
                deletions.append((test['full_start'], test['full_end']))

    if deletions:
        deletions.sort(reverse=True)
        new_text = original_content
        for start, end in deletions:
            pre_start = start
            while pre_start > 0 and new_text[pre_start-1] in (' ', '\t', '\n'):
                if new_text[pre_start-1] == '\n':
                    pre_start -= 1
                    break
                pre_start -= 1
            new_text = new_text[:pre_start] + new_text[end:]
        return new_text

    return None

def process_file(filepath):
    new_text = collect_modifications(filepath)
    if new_text is not None:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_text)

for root, _, files in os.walk('.'):
    if 'target' in root: continue
    for f in files:
        if f.endswith('.rs'):
            process_file(os.path.join(root, f))
