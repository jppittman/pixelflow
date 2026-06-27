import os
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    out_lines = []

    i = 0
    modified = False

    while i < len(lines):
        line = lines[i]

        # Look for #[test]
        # We need to correctly identify functions decorated with #[test]
        # and checking if they only contain things that aren't assertions
        if line.strip().startswith('#[test]'):
            # Look ahead for `fn `
            j = i + 1
            while j < len(lines) and not re.search(r'\bfn\s+', lines[j]):
                if lines[j].strip().startswith('#') or lines[j].strip() == '':
                    j += 1
                else:
                    break

            if j < len(lines) and re.search(r'\bfn\s+', lines[j]):
                # Found the start of the function.
                brace_count = 0
                started = False
                func_start = i

                k = j
                func_text = []
                while k < len(lines):
                    func_text.append(lines[k])

                    for char in lines[k]:
                        if char == '{':
                            started = True
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1

                    k += 1
                    if started and brace_count == 0:
                        break

                func_body = '\n'.join(func_text)

                # Check for assertions
                # What counts as an assertion?
                # assert!, assert_eq!, assert_ne!, panic!, unwrap(), expect()
                has_assert = bool(re.search(r'\bassert!|\bassert_eq!|\bassert_ne!|\bpanic!|\bunwrap\b|\bexpect\b', func_body))
                is_just_assert_true = False

                if has_assert:
                    # check if the body is just assert!(true); or similar
                    assert_true_count = len(re.findall(r'assert!\s*\(\s*true\s*\)', func_body))
                    other_asserts = len(re.findall(r'\bassert!|\bassert_eq!|\bassert_ne!|\bpanic!|\bunwrap\b|\bexpect\b', func_body))
                    if assert_true_count > 0 and other_asserts == assert_true_count:
                        is_just_assert_true = True

                if not has_assert or is_just_assert_true:
                    # Delete this test
                    i = k
                    modified = True
                    continue

                # Otherwise, it has assertions. We need to check its name.
                match = re.search(r'\bfn\s+test_([a-zA-Z0-9_]+)', lines[j])
                if match:
                    func_name = match.group(1)
                    new_name = func_name
                    # If starts with digit or is a reserved word (or shadows, hard to know in python)
                    if new_name and new_name[0].isdigit():
                        new_name = 'verify_' + new_name

                    # Replace in lines[j]
                    lines[j] = lines[j][:match.start(0)] + "fn " + new_name + lines[j][match.end(0):]
                    modified = True

                # Append the function
                for idx in range(func_start, k):
                    out_lines.append(lines[idx])
                i = k
                continue

        out_lines.append(lines[i])
        i += 1

    if modified:
        with open(filepath, 'w') as f:
            f.write('\n'.join(out_lines))
        print(f"Modified {filepath}")

for root, _, files in os.walk('.'):
    if 'src' in root or 'tests' in root:
        if '/target/' not in root:
            for file in files:
                if file.endswith('.rs'):
                    process_file(os.path.join(root, file))
