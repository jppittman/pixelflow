import os
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]

        # Check if this line has #[test]
        if '#[test]' in line.replace(' ', ''):
            new_lines.append(line)
            i += 1

            lookahead = min(5, len(lines) - i)
            for j in range(lookahead):
                # Search for any test starting with fn test_
                # (but ONLY if it's the one under #[test])
                match = re.search(r'(pub\s+fn\s+|fn\s+)test_([a-zA-Z0-9_]+)', lines[i+j])
                if match:
                    prefix = match.group(1)
                    name = match.group(2)

                    if name[0].isdigit():
                        name = "verify_" + name
                    elif re.search(r'\b' + re.escape(name) + r'\s*[\(!<]', content):
                        name = "verify_" + name

                    lines[i+j] = lines[i+j].replace(match.group(0), prefix + name)
                    changed = True
                    break
        else:
            new_lines.append(line)
            i += 1

    if changed:
        print(f"Updating {filepath}")
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'target']
    for file in files:
        if file.endswith('.rs'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    if '#[test]' in content and 'fn test_' in content:
                        process_file(filepath)
            except Exception as e:
                pass
