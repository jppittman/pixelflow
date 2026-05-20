import os

for root, dirs, files in os.walk('.'):
    if '/src' in root:
        for file in files:
            if file.endswith('.rs'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                    if 'DisplayControl::SetVisible' in content:
                        print(f"File: {filepath}")
