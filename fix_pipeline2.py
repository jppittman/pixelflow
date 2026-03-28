import os
import glob

# add allow(warnings) to all binaries and lib.rs in pixelflow-pipeline
for path in glob.glob('pixelflow-pipeline/src/**/*.rs', recursive=True):
    if path.endswith('main.rs') or path.endswith('lib.rs') or 'bin/' in path:
        with open(path, 'r') as f:
            content = f.read()
        if not content.startswith("#![allow(warnings)]"):
            content = "#![allow(warnings)]\n" + content
        with open(path, 'w') as f:
            f.write(content)
