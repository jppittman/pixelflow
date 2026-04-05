import codecs

file_path = "pixelflow-runtime/src/platform/macos/cocoa.rs"
with open(file_path, "rb") as f:
    data = f.read()

text = data.decode('utf-8').replace('\x00', '\\0')

with open(file_path, "w", encoding='utf-8') as f:
    f.write(text)
