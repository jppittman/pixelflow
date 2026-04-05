import codecs

file_path = "pixelflow-runtime/src/platform/macos/cocoa.rs"
with open(file_path, "rb") as f:
    data = f.read()

# Try to decode and fix literal null bytes that were meant to be string escapes
try:
    text = data.decode('utf-8')
    text = text.replace('\x00', '\\0')
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(text)
    print("Fixed!")
except Exception as e:
    print("Failed", e)
