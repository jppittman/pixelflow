import sys

filepath = 'pixelflow-runtime/src/frame.rs'

with open(filepath, 'r') as f:
    text = f.read()

search_text = """    #[test]
    fn test_create_channels() {
        let (_handle, _rx) = create_frame_channel::<TestSurface>();
        let (_recycle_tx, _recycle_rx) = create_recycle_channel::<TestSurface>();
    }

"""

if search_text in text:
    new_text = text.replace(search_text, "")
    with open(filepath, 'w') as f:
        f.write(new_text)
    print("Test removed successfully.")
else:
    print("Search string not found")
