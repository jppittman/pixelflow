with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'r', encoding='utf-8') as f:
    text = f.read()

new_text = text.replace('.clone()', '')
with open('pixelflow-graphics/src/fonts/ttf_curve_analytical.rs', 'w', encoding='utf-8') as f:
    f.write(new_text)
