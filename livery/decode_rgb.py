"""Debug: decodifica e ispeziona i file Radiance RGB."""

from pathlib import Path
from PIL import Image

def decode_radiance_rgb(path: Path) -> Image.Image | None:
    """Decodifica un file Radiance RGB e restituisce un'immagine PIL."""
    with open(path, "rb") as f:
        # Legge l'header
        magic = f.read(2)
        if magic != b"\x01\xda":
            print(f"Invalid magic: {magic.hex()}")
            return None

        fmt = f.read(1)[0]
        components = f.read(1)[0]
        width_bytes = f.read(2)
        height_bytes = f.read(2)

        # Prova entrambi gli ordinamenti di byte (endianness)
        width_be = int.from_bytes(width_bytes, "big")
        height_be = int.from_bytes(height_bytes, "big")
        width_le = int.from_bytes(width_bytes, "little")
        height_le = int.from_bytes(height_bytes, "little")

        print(f"Format: {fmt}, Components: {components}")
        print(f"Width (BE): {width_be}, Height (BE): {height_be}")
        print(f"Width (LE): {width_le}, Height (LE): {height_le}")

        # Per ora assume big-endian
        width, height = width_be, height_be

        if width <= 0 or height <= 0 or width > 4096 or height > 4096:
            print("Invalid dimensions, trying little-endian")
            width, height = width_le, height_le

        print(f"Using: {width}x{height}, {components} components")

        # Legge le scanline con decompressione RLE
        pixels = []
        for y in range(height):
            row = []
            while len(row) < width * components:
                byte = f.read(1)[0]
                if byte > 0x80:
                    # RLE: ripete il prossimo byte (count & 0x7F) volte
                    count = byte & 0x7F
                    value = f.read(1)[0]
                    row.extend([value] * count)
                else:
                    # Byte grezzi
                    row.extend(f.read(byte))
            pixels.extend(row[:width * components])

        # Converte in immagine PIL
        if components == 3:
            img = Image.new("RGB", (width, height))
            img.putdata([(pixels[i], pixels[i+1], pixels[i+2])
                         for i in range(0, len(pixels)-2, 3)])
            return img

if __name__ == "__main__":
    converted = Path(r"U:\AI-Partition\torcs\torcs\cars\car1-stock1\car1-stock1.rgb")
    print(f"Decoding: {converted}")
    img = decode_radiance_rgb(converted)

    if img:
        print(f"Successfully decoded: {img.size}")
        # Salva l'anteprima
        preview = converted.parent / "car1-stock1.rgb.preview.png"
        img.save(preview)
        print(f"Saved preview: {preview}")
    else:
        print("Failed to decode")
