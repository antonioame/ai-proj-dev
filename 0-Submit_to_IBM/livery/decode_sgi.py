"""Decode SGI/Radiance RGB files to PNG for visual inspection.

PURPOSE:
  TORCS uses the SGI RGB format (512x512 RGBA) for car textures.
  This script decodes .rgb files into standard PNG so the livery
  preview can be inspected BEFORE installing it into TORCS.

  Common uses:
  - Check how the new livery looks
  - Compare different livery versions
  - Debug: verify that the PNG->RGB conversion is correct

SGI RGB FORMAT:
  - Fixed 512-byte header (metadata: dimensions, channels, compression)
  - Pixel data in planar format: all Red, then Green, then Blue
  - TORCS requires 512x512 RGBA (uncompressed)
  - File size for car1-ow1: 512*512*4 + 512 = 1,049,088 bytes
"""
import struct, pathlib
from PIL import Image

def read_sgi(path):
    """Decode SGI RGB file to PIL Image (RGB)."""
    data = pathlib.Path(path).read_bytes()
    # Read SGI header: magic(2) storage(1) bpc(1) dim(2) xsize(2) ysize(2) zsize(2)
    magic, storage, bpc, dim, xsize, ysize, zsize = struct.unpack_from('>HBBHHhH', data, 0)
    print(f"{path.name}: {xsize}x{ysize} channels={zsize} storage={storage}")

    # Pixel data starts after the fixed 512-byte header
    offset = 512
    pixels = data[offset:]
    n = xsize * ysize

    # Extract channels: in the planar SGI format, colors are separated
    r = list(pixels[0:n])        # Red: first plane
    g = list(pixels[n:2*n])      # Green: second plane
    b = list(pixels[2*n:3*n])    # Blue: third plane

    # Build RGB image and load pixel data
    img = Image.new('RGB', (xsize, ysize))
    img.putdata(list(zip(r, g, b)))
    return img

# Decode the new livery and save a preview
root = pathlib.Path(__file__).resolve().parent
new = read_sgi(root / "car1-ow1.rgb")
new.save(root / "livery_preview.png")
print("Preview saved: livery_preview.png")
