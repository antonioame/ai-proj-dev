"""Decodifica i file SGI/Radiance RGB in PNG per l'ispezione visiva.

UTILITÀ:
  TORCS usa il formato SGI RGB (512x512 RGBA) per le texture delle auto.
  Questo script decodifica i file .rgb in PNG standard per visualizzare
  l'anteprima della livrea PRIMA di installarla in TORCS.

  Usi comuni:
  - Verificare come appare la nuova livrea
  - Confrontare versioni diverse della livrea
  - Debug: controllare che la conversione PNG→RGB sia corretta

FORMATO SGI RGB:
  - Header fisso di 512 bytes (metadati: dimensioni, canali, compressione)
  - Dati pixel in formato planare: tutti i Red, poi Green, poi Blue
  - TORCS richiede 512x512 RGBA (senza compressione)
  - File size per car1-ow1: 512*512*4 + 512 = 1.049.088 bytes
"""
import struct, pathlib
from PIL import Image

def read_sgi(path):
    """Decodifica un file SGI RGB in un'immagine PIL (RGB)."""
    data = pathlib.Path(path).read_bytes()
    # Leggi header SGI: magic(2) storage(1) bpc(1) dim(2) xsize(2) ysize(2) zsize(2)
    magic, storage, bpc, dim, xsize, ysize, zsize = struct.unpack_from('>HBBHHhH', data, 0)
    print(f"{path.name}: {xsize}x{ysize} channels={zsize} storage={storage}")

    # I dati pixel iniziano dopo l'header fisso di 512 bytes
    offset = 512
    pixels = data[offset:]
    n = xsize * ysize

    # Estrai i canali: nel formato SGI planare, i colori sono separati
    r = list(pixels[0:n])        # Red: primo megapixel
    g = list(pixels[n:2*n])      # Green: secondo megapixel
    b = list(pixels[2*n:3*n])    # Blue: terzo megapixel

    # Crea immagine RGB e carica i dati pixel
    img = Image.new('RGB', (xsize, ysize))
    img.putdata(list(zip(r, g, b)))
    return img

# Decodifica la nuova livrea e salva anteprima
root = pathlib.Path(__file__).resolve().parent
new = read_sgi(root / "car1-ow1.rgb")
new.save(root / "livery_preview.png")
print("Preview salvato: livery_preview.png")
