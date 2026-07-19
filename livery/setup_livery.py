"""Installa e ripristina la livrea car1-ow1 in TORCS.

Usage:
    python livery/setup_livery.py                 installa livery/car1-ow1.rgb così com'è
    python livery/setup_livery.py mia_livrea.png   converte PNG -> livery/car1-ow1.rgb, poi installa
    python livery/setup_livery.py --reset          ripristina la livrea IBM di default
    python livery/setup_livery.py --status         mostra lo stato attuale
    python livery/setup_livery.py --rollback       ripristina l'ultimo backup lato TORCS
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LIVERY_DIR = Path(__file__).resolve().parent
TORCS_ROOT = Path(r"U:\AI-Partition\torcs\torcs")
CAR = "car1-ow1"
TEXTURE_SIZE = (512, 512)  # car1-ow1's native SGI RGB texture size

LIVERY_RGB = LIVERY_DIR / f"{CAR}.rgb"
IBM_DIR = LIVERY_DIR / "original_IBM_livery"
IBM_PNG = IBM_DIR / "original_IBM_livery.png"
IBM_RGB = IBM_DIR / f"{CAR}.rgb"

TORCS_TEXTURE_PATH = TORCS_ROOT / "cars" / CAR / f"{CAR}.rgb"
# TORCS mantiene una seconda copia indipendente della stessa texture per ogni slot
# driver scr_server. La race config (corkscrew_solo.xml) usa idx=0, ed è questa la
# copia che il gioco carica davvero per l'auto in gara: aggiornare solo
# TORCS_TEXTURE_PATH sopra non ha alcun effetto visibile in-game.
SCR_SERVER_TEXTURE_PATH = TORCS_ROOT / "drivers" / "scr_server" / "0" / f"{CAR}.rgb"
STATE_FILE = LIVERY_DIR / f".livery_state_{CAR}.json"


def _check_dependencies() -> None:
    """Verifica che PIL sia disponibile per la conversione PNG→RGB."""
    if Image is None:
        logger.error("PIL/Pillow not found. Install with: pip install Pillow")
        raise ImportError("PIL/Pillow required for PNG→RGB conversion")


def _png_to_sgi_rgb(png_path: Path, rgb_path: Path, size: tuple[int, int] = TEXTURE_SIZE) -> None:
    """Converte un PNG nel formato SGI RGB non compresso atteso da car1-ow1.

    Header di 512 byte (magic 0x01DA, storage=0/verbatim, 4 canali) seguito da
    un piano di (width*height) byte per canale, in ordine R, G, B, A, lo
    stesso layout che livery/decode_sgi.py rilegge, verificato contro il vero
    car1-ow1.rgb distribuito con l'auto (anch'esso storage=0, 512x512, 4 canali).
    """
    logger.info(f"Converting {png_path} to SGI RGB format: {rgb_path}")
    img = Image.open(png_path).convert("RGBA")
    if img.size != size:
        logger.info(f"Resizing image from {img.size} to {size}")
        img = img.resize(size, Image.Resampling.LANCZOS)

    width, height = img.size
    planes = img.split()  # (R, G, B, A), each a single-channel Image

    header = bytearray(512)
    header[0:2] = (0x01DA).to_bytes(2, "big")  # magic
    header[2] = 0                                # storage: 0 = verbatim (non compresso)
    header[3] = 1                                # bpc: 1 byte per canale
    header[4:6] = (3).to_bytes(2, "big")         # dim: 3 = immagine multicanale
    header[6:8] = width.to_bytes(2, "big")
    header[8:10] = height.to_bytes(2, "big")
    header[10:12] = (4).to_bytes(2, "big")       # zsize: 4 canali (RGBA)
    header[12:16] = (0).to_bytes(4, "big")       # pixmin
    header[16:20] = (255).to_bytes(4, "big")     # pixmax
    # byte 20:512 (dummy, nome immagine, colormap, padding) lasciati a zero

    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rgb_path, "wb") as f:
        f.write(bytes(header))
        for plane in planes:
            f.write(plane.tobytes())

    logger.info(f"Successfully converted to SGI RGB: {rgb_path} ({width}x{height}, RGBA)")


def _install_to(source_rgb: Path, target_path: Path) -> None:
    """Copia source_rgb in un singolo slot texture TORCS, facendo prima il backup dell'originale."""
    if not target_path.parent.exists():
        logger.error(f"TORCS target directory not found: {target_path.parent}")
        raise FileNotFoundError(f"Target directory missing: {target_path.parent}")

    if target_path.exists():
        backup_path = target_path.with_suffix(".rgb.backup")
        if not backup_path.exists():
            logger.info(f"Backing up original: {target_path} → {backup_path}")
            shutil.copy2(target_path, backup_path)
        else:
            logger.info(f"Backup already exists: {backup_path}")

    logger.info(f"Installing livery: {source_rgb} → {target_path}")
    shutil.copy2(source_rgb, target_path)


def _install(source_rgb: Path) -> None:
    """Copia source_rgb in tutti gli slot texture car1-ow1 di TORCS (cars/ e lo
    slot driver scr_server usato davvero in gara), facendo prima il backup degli originali."""
    if not source_rgb.exists():
        logger.error(f"Livery source not found: {source_rgb}")
        raise FileNotFoundError(f"Missing livery: {source_rgb}")
    if not TORCS_ROOT.exists():
        logger.error(f"TORCS root not found: {TORCS_ROOT}")
        raise FileNotFoundError(f"TORCS not found at {TORCS_ROOT}")

    for target_path in (TORCS_TEXTURE_PATH, SCR_SERVER_TEXTURE_PATH):
        _install_to(source_rgb, target_path)

    state = {
        "installed": True,
        "backup_exists": TORCS_TEXTURE_PATH.with_suffix(".rgb.backup").exists(),
        "scr_server_backup_exists": SCR_SERVER_TEXTURE_PATH.with_suffix(".rgb.backup").exists(),
        "torcs_path": str(TORCS_TEXTURE_PATH),
        "scr_server_path": str(SCR_SERVER_TEXTURE_PATH),
        "source": str(source_rgb),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("Livery installed successfully.")


def install_from_png(png_path: Path) -> None:
    """Converte una livrea PNG in livery/car1-ow1.rgb e la installa."""
    _check_dependencies()
    if not png_path.exists():
        logger.error(f"PNG not found: {png_path}")
        raise FileNotFoundError(f"Missing PNG: {png_path}")
    _png_to_sgi_rgb(png_path, LIVERY_RGB)
    _install(LIVERY_RGB)


def install_existing() -> None:
    """Installa la livrea già presente in livery/car1-ow1.rgb, così com'è."""
    _install(LIVERY_RGB)


def reset_to_default() -> None:
    """Rigenera la livrea IBM originale a partire dal suo PNG sorgente e la installa."""
    _check_dependencies()
    if not IBM_PNG.exists():
        logger.error(f"Original IBM livery PNG not found: {IBM_PNG}")
        raise FileNotFoundError(f"Missing source: {IBM_PNG}")
    _png_to_sgi_rgb(IBM_PNG, IBM_RGB)
    _install(IBM_RGB)


def rollback_livery() -> None:
    """Ripristina byte-per-byte qualunque texture TORCS fosse stata salvata in backup prima dell'ultima installazione."""
    any_restored = False
    for target_path in (TORCS_TEXTURE_PATH, SCR_SERVER_TEXTURE_PATH):
        backup_path = target_path.with_suffix(".rgb.backup")
        if not backup_path.exists():
            logger.warning(f"No backup found, skipping: {backup_path}")
            continue
        logger.info(f"Restoring from backup: {backup_path} → {target_path}")
        shutil.copy2(backup_path, target_path)
        any_restored = True

    if not any_restored:
        logger.error("Cannot rollback: no backups exist")
        raise FileNotFoundError("Cannot rollback: no backup exists")

    state = {
        "installed": False,
        "backup_exists": TORCS_TEXTURE_PATH.with_suffix(".rgb.backup").exists(),
        "scr_server_backup_exists": SCR_SERVER_TEXTURE_PATH.with_suffix(".rgb.backup").exists(),
        "torcs_path": str(TORCS_TEXTURE_PATH),
        "scr_server_path": str(SCR_SERVER_TEXTURE_PATH),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("Livery rolled back to the previous TORCS-side backup.")


def show_status() -> None:
    """Mostra lo stato attuale della livrea."""
    print(f"\n--- Livery Status ({CAR}) ---")
    print(f"Livery dir:       {LIVERY_DIR}")
    print(f"TORCS root:       {TORCS_ROOT}")
    print(f"Current livery:   {LIVERY_RGB} {'✓' if LIVERY_RGB.exists() else '✗'}")
    print(f"IBM default PNG:  {IBM_PNG} {'✓' if IBM_PNG.exists() else '✗'}")

    for target_path in (TORCS_TEXTURE_PATH, SCR_SERVER_TEXTURE_PATH):
        backup_path = target_path.with_suffix(".rgb.backup")
        print(f"Texture target:   {target_path}")
        print(f"  Exists:         {'✓' if target_path.exists() else '✗'}")
        print(f"  Backup:         {backup_path}")
        print(f"    Exists:       {'✓' if backup_path.exists() else '✗'}")
    print()

    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
            print(f"Last action:      {'installed' if state['installed'] else 'rolled back'}")
            if "source" in state:
                print(f"Last source:      {state['source']}")
    else:
        print("Last action:      unknown (state file not found)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the car1-ow1 livery for TORCS")
    parser.add_argument("png", nargs="?", default=None,
                        help="PNG to convert and install (omit to install livery/car1-ow1.rgb as-is)")
    parser.add_argument("--reset", action="store_true", help="Restore the default IBM livery")
    parser.add_argument("--rollback", action="store_true", help="Restore the last TORCS-side backup")
    parser.add_argument("--status", action="store_true", help="Show current state")
    args = parser.parse_args()

    if args.reset:
        reset_to_default()
    elif args.rollback:
        rollback_livery()
    elif args.status:
        show_status()
    elif args.png:
        png_path = Path(args.png)
        if png_path.suffix.lower() != ".png":
            parser.error(f"Expected a .png file, got: {args.png}")
        install_from_png(png_path)
    else:
        install_existing()


if __name__ == "__main__":
    main()
