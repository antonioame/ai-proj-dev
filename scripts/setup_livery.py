"""Setup and rollback car livery.

This script manages car livery installation for TORCS in a reversible manner.

Usage:
    python scripts/setup_livery.py --install    (install livery with backup)
    python scripts/setup_livery.py --rollback   (restore original)
    python scripts/setup_livery.py --status     (show current state)
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

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCS_ROOT = Path(r"U:\AI-Partition\torcs\torcs")
LIVERY_SOURCE = PROJECT_ROOT / "livrea.png"
LIVERY_CAR = "car1-stock1"
LIVERY_TEXTURE = f"{LIVERY_CAR}.rgb"
TORCS_TEXTURE_PATH = TORCS_ROOT / "cars" / LIVERY_CAR / LIVERY_TEXTURE
LIVERY_STATE_FILE = PROJECT_ROOT / ".livery_state.json"


def _check_dependencies() -> None:
    """Verify PIL is available for PNG→RGB conversion."""
    if Image is None:
        logger.error("PIL/Pillow not found. Install with: pip install Pillow")
        raise ImportError("PIL/Pillow required for PNG→RGB conversion")


def _png_to_rgb(png_path: Path, rgb_path: Path) -> None:
    """Convert PNG image to TORCS Radiance RGB format.

    Note: This writes RGB as a simple 8-bit RGB dump (no RLE compression).
    TORCS can load uncompressed RGB files, though compressed versions are also valid.
    """
    logger.info(f"Converting {png_path} to RGB format: {rgb_path}")
    img = Image.open(png_path).convert("RGB")

    # Write as uncompressed Radiance RGB (simple dump of RGB bytes)
    width, height = img.size
    with open(rgb_path, "wb") as f:
        # Radiance RGB header (magic, xres, yres)
        f.write(b"\x01\xda")  # Radiance RGB magic
        f.write(bytes([0x01, 0x01]))  # One byte per component, XYES order
        f.write(width.to_bytes(2, "big"))  # X resolution (width)
        f.write(height.to_bytes(2, "big"))  # Y resolution (height)

        # Write RGB data (row by row, top to bottom)
        for y in range(height):
            row = img.crop((0, y, width, y + 1))
            f.write(row.tobytes())

    logger.info(f"Successfully converted to RGB: {rgb_path}")


def install_livery() -> None:
    """Install livery with automatic backup."""
    _check_dependencies()

    if not LIVERY_SOURCE.exists():
        logger.error(f"Livery source not found: {LIVERY_SOURCE}")
        raise FileNotFoundError(f"Missing livery: {LIVERY_SOURCE}")

    if not TORCS_ROOT.exists():
        logger.error(f"TORCS root not found: {TORCS_ROOT}")
        raise FileNotFoundError(f"TORCS not found at {TORCS_ROOT}")

    if not TORCS_TEXTURE_PATH.parent.exists():
        logger.error(f"TORCS car directory not found: {TORCS_TEXTURE_PATH.parent}")
        raise FileNotFoundError(f"Car directory missing: {TORCS_TEXTURE_PATH.parent}")

    # Backup original if it exists and not already backed up
    if TORCS_TEXTURE_PATH.exists():
        backup_path = TORCS_TEXTURE_PATH.with_suffix(".rgb.backup")
        if not backup_path.exists():
            logger.info(f"Backing up original: {TORCS_TEXTURE_PATH} → {backup_path}")
            shutil.copy2(TORCS_TEXTURE_PATH, backup_path)
        else:
            logger.info(f"Backup already exists: {backup_path}")

    # Convert and install new livery
    logger.info(f"Installing livery: {LIVERY_SOURCE} → {TORCS_TEXTURE_PATH}")
    _png_to_rgb(LIVERY_SOURCE, TORCS_TEXTURE_PATH)

    # Record state
    state = {
        "installed": True,
        "backup_exists": TORCS_TEXTURE_PATH.with_suffix(".rgb.backup").exists(),
        "torcs_path": str(TORCS_TEXTURE_PATH),
    }
    with open(LIVERY_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("Livery installed successfully.")


def rollback_livery() -> None:
    """Restore original livery from backup."""
    backup_path = TORCS_TEXTURE_PATH.with_suffix(".rgb.backup")

    if not backup_path.exists():
        logger.error(f"No backup found: {backup_path}")
        raise FileNotFoundError("Cannot rollback: no backup exists")

    logger.info(f"Restoring from backup: {backup_path} → {TORCS_TEXTURE_PATH}")
    shutil.copy2(backup_path, TORCS_TEXTURE_PATH)

    # Update state
    state = {
        "installed": False,
        "backup_exists": True,
        "torcs_path": str(TORCS_TEXTURE_PATH),
    }
    with open(LIVERY_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("Livery rolled back to original.")


def show_status() -> None:
    """Show current livery state."""
    backup_path = TORCS_TEXTURE_PATH.with_suffix(".rgb.backup")

    print(f"\n--- Livery Status ---")
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"TORCS root:       {TORCS_ROOT}")
    print(f"Livery source:    {LIVERY_SOURCE} {'✓' if LIVERY_SOURCE.exists() else '✗'}")
    print(f"Texture target:   {TORCS_TEXTURE_PATH}")
    print(f"  Exists:         {'✓' if TORCS_TEXTURE_PATH.exists() else '✗'}")
    print(f"Backup:           {backup_path}")
    print(f"  Exists:         {'✓' if backup_path.exists() else '✗'}")
    print()

    if LIVERY_STATE_FILE.exists():
        with open(LIVERY_STATE_FILE) as f:
            state = json.load(f)
            print(f"Last action:      {'installed' if state['installed'] else 'rolled back'}")
    else:
        print(f"Last action:      unknown (state file not found)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage car livery for TORCS")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install livery with backup",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Restore original livery",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current state",
    )
    args = parser.parse_args()

    if not any([args.install, args.rollback, args.status]):
        parser.print_help()
        return

    if args.install:
        install_livery()
    elif args.rollback:
        rollback_livery()
    elif args.status:
        show_status()


if __name__ == "__main__":
    main()
