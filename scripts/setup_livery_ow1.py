"""Setup and rollback car livery for car1-ow1 (using pre-converted RGB file).

This script uses the car1-ow1 (1).rgb file from the project as the target texture.

Usage:
    python scripts/setup_livery_ow1.py --install    (install livery with backup)
    python scripts/setup_livery_ow1.py --rollback   (restore original)
    python scripts/setup_livery_ow1.py --status     (show current state)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCS_ROOT = Path(r"U:\AI-Partition\torcs\torcs")
SOURCE_RGB = PROJECT_ROOT / "car1-ow1.rgb"  # Pre-converted RGB from project
LIVERY_CAR = "car1-ow1"
LIVERY_TEXTURE = f"{LIVERY_CAR}.rgb"
TORCS_TEXTURE_PATH = TORCS_ROOT / "cars" / LIVERY_CAR / LIVERY_TEXTURE
LIVERY_STATE_FILE = PROJECT_ROOT / ".livery_state_ow1.json"


def install_livery() -> None:
    """Install livery using the pre-converted RGB file."""
    if not SOURCE_RGB.exists():
        logger.error(f"Source RGB not found: {SOURCE_RGB}")
        raise FileNotFoundError(f"Missing source RGB: {SOURCE_RGB}")

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

    # Copy livery RGB to TORCS
    logger.info(f"Installing livery: {SOURCE_RGB} → {TORCS_TEXTURE_PATH}")
    shutil.copy2(SOURCE_RGB, TORCS_TEXTURE_PATH)

    # Record state
    state = {
        "installed": True,
        "backup_exists": TORCS_TEXTURE_PATH.with_suffix(".rgb.backup").exists(),
        "torcs_path": str(TORCS_TEXTURE_PATH),
        "source": str(SOURCE_RGB),
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

    print(f"\n--- Livery Status (car1-ow1) ---")
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"TORCS root:       {TORCS_ROOT}")
    print(f"Source RGB:       {SOURCE_RGB} {'✓' if SOURCE_RGB.exists() else '✗'}")
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
    parser = argparse.ArgumentParser(description="Manage car1-ow1 livery for TORCS")
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
