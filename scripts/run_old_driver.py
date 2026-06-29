"""
Run old project driver (torcs_jm_par.py) and record telemetry for BC retraining.

Usage:
    conda run -n ai_env python scripts/run_old_driver.py --host localhost --port 3001 --laps 5
"""

import sys
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add old project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vecchio_progetto"))

# Import the old driver
import torcs_jm_par as old_driver

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(host: str = "localhost", port: int = 3001, laps: int = 5):
    """Run old driver and record telemetry."""

    # Parse argomenti manualmente per evitare il parser del vecchio Client
    # Pulisci sys.argv per il vecchio Client (non vuole --laps, --host, --port)
    old_sys_argv = sys.argv
    sys.argv = [sys.argv[0]]

    # Connect to TORCS
    print(f"[INFO] Connecting to old driver ({host}:{port})...")
    client = old_driver.Client(H=host, p=port, vision=False)

    # Restore sys.argv
    sys.argv = old_sys_argv
    print("[INFO] Connected. Starting data collection...\n")

    # CSV output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PROJECT_ROOT / "data" / f"old_driver_{timestamp}.csv"
    output_file.parent.mkdir(exist_ok=True)

    # CSV columns match the format we use for training
    csv_columns = [
        "timestamp", "angle", "speed", "speedY", "speedZ", "trackPos",
        "track_0", "track_1", "track_2", "track_3", "track_4", "track_5",
        "track_6", "track_7", "track_8", "track_9", "track_10", "track_11",
        "track_12", "track_13", "track_14", "track_15", "track_16", "track_17",
        "track_18", "rpm", "gear", "damage", "distRaced", "curLapTime",
        "steer", "accel", "brake", "speedY", "speedZ"
    ]

    rows = []
    lap_count = 0
    tempo_giro_precedente = 0.0
    contatore_passi = 0

    print(f"Running {laps} laps. Press Ctrl+C to stop.\n")

    try:
        while lap_count < laps:
            client.get_servers_input()
            S = client.S.d

            # Lap detection
            tempo_giro_corrente = S.get("curLapTime", 0.0)
            if tempo_giro_corrente < tempo_giro_precedente - 1.0:
                lap_count += 1
                print(f"[LAP {lap_count}/{laps}] Completed. Time: {tempo_giro_precedente:.1f}s")
                if lap_count >= laps:
                    break
            tempo_giro_precedente = tempo_giro_corrente

            # Run old driver logic
            old_driver.drive_example(client)
            R = client.R.d

            # Send response
            client.respond_to_server()

            # Record telemetry
            track_list = S.get("track", [200.0] * 19)

            row = {
                "timestamp": time.time(),
                "angle": S.get("angle", 0.0),
                "speed": S.get("speedX", 0.0),
                "speedY": S.get("speedY", 0.0),
                "speedZ": S.get("speedZ", 0.0),
                "trackPos": S.get("trackPos", 0.0),
            }

            for i in range(19):
                row[f"track_{i}"] = track_list[i] if i < len(track_list) else 200.0

            row["rpm"] = S.get("rpm", 0.0)
            row["gear"] = S.get("gear", 0)
            row["damage"] = S.get("damage", 0.0)
            row["distRaced"] = S.get("distRaced", 0.0)
            row["curLapTime"] = S.get("curLapTime", 0.0)
            row["steer"] = R.get("steer", 0.0)
            row["accel"] = R.get("accel", 0.0)
            row["brake"] = R.get("brake", 0.0)
            row["speedY"] = S.get("speedY", 0.0)
            row["speedZ"] = S.get("speedZ", 0.0)

            rows.append(row)
            contatore_passi += 1

            if contatore_passi % 50 == 0:
                print(f"  Step {contatore_passi} | Lap {lap_count} | Speed {S.get('speedX', 0):.1f} km/h | "
                      f"Steer {R.get('steer', 0):.3f} | Accel {R.get('accel', 0):.2f} | Brake {R.get('brake', 0):.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # Save CSV
    print(f"\n[INFO] Saving {len(rows)} samples to {output_file}")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Data saved: {output_file}")
    print(f"[INFO] Collected {lap_count} complete lap(s), {len(rows)} samples")

    client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="TORCS server host")
    parser.add_argument("--port", type=int, default=3001, help="TORCS server port")
    parser.add_argument("--laps", type=int, default=5, help="Number of laps to record")
    args = parser.parse_args()
    run(host=args.host, port=args.port, laps=args.laps)
