"""Test all models and record lap times."""
import subprocess
import json
import re
from pathlib import Path

models_to_test = [
    ("rule_based", "Baseline (Rule-Based)"),
    ("bc_model", "BC v2 (35.4k samples)"),
    ("rl_bc_warmstart", "RL + BC Warm-Start (50k steps)"),
]

results = {}

for driver_name, description in models_to_test:
    print("\n" + "="*70)
    print("Testing: " + description)
    print("Driver: " + driver_name)
    print("="*70)
    
    cmd = "conda run -n ai_env python scripts/run_agent.py --driver " + driver_name + " --laps 1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
    
    # Extract lap time from output
    output = result.stdout + result.stderr
    lap_time_match = re.search(r'"best_lap":\s*([\d.]+)', output)
    off_track_match = re.search(r'"off_track_pct":\s*([\d.]+)', output)
    
    if lap_time_match:
        lap_time = float(lap_time_match.group(1))
        off_track = float(off_track_match.group(1)) if off_track_match else 0.0
        results[driver_name] = {
            "description": description,
            "lap_time": lap_time,
            "off_track_pct": off_track
        }
        print("OK - Lap Time: {:.2f}s".format(lap_time))
        print("   Off-Track: {:.1f}%".format(off_track))
    else:
        results[driver_name] = {
            "description": description,
            "lap_time": None,
            "off_track_pct": None,
            "error": "Could not extract lap time"
        }
        print("FAIL - Could not extract lap time")
        print("Output snippet: " + output[-500:])

# Save results
with open("baseline_performance.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("BASELINE PERFORMANCE SUMMARY")
print("="*70)
for driver, data in results.items():
    if data.get("lap_time"):
        print("{:40s} | {:7.2f}s | {:5.1f}% off".format(
            data['description'], data['lap_time'], data['off_track_pct']))
    else:
        print("{:40s} | ERROR: {}".format(data['description'], data.get('error', 'Unknown')))

print("\nResults saved to: baseline_performance.json")
