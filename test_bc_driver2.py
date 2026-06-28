"""Test BC driver directly with logging."""
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')

from drivers.bc.driver import BCDriver
from torcs_env.sensors import SensorState
import time

print("Creating BCDriver...")
driver = BCDriver()
print("Waiting for model load...")
time.sleep(3)  # Wait for model load

print("\nTesting inference:")
state = SensorState(speed=39.3, trackPos=0.29, angle=0.05, rpm=6218, gear=1, lastLapTime=-1, lap=1,
                    distFromStart=7, distRaced=0, curLapTime=0, damage=0, track=[0]*19, wheelSpinVel=[0]*4)
for i in range(3):
    action = driver.step(state)
    print(f"Step {i}: steer={action.steer:.6f}, accel={action.accel:.6f}")
