"""Test BC driver directly."""
from drivers.bc.driver import BCDriver
from torcs_env.sensors import SensorState
import time

driver = BCDriver()
time.sleep(2)  # Wait for model load

# Test the same states as debug script
states = [
    SensorState(speed=0.0, trackPos=0.33, angle=0.01, rpm=7869, gear=0, lastLapTime=-1, lap=1, 
                distFromStart=3598, distRaced=0, curLapTime=0, damage=0, track=[0]*19, wheelSpinVel=[0]*4),
    SensorState(speed=39.3, trackPos=0.29, angle=0.05, rpm=6218, gear=1, lastLapTime=-1, lap=1,
                distFromStart=7, distRaced=0, curLapTime=0, damage=0, track=[0]*19, wheelSpinVel=[0]*4),
    SensorState(speed=75.4, trackPos=-0.63, angle=0.22, rpm=11940, gear=1, lastLapTime=-1, lap=1,
                distFromStart=41, distRaced=0, curLapTime=0, damage=0, track=[0]*19, wheelSpinVel=[0]*4),
]

for i, state in enumerate(states):
    action = driver.step(state)
    print(f"State {i}: steer={action.steer:.6f}, accel={action.accel:.6f}, brake={action.brake:.6f}")
