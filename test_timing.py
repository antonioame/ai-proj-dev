"""Quick timing test: measure how long one step cycle takes."""
import time
from torcs_env.client import TORCSClient
from torcs_env.actions import Action

client = TORCSClient(host="localhost", port=3001)
client.connect()

print("Testing 10 action cycles...")
times = []
for i in range(10):
    t0 = time.perf_counter()

    state = client.receive()
    if isinstance(state, str):
        print(f"Step {i}: Got control message {state}, stopping")
        break

    action = Action(steer=0.1, accel=0.5, brake=0.0, gear=3)
    client.send(action)

    t1 = time.perf_counter()
    elapsed = (t1 - t0) * 1000  # ms
    times.append(elapsed)
    print(f"Step {i}: {elapsed:.1f} ms")

if times:
    print(f"\nAverage: {sum(times)/len(times):.1f} ms")
    print(f"Max: {max(times):.1f} ms")
    print(f"TORCS timeout: 2850 ms")
    if max(times) > 2850:
        print("⚠️  TIMING EXCEEDED TORCS TIMEOUT!")
    else:
        print("✓ Timing OK")

client.close()
