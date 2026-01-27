
import sys
import time

print("=" * 60)
print("FLUXION OPTIMIZER MONITOR")
print("=" * 60)
print()
sys.stdout.flush()

while True:
    try:
        line = sys.stdin.readline()
        if not line:
            time.sleep(0.1)
            continue
        print(line.rstrip())
        sys.stdout.flush()
    except:
        break
