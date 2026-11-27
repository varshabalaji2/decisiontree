import math, random

target = 0.10          # we want it to fire ~10% of steps
a = 0.0                # the raw learnable number
inp = 1.2
threshold = 1.0

for r in range(15):                      # NEW: outer loop, 15 attempts
    k = 1 / (1 + math.exp(-a))           # NEW: k comes from a
    v = 0.0
    spikes = 0
    for t in range(200):                 # your old loop, now nested
        noise = random.uniform(-0.3, 0.3)
        v = v + k * ((inp + noise) - v)
        if v >= threshold:
            spikes = spikes + 1
            v = 0.0
    rate = spikes / 200                  # NEW: measure it
    print(r, round(a, 3), round(k, 3), rate)
    a = a - 3.0 * (rate - target)        # NEW: the nudges