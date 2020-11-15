


C24 = [
    56.36, 59.25
]

C48 = [
    30.86, 32.32
]

C60 = [
    35.44
]


C90 = [
    11.73, 20.67
]

C120 = [
    19.28
]


C180 = [
    12.55
]

S24 = [
    55.52, 55.35
]

S48 = [
    30.87, 31.83
]

S60 = [
    35.31
]

S90 = [
    10.97, 27.2
]

S120 = [
    18.57,
]

S180 = [
    13.93
]

cores = {
    24: 6, 48: 24, 60: 48, 90: 96, 120: 168, 180: 384
}

import numpy as np
CS = np.array([
    np.mean(C24) / cores[24],
    np.mean(C48) / cores[48],
    np.mean(C60) / cores[60],
    np.mean(C90) / cores[90],
    np.mean(C120) / cores[120],
    np.mean(C180) / cores[180],
]) # days / core day

SG = np.array([
    np.mean(S24) / cores[24],
    np.mean(S48) / cores[48],
    np.mean(S60) / cores[60],
    np.mean(S90) / cores[90],
    np.mean(S120) / cores[120],
    np.mean(S180) / cores[180],
])

CS = CS * 365 / (365/12)  # months / cy
SG = SG * 365 / (365/12)  # months / cy

res = np.array([24, 48, 60, 90, 120, 180])

import matplotlib.pyplot as plt

plt.plot(res**2, 1/CS, label='Cubed-sphere', marker='o', linewidth=3)
plt.plot(res**2, 1/SG, label='Stretched cubed-sphere', marker='o', linewidth=3)

ax = plt.gca()

# ax.set_xscale('log', basex=2)
# ax.set_yscale('log', basey=2)

plt.xlabel('Cubed-sphere resolution')

plt.ylabel('Core years / month')

plt.xticks(
    res**2,
    [f'{r}'for r in [24, 48, 60, 90, 120, 180]],
)

# plt.yticks(
#     [2/64, 2/16, 2/4, 2],
#     ['2/64', '2/16', '2/4', '2'],
# )

# plt.yticks(
#     [3/4, 3, 12, 48],
#     [3/4, 3, 12, 48],
# )

plt.legend()
plt.show()