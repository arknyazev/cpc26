import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("poincare_all_points.txt", comments="#")

s = data[:, 0]
theta = data[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(theta, s, s=0.5, color="steelblue", alpha=0.6, rasterized=True)

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$s$")
ax.set_title("Poincaré section — passing particles, cpu")

plt.tight_layout()
plt.savefig("poincare_passing_cpu.png", dpi=200)
plt.show()
