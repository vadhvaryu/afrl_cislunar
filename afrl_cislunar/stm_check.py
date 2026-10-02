"""
State transition matrix (STM) check
===================================

1. Checks the STM against brute force: nudge each starting value a tiny bit,
   re-propagate, and compare the change with what the STM predicted.
2. Shows what the STM is good for: predicting how a small starting error
   (1 km, 1 m/s) grows along an Artemis II-like trajectory, and how
   sensitive the trajectory is over time.

Starting state = Orion just after TLI, copied from artemis2_vs_cr3bp.py output,
so this runs without the NASA data file.

Requires: numpy, scipy, matplotlib
"""

# %% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cr3bp import nondim_cr3bp, nondim_cr3bp_stm

# %% 1. Setup (same constants as the Artemis script)
mu_E = 398600.4418        # Earth GM [km^3/s^2]
mu_M = 4902.800066        # Moon  GM [km^3/s^2]
pi_2 = mu_M / (mu_E + mu_M)
L = 396725.0              # distance unit [km], from the Artemis run
T_unit = 4.553 * 86400    # time unit [s]
V_unit = L / T_unit       # velocity unit [km/s]

Y_0 = np.array([5.53700e-02, -4.60000e-03, 9.60000e-04,
                4.68573e+00, 2.27327e+00, 4.52900e-02])
days = 8
tau = np.linspace(0, days * 86400 / T_unit, 2000)
tol = dict(method="DOP853", rtol=1e-12, atol=1e-12)

# %% 2. Propagate the trajectory together with the STM
Y_0_stm = np.hstack((Y_0, np.eye(6).ravel()))       # STM starts as the identity
sol = solve_ivp(nondim_cr3bp_stm, [0, tau[-1]], Y_0_stm, t_eval=tau, args=(pi_2,), **tol)
Phi = sol.y[6:].T.reshape(-1, 6, 6)                  # one 6x6 STM per time step
Phi_end = Phi[-1]

# %% 3. Check 1: STM vs brute force (finite differences)
h = 1e-8                                             # tiny nudge (nondimensional)
Phi_fd = np.zeros((6, 6))
for j in range(6):
    dY = np.zeros(6)
    dY[j] = h
    plus = solve_ivp(nondim_cr3bp, [0, tau[-1]], Y_0 + dY, args=(pi_2,), **tol).y[:, -1]
    minus = solve_ivp(nondim_cr3bp, [0, tau[-1]], Y_0 - dY, args=(pi_2,), **tol).y[:, -1]
    Phi_fd[:, j] = (plus - minus) / (2 * h)

rel_err = np.abs(Phi_end - Phi_fd).max() / np.abs(Phi_end).max()
print(f"STM vs brute force after {days} days: relative difference {rel_err:.1e}")
print("  (small, e.g. below 1e-4, means the STM is right)")

# %% 4. Check 2: does the STM predict how a real-sized error grows?
# 1 km error in each position direction, 1 m/s in each velocity direction
dY_0 = np.hstack((np.full(3, 1.0 / L), np.full(3, 0.001 / V_unit)))
pert = solve_ivp(nondim_cr3bp, [0, tau[-1]], Y_0 + dY_0, t_eval=tau, args=(pi_2,), **tol)

actual_km = np.linalg.norm(pert.y[:3].T - sol.y[:3].T, axis=1) * L
stm_km = np.linalg.norm((Phi @ dY_0)[:, :3], axis=1) * L
t_days = tau * T_unit / 86400
for d in (1, 2, 3, 4, 5, 6, 7):
    k = np.searchsorted(t_days, d)
    print(f"  after {d} day(s): actual error {actual_km[k]:,.1f} km, STM predicted {stm_km[k]:,.1f} km")

# %% 5. Plots
fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=100)

ax = axs[0]
ax.semilogy(t_days, actual_km, "k", lw=2, label="Actual (re-propagated)")
ax.semilogy(t_days, stm_km, "r--", lw=1.5, label="STM prediction")
ax.set_title("Growth of a 1 km / 1 m/s starting error")
ax.set_xlabel("Days after TLI")
ax.set_ylabel("Position error (km, log scale)")
ax.legend()
ax.grid(alpha=0.3, which="both")

ax = axs[1]
growth = np.linalg.norm(Phi, ord=2, axis=(1, 2))    # biggest stretch factor of the STM
ax.semilogy(t_days, growth, "C0")
ax.set_title("How sensitive the trajectory is (STM size)")
ax.set_xlabel("Days after TLI")
ax.set_ylabel("Largest error amplification (log scale)")
ax.grid(alpha=0.3, which="both")

fig.suptitle("State transition matrix check, Artemis II-like trajectory", fontsize=14)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/stm_check.png", bbox_inches="tight")
plt.show()
