"""
Synthetic observation generator (angles-only)
=============================================

Makes fake telescope observations of a cislunar object, so we can test IOD
(initial orbit determination) on data where we know the right answer.

1. Propagates a "truth" trajectory with the CR3BP (cr3bp.py).
   Default: the L1 northern halo orbit from Heidrich & Holzinger (2025), Table 1.
2. Places an observer at Earth's center (version 1; a real ground station is next).
3. At each observation time, computes the two angles a telescope would measure,
   right ascension (RA) and declination (Dec), and adds random noise.
4. Saves two files:
     data/synthetic_obs.csv    what the telescope "sees"  -> input to IOD
     data/synthetic_truth.csv  the real answer            -> only for grading IOD
5. Plots the trajectory, the lines of sight, and RA/Dec over time.

Angles are measured in an inertial (non-rotating) frame centered on Earth, with
its x-y plane = the Earth-Moon orbital plane and x pointing at the Moon at t = 0.
That is what a telescope sees (the stars don't spin with the Earth-Moon frame).
Converting to true equatorial RA/Dec (Earth's equator) is a fixed rotation that
Orekit can handle later.

Requires: numpy, scipy, matplotlib
"""

# %% Imports
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cr3bp import nondim_cr3bp

# %% 1. Settings (change these to make different test cases)
MU = 0.0121505             # Earth-Moon mass ratio (same as the paper)
DU = 384338.0              # distance unit [km]
TU = 375099.0              # time unit [s]

# Truth starting state [x, y, z, vx, vy, vz], nondimensional rotating frame
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])   # L1 northern halo

N_OBS = 5                  # number of observations (IOD needs at least 3)
SPAN_DAYS = 8.0            # time from first to last observation (halo period is ~9.6 days)
NOISE_ARCSEC = 10.0        # 1-sigma telescope noise per angle (paper uses 10")
SEED = 1                   # fixed random seed so results are repeatable

OBSERVER = np.array([-MU, 0.0, 0.0])   # Earth's center in the rotating frame

# %% 2. Propagate the truth trajectory
tau_end = SPAN_DAYS * 86400 / TU
tau_fine = np.linspace(0, tau_end, 2000)                 # smooth curve for plots
tau_obs = np.linspace(0, tau_end, N_OBS)                 # observation times

tol = dict(method="DOP853", rtol=1e-12, atol=1e-12)
sol = solve_ivp(nondim_cr3bp, [0, tau_end], TRUTH_STATE, args=(MU,),
                dense_output=True, **tol)
Y_fine = sol.sol(tau_fine).T                             # (2000, 6)
Y_obs = sol.sol(tau_obs).T                               # (N_OBS, 6)


# %% 3. From positions to telescope angles
def line_of_sight_inertial(r_rot, tau):
    """Observer-to-object vector, rotated from the spinning Earth-Moon frame
    into the fixed (inertial) frame. The frame has turned by angle tau since t0."""
    l = r_rot - OBSERVER                                 # relative position, rotating axes
    c, s = np.cos(tau), np.sin(tau)
    return np.column_stack((c * l[:, 0] - s * l[:, 1],
                            s * l[:, 0] + c * l[:, 1],
                            l[:, 2]))


def ra_dec(l):
    """Right ascension and declination [deg] of each line-of-sight vector (paper Eq. 4)."""
    ra = np.degrees(np.arctan2(l[:, 1], l[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(l[:, 2] / np.linalg.norm(l, axis=1)))
    return ra, dec


l_obs = line_of_sight_inertial(Y_obs[:, :3], tau_obs)
ra_true, dec_true = ra_dec(l_obs)
range_km = np.linalg.norm(l_obs, axis=1) * DU            # NOT observed (ranging is hard)

l_fine = line_of_sight_inertial(Y_fine[:, :3], tau_fine)
ra_fine, dec_fine = ra_dec(l_fine)

# %% 4. Add telescope noise
rng = np.random.default_rng(SEED)
sigma_deg = NOISE_ARCSEC / 3600.0
ra_meas = (ra_true + rng.normal(0, sigma_deg, N_OBS)) % 360.0
dec_meas = dec_true + rng.normal(0, sigma_deg, N_OBS)

# %% 5. Save the observations (IOD input) and the truth (for grading)
os.makedirs("data", exist_ok=True)
t_days = tau_obs * TU / 86400

with open("data/synthetic_obs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["obs_id", "time_days", "ra_deg", "dec_deg", "sigma_arcsec"])
    for i in range(N_OBS):
        w.writerow([i, f"{t_days[i]:.6f}", f"{ra_meas[i]:.8f}", f"{dec_meas[i]:.8f}", NOISE_ARCSEC])

with open("data/synthetic_truth.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["obs_id", "time_days", "x", "y", "z", "vx", "vy", "vz",
                "range_km", "ra_true_deg", "dec_true_deg"])
    for i in range(N_OBS):
        w.writerow([i, f"{t_days[i]:.6f}", *[f"{v:.10f}" for v in Y_obs[i]],
                    f"{range_km[i]:.3f}", f"{ra_true[i]:.8f}", f"{dec_true[i]:.8f}"])

print(f"Truth orbit: L1 halo, {SPAN_DAYS} days, {N_OBS} observations, {NOISE_ARCSEC}\" noise")
print(f"{'obs':>3} {'day':>6} {'RA (deg)':>10} {'Dec (deg)':>10} {'range (km)':>12}")
for i in range(N_OBS):
    print(f"{i:>3} {t_days[i]:6.2f} {ra_meas[i]:10.4f} {dec_meas[i]:10.4f} {range_km[i]:12,.0f}")
print("Saved data/synthetic_obs.csv (IOD input) and data/synthetic_truth.csv (answer key)")

# %% 6. Plots
fig = plt.figure(figsize=(15, 4.8), dpi=100)
axs = [fig.add_subplot(1, 3, 1, projection="3d"), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
days_fine = tau_fine * TU / 86400

# (a) Where the object is, and the lines of sight from Earth (rotating frame, 3D)
ax = axs[0]
full = sol.sol(np.linspace(0, 2.2194, 400))          # one full halo period, for context
ax.plot(full[0], full[1], full[2], color="0.75", lw=1)
ax.plot(Y_fine[:, 0], Y_fine[:, 1], Y_fine[:, 2], "k", lw=1.5, label="Truth orbit (observed arc)")
for i in range(N_OBS):
    ax.plot(*zip(OBSERVER, Y_obs[i, :3]), "r--", lw=0.7)
ax.plot(Y_obs[:, 0], Y_obs[:, 1], Y_obs[:, 2], "rs", ms=5, label="Observation times")
ax.plot([-MU], [0], [0], "bo", ms=8, label="Earth (observer)")
ax.plot([1 - MU], [0], [0], "o", color="gray", ms=5, label="Moon")
ax.set_xlabel("x (DU)")
ax.set_ylabel("y (DU)")
ax.set_zlabel("z (DU)")
ax.set_title("Truth orbit and lines of sight")
ax.view_init(elev=22, azim=-60)
ax.legend(fontsize=7, loc="upper left")

# (b) Right ascension over time
ax = axs[1]
ra_plot = np.degrees(np.unwrap(np.radians(ra_fine)))   # avoid jumps at 360 -> 0
ax.plot(days_fine, ra_plot, "k", lw=1.2, label="True RA")
ra_obs_plot = ra_meas + 360.0 * np.round((np.interp(t_days, days_fine, ra_plot) - ra_meas) / 360.0)
ax.plot(t_days, ra_obs_plot, "rs", ms=7, mfc="none", mew=1.5, label="Noisy observations")
ax.set_xlabel("Days")
ax.set_ylabel("Right ascension (deg)")
ax.set_title("What the telescope measures: RA")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (c) Declination over time
ax = axs[2]
ax.plot(days_fine, dec_fine, "k", lw=1.2, label="True Dec")
ax.plot(t_days, dec_meas, "rs", ms=7, mfc="none", mew=1.5, label="Noisy observations")
ax.set_xlabel("Days")
ax.set_ylabel("Declination (deg)")
ax.set_title("What the telescope measures: Dec")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Synthetic angles-only observations of a cislunar object (observer at Earth's center)",
             fontsize=13)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/synthetic_obs.png", bbox_inches="tight")
plt.show()
