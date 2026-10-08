# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from astropy.time import Time
from astropy.coordinates import get_body, solar_system_ephemeris
import astropy.units as u
import os
from cr3bp import nondim_cr3bp, jacobi       # shared CR3BP functions (cr3bp.py)

# FIX: NASA file lives in the data/ folder. Run from the repo's main folder.
OEM_FILE = "data/Artemis_II_OEM_2026_04_10_Post-ICPS-Sep-to-EI.asc"
START_AFTER = "2026-04-03T01:00:00"   # TLI burn ends ~2026-04-02 23:56 UTC

# %% 1. Read the real trajectory (OEM file)
times, states = [], []
with open(OEM_FILE) as f:
    for line in f:
        parts = line.split()
        # data lines look like: 2026-04-03T01:02:48.802  x y z vx vy vz
        if len(parts) == 7 and parts[0][:2] == "20" and "T" in parts[0]:
            times.append(parts[0])
            states.append([float(p) for p in parts[1:]])

t_real = Time(times, scale="utc")
states = np.array(states)                 # km, km/s (Earth-centered, inertial)

# Keep only the coast after TLI
keep = t_real >= Time(START_AFTER, scale="utc")
t_real, states = t_real[keep], states[keep]
r_real, v_real = states[:, :3], states[:, 3:]
print(f"Using {len(t_real)} points: {t_real[0].iso} -> {t_real[-1].iso} UTC")

# %% 2. Where is the Moon? (astropy's built-in lunar model, no download needed)
def moon_state(t):
    """Moon position [km] and velocity [km/s] relative to Earth, inertial axes."""
    dt = 30 * u.s
    with solar_system_ephemeris.set("builtin"):
        r = get_body("moon", t).cartesian.xyz.to_value(u.km).T
        r_plus = get_body("moon", t + dt).cartesian.xyz.to_value(u.km).T
        r_minus = get_body("moon", t - dt).cartesian.xyz.to_value(u.km).T
    v = (r_plus - r_minus) / (2 * dt.to_value(u.s))   # central difference
    return r, v

r_moon, v_moon = moon_state(t_real)

# %% 3. CR3BP constants (same idea as the textbook, with real numbers)
mu_E = 398600.4418        # Earth GM [km^3/s^2]
mu_M = 4902.800066        # Moon  GM [km^3/s^2]
pi_2 = mu_M / (mu_E + mu_M)                    # mass ratio, ~0.01215

L = np.linalg.norm(r_moon[0])                  # distance unit: Earth-Moon distance at start [km]
n = np.sqrt((mu_E + mu_M) / L**3)              # rotation rate of the frame [rad/s]
T_unit = 1 / n                                 # time unit [s]
V_unit = L * n                                 # velocity unit [km/s]
print(f"pi_2 = {pi_2:.6f},  L = {L:,.0f} km,  time unit = {T_unit/86400:.3f} days,"
      f"  velocity unit = {V_unit:.4f} km/s")

# %% 4. Convert inertial (Earth-centered) states into the rotating, nondimensional frame
def to_rotating(r, v, rm, vm, rotate_velocity=True):
    """
    r, v   : spacecraft position/velocity, Earth-centered inertial [km, km/s]
    rm, vm : Moon position/velocity, Earth-centered inertial
    Returns nondimensional position (and velocity) in the CR3BP rotating frame.
    """
    # Unit vectors of the rotating frame at this instant
    x_hat = rm / np.linalg.norm(rm)                    # points at the Moon
    h = np.cross(rm, vm)                               # Moon's orbital angular momentum
    z_hat = h / np.linalg.norm(h)                      # perpendicular to the Moon's orbit
    y_hat = np.cross(z_hat, x_hat)                     # completes the right-handed set
    C = np.vstack((x_hat, y_hat, z_hat))               # rotation matrix: inertial -> rotating

    # Shift origin from Earth's center to the Earth-Moon barycenter
    r_b, v_b = pi_2 * rm, pi_2 * vm
    rho = C @ (r - r_b)                                # position in rotating axes [km]
    if not rotate_velocity:
        # For plotting the real path, scale by the Earth-Moon distance AT THAT MOMENT.
        # The real Moon's distance varies by about +/-5%, so this keeps it pinned at x* = 1 - pi_2.
        return rho / np.linalg.norm(rm)

    # Velocity seen from the spinning frame: subtract (n x r)
    # FIX: use the CR3BP frame rate n (not the Moon's actual rate) so the model
    # starts with Orion's exact real velocity
    v_rot = C @ (v - v_b) - np.cross([0, 0, n], rho)
    return rho / L, v_rot / V_unit

# Initial condition = Orion's real state at the first post-TLI point
r0_star, v0_star = to_rotating(r_real[0], v_real[0], r_moon[0], v_moon[0])
Y_0 = np.hstack((r0_star, v0_star))
print("Initial state (nondimensional):", np.round(Y_0, 5))

# Whole real trajectory in the rotating frame (for plotting)
r_real_rot = np.array([to_rotating(r_real[i], v_real[i], r_moon[i], v_moon[i],
                                   rotate_velocity=False) for i in range(len(t_real))])

# %% 5. The CR3BP equations of motion
# These now live in cr3bp.py (shared with the other scripts), imported at the top.

# %% 6. Solve -- evaluate at the same instants as the real data
tau_eval = (t_real - t_real[0]).to_value("s") / T_unit   # real times -> nondimensional time
sol = solve_ivp(nondim_cr3bp, [0, tau_eval[-1]], Y_0, t_eval=tau_eval, args=(pi_2,),
                method="DOP853", rtol=1e-12, atol=1e-12)   # tight tolerances (lesson from the page)
r_cr3bp = sol.y.T[:, :3]
v_cr3bp = sol.y.T[:, 3:]

# %% 7. Jacobi constant check (should be flat)
J = jacobi(r_cr3bp, v_cr3bp, pi_2)
print(f"Jacobi constant drift over the run: {np.ptp(J):.2e}")

# %% 8. How far apart are model and reality?
ell = np.linalg.norm(r_moon, axis=1)                  # actual Earth-Moon distance at each time [km]

# FIX: compare in Earth-centered inertial km (same frame as the NASA file).
# The old version scaled the model by ell instead of L, which added fake error.
x_hat0 = r_moon[0] / L
z_hat0 = np.cross(r_moon[0], v_moon[0])
z_hat0 = z_hat0 / np.linalg.norm(z_hat0)
C0 = np.vstack((x_hat0, np.cross(z_hat0, x_hat0), z_hat0))   # same rotation as to_rotating, at t0
rho_E = (r_cr3bp + [pi_2, 0, 0]) * L                 # model position relative to Earth [km], rotating axes
c, s = np.cos(sol.t), np.sin(sol.t)                  # frame has turned by angle tau since t0
rho_E = np.column_stack((c * rho_E[:, 0] - s * rho_E[:, 1], s * rho_E[:, 0] + c * rho_E[:, 1], rho_E[:, 2]))
r_cr3bp_inertial = rho_E @ C0                         # rotating-at-t0 axes -> inertial axes
miss_km = np.linalg.norm(r_cr3bp_inertial - r_real, axis=1)
days = (t_real - t_real[0]).to_value("day")

moon_x = 1 - pi_2
d_moon_real = np.linalg.norm(r_real_rot - [moon_x, 0, 0], axis=1) * ell
d_moon_model = np.linalg.norm(r_cr3bp - [moon_x, 0, 0], axis=1) * L
i_r, i_m = d_moon_real.argmin(), d_moon_model.argmin()
print(f"Closest approach to Moon (center): real {d_moon_real[i_r]:,.0f} km at {t_real[i_r].iso}, "
      f"CR3BP {d_moon_model[i_m]:,.0f} km at {t_real[i_m].iso}")
for d in (1, 2, 3, 4, 5, 6, 7):
    k = np.searchsorted(days, d)
    if k < len(days):
        print(f"  after {d} day(s): model is {miss_km[k]:,.0f} km from the real Orion")

# %% 9. Plots
fig, axs = plt.subplots(2, 2, figsize=(13, 11), dpi=100)

# (a) Whole trip in the rotating frame
ax = axs[0, 0]
ax.plot(r_real_rot[:, 0], r_real_rot[:, 1], "k", lw=2, label="Real Artemis II (NASA OEM)")
ax.plot(r_cr3bp[:, 0], r_cr3bp[:, 1], "r--", lw=1.5, label="CR3BP prediction")
ax.plot(-pi_2, 0, "bo", ms=10, label="Earth")
ax.plot(moon_x, 0, "o", color="gray", ms=6, label="Moon")
ax.plot(*r0_star[:2], "g^", ms=9, label="Start (post-TLI)")
ax.set_aspect("equal")
ax.set_title("Rotating Earth–Moon frame (nondimensional)")
ax.set_xlabel("x*  (Earth–Moon distances)")
ax.set_ylabel("y*")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.3)

# (b) Zoom on the lunar flyby, in km relative to the Moon
ax = axs[0, 1]
win = d_moon_real < 60000
ax.plot((r_real_rot[win, 0] - moon_x) * ell[win], r_real_rot[win, 1] * ell[win], "k", lw=2, label="Real")
win_m = d_moon_model < 60000
ax.plot((r_cr3bp[win_m, 0] - moon_x) * L, r_cr3bp[win_m, 1] * L, "r--", lw=1.5, label="CR3BP")
ax.add_patch(plt.Circle((0, 0), 1737.4, color="gray"))
ax.set_aspect("equal")
ax.set_title("Lunar flyby close-up (km from Moon's center)")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (c) Separation between model and reality
ax = axs[1, 0]
ax.plot(days, miss_km / 1000, "C3")
ax.set_title("How far the CR3BP drifts from the real Orion")
ax.set_xlabel(f"Days since {t_real[0].iso[:16]} UTC")
ax.set_ylabel("Position error (thousand km)")
ax.grid(alpha=0.3)

# (d) Jacobi constant
ax = axs[1, 1]
ax.plot(days, J - J[0], "C0")
ax.set_title("Jacobi constant change (numerical accuracy check)")
ax.set_xlabel("Days")
ax.set_ylabel("J − J₀")
ax.grid(alpha=0.3)

fig.suptitle("Artemis II: textbook CR3BP vs. real trajectory", fontsize=15)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/artemis2_cr3bp_vs_real.png", bbox_inches="tight")
plt.show()

# notes
#   The start time is about an hour after a burn, and minimal burns occured during the modeled flight time. These parameters were chosen to
#   enhance the accuracy of the model. 
#   Inaccuracies are clearly displayed and can be attributed to the following: 
#     - the moon does not follow a circular path
#     - the CR3BP does not account for the significant gravitational influence of the sun
#     - minor course-correction burns

