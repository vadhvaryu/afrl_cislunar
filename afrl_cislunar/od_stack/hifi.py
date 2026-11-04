"""
High-fidelity truth and the model-mismatch experiment: how much does fitting real-world-like
data with the CR3BP cost, and how much do the usual robustness tools win back?

    python -m od_stack.hifi build                 # truth + observations -> results/hifi/<orbit>/
    python -m od_stack.hifi run --seeds 20        # every estimator variant on the same cases
    python -m od_stack.hifi run --orbit NRHO_9_2  # another orbit from orbit_catalog.py

Truth (Orekit numerical propagation, real GCRF, real UTC epoch, DE440 ephemerides from orekit-data):
    Earth point mass + Earth J2, Moon, Sun, Venus, Mars, Jupiter, Saturn, and solar radiation
    pressure on a 1,000 kg spacecraft with 15 m^2 (A/m = 0.015 m^2/kg), Cr = 1.3, Earth and Moon shadows.
Initial state: the orbit's CR3BP state at tau = 0 mapped into the real Earth-Moon geometry at the
    epoch (instantaneous Moon distance D, rate w and dD/dt): r = M D rho, v = M (dD/dt rho + D w (rho' + z x rho)),
    with M the EMI axes in GCRF (x to the Moon, z along the Earth-Moon angular momentum). This is what
    a CR3BP-designed orbit looks like in the real system without station-keeping.
Observations: same recipe as obs_generator.py (5 RA/Dec pairs over 8 days from Earth's center,
    Gaussian noise in degrees, all RA draws then all Dec draws), but the noise-free angles come from
    Orekit's AngularRaDec on the truth, so they include light time, like real data.

EMI: the stack's frame is pinned to the real Moon at the epoch (EMI = M^T GCRF). Variants that use
the CR3BP fit the observations rotated into EMI; everything is graded in GCRF against the truth.

Estimator variants (all Orekit BatchLSEstimator + Levenberg-Marquardt, same measurement model, so
the only difference is the force model):
    cr3bp_nominal      the stack's CR3BP (DU 384,338 km, TU 375,099 s)
    cr3bp_epoch        CR3BP scaled to the Moon's actual distance and rate at the epoch
    cr3bp_epoch_emp    cr3bp_epoch + 3 estimated constant inertial accelerations (empirical forces)
    earth_moon         Earth + real (DE440) Moon
    sun_moon           Earth + real Moon + Sun
    full_no_srp        everything except solar radiation pressure (spacecraft unknown)
    full_cr_estimated  everything, with the reflection coefficient estimated (starts at 1.0, truth 1.3)
    full               everything, matching the truth exactly (upper bound)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys as _sys

import numpy as np

from .frames import EarthMoonSystem, rot_z

DAY = 86400.0
OUT_DIR = "results/hifi"
EPOCH_UTC = (2026, 10, 26, 0, 0, 0.0)
OBS_DAYS = np.linspace(0.0, 8.0, 5)
MASS_KG, AREA_M2, CR_TRUE = 1000.0, 15.0, 1.3
VARIANTS = ["cr3bp_nominal", "cr3bp_epoch", "cr3bp_epoch_emp", "earth_moon", "sun_moon", "full_no_srp",
            "full_cr_estimated", "full"]


# ----------------------------------------------------------------------------- Orekit pieces

def _orekit():
    from .orekit_setup import start_orekit

    start_orekit()


def epoch_date(epoch_utc=EPOCH_UTC):
    _orekit()
    from org.orekit.time import AbsoluteDate, TimeScalesFactory

    return AbsoluteDate(*epoch_utc[:5], float(epoch_utc[5]), TimeScalesFactory.getUTC())


def _v3(a):
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    return Vector3D(*map(float, a))


def _arr(v):
    return np.array([v.getX(), v.getY(), v.getZ()])


def earth_moon_geometry(epoch):
    """Real Moon (DE440) relative to Earth at the epoch: (M = EMI axes in GCRF, D [m], w [rad/s], dD/dt [m/s])."""
    from org.orekit.bodies import CelestialBodyFactory
    from org.orekit.frames import FramesFactory

    pv = CelestialBodyFactory.getMoon().getPVCoordinates(epoch, FramesFactory.getGCRF())
    rm, vm = _arr(pv.getPosition()), _arr(pv.getVelocity())
    D = np.linalg.norm(rm)
    h = np.cross(rm, vm)
    x, z = rm / D, h / np.linalg.norm(h)
    return np.column_stack([x, np.cross(z, x), z]), D, np.linalg.norm(h) / D**2, rm @ vm / D


def force_models(level: str, cr: float = CR_TRUE, estimate_cr: bool = False):
    """Orekit force models (besides Earth's point mass, which the orbit's mu provides)."""
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import Constants, IERSConventions

    moon, sun = CelestialBodyFactory.getMoon(), CelestialBodyFactory.getSun()
    forces = [ThirdBodyAttraction(moon)]
    if level in ("sun_moon", "full_no_srp", "full"):
        forces.append(ThirdBodyAttraction(sun))
    if level in ("full_no_srp", "full"):
        for b in (CelestialBodyFactory.getVenus(), CelestialBodyFactory.getMars(), CelestialBodyFactory.getJupiter(),
                  CelestialBodyFactory.getSaturn()):
            forces.append(ThirdBodyAttraction(b))
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        forces.append(HolmesFeatherstoneAttractionModel(itrf, GravityFieldFactory.getNormalizedProvider(2, 0)))
    if level == "full":
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf)
        sc = IsotropicRadiationSingleCoefficient(AREA_M2, float(cr))
        srp = SolarRadiationPressure(sun, earth, sc)
        srp.addOccultingBody(moon, Constants.MOON_EQUATORIAL_RADIUS)
        if estimate_cr:
            for d in srp.getParametersDrivers():
                if "reflection" in d.getName().lower():
                    d.setSelected(True)
        forces.append(srp)
    return forces


def _integrator_builder(position_tolerance_m=1.0):
    from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder

    return DormandPrince853IntegratorBuilder(1e-3, 3600.0, float(position_tolerance_m))


def gcrf_builder(epoch, t, r, v, level, cr=CR_TRUE, estimate_cr=False, position_tolerance_m=1.0):
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import CartesianOrbit, PositionAngleType
    from org.orekit.propagation.conversion import NumericalPropagatorBuilder
    from org.orekit.utils import Constants, PVCoordinates

    orbit = CartesianOrbit(PVCoordinates(_v3(r), _v3(v)), FramesFactory.getGCRF(), epoch.shiftedBy(float(t)),
                           Constants.EIGEN5C_EARTH_MU)
    b = NumericalPropagatorBuilder(orbit, _integrator_builder(position_tolerance_m), PositionAngleType.TRUE, 1.0)
    b.setMass(MASS_KG)
    for f in force_models(level, cr, estimate_cr):
        b.addForceModel(f)
    return b


def add_empirical(builder, ref_date):
    """3 estimated constant accelerations along the inertial x, y, z axes (empirical forces)."""
    from org.orekit.forces.empirical import ParametricAcceleration, PolynomialAccelerationModel

    for name, axis in (("emp_x", (1, 0, 0)), ("emp_y", (0, 1, 0)), ("emp_z", (0, 0, 1))):
        model = PolynomialAccelerationModel(name, ref_date, 0)
        acc = ParametricAcceleration(_v3(axis), True, model)
        for d in acc.getParametersDrivers():
            d.setScale(1e-9)
            d.setSelected(True)
        builder.addForceModel(acc)
    return builder


def radec_measurements(t, ra, dec, sigma, epoch, frame):
    from jpype import JArray, JDouble
    from org.orekit.estimation.measurements import AngularRaDec, ObservableSatellite

    from .orekit_models import geocenter_station

    station, sat = geocenter_station(), ObservableSatellite(0)
    out = []
    for ti, a, d, s in zip(t, ra, dec, sigma):
        a = float(np.arctan2(np.sin(a), np.cos(a)))
        out.append(AngularRaDec(station, frame, epoch.shiftedBy(float(ti)), JArray(JDouble)([a, float(d)]),
                                JArray(JDouble)([float(s), float(s)]), JArray(JDouble)([1.0, 1.0]), sat))
    return out


def run_bls(builder, measurements, frame, max_iterations=50, max_evaluations=100, threshold=1e-3):
    """Orekit BLS; returns (r, v, 6x6 covariance or None, rms, iterations, estimated propagation params, propagator)."""
    from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer
    from org.orekit.estimation.leastsquares import BatchLSEstimator

    est = BatchLSEstimator(LevenbergMarquardtOptimizer(), builder)
    est.setParametersConvergenceThreshold(threshold)
    est.setMaxIterations(max_iterations)
    est.setMaxEvaluations(max_evaluations)
    for m in measurements:
        est.addMeasurement(m)
    prop = est.estimate()[0]
    pv = prop.getInitialState().getPVCoordinates(frame)
    try:
        C = est.getPhysicalCovariances(1e-12)
        cov = np.array([[C.getEntry(i, j) for j in range(6)] for i in range(6)])
    except Exception:
        cov = None
    params = {d.getName(): float(d.getValue()) for d in est.getPropagatorParametersDrivers(True).getDrivers()}
    return (_arr(pv.getPosition()), _arr(pv.getVelocity()), cov, float(est.getOptimum().getRMS()),
            int(est.getIterationsCount()), params, prop)


def _ephemeris(prop, t0_date, t_end):
    gen = prop.getEphemerisGenerator()
    prop.propagate(t0_date, t0_date.shiftedBy(float(t_end)))
    return gen.getGeneratedEphemeris()


# ----------------------------------------------------------------------------- scenario (truth + data)

class Scenario:
    """Truth trajectory, geometry and the observation recipe for one orbit and epoch."""

    def __init__(self, orbit: str = "L1_halo_truth", epoch_utc=EPOCH_UTC, sys: EarthMoonSystem | None = None):
        _orekit()
        from org.orekit.frames import FramesFactory

        from .orbit_catalog import ORBITS, correct_periodic

        self.sys = sys or EarthMoonSystem()
        self.orbit, self.epoch = orbit, epoch_date(epoch_utc)
        self.gcrf = FramesFactory.getGCRF()
        seed, planar, _ = ORBITS[orbit]
        x_rot = np.array(seed, float) if orbit == "L1_halo_truth" else correct_periodic(seed, self.sys.mu, planar)[0]
        self.M, self.D, self.w, self.Ddot = earth_moon_geometry(self.epoch)
        rho, rhod = x_rot[:3] - self.sys.earth_rot, x_rot[3:]
        self.r0 = self.M @ (rho * self.D)
        self.v0 = self.M @ (self.Ddot * rho + self.D * self.w * (rhod + np.cross([0, 0, 1], rho)))
        b = gcrf_builder(self.epoch, 0.0, self.r0, self.v0, "full", position_tolerance_m=1e-3)
        self.truth_eph = _ephemeris(b.buildPropagator(b.getSelectedNormalizedParameters()), self.epoch, 12 * DAY)
        self.t_obs = OBS_DAYS * DAY
        self.epoch_sys = EarthMoonSystem(mu=self.sys.mu, du_km=self.D / 1e3, tu_s=1.0 / self.w)

    def truth(self, t):
        """GCRF (r, v) arrays at times t [s since epoch]."""
        t = np.atleast_1d(np.asarray(t, float))
        pv = [self.truth_eph.getPVCoordinates(self.epoch.shiftedBy(float(x)), self.gcrf) for x in t]
        return np.array([_arr(p.getPosition()) for p in pv]), np.array([_arr(p.getVelocity()) for p in pv])

    def noise_free_angles(self):
        """Orekit's AngularRaDec of the truth at the observation times (includes light time), GCRF."""
        from jpype import JArray
        from org.orekit.propagation import SpacecraftState

        meas = radec_measurements(self.t_obs, np.zeros(5), np.zeros(5), np.ones(5), self.epoch, self.gcrf)
        out = []
        for m in meas:
            for d in m.getParametersDrivers():
                if d.getReferenceDate() is None:
                    d.setReferenceDate(self.epoch)
            st = self.truth_eph.propagate(m.getDate())
            e = m.estimateWithoutDerivatives(JArray(SpacecraftState)([st])).getEstimatedValue()
            out.append((e[0] % (2 * np.pi), e[1]))
        return np.array(out)

    def observations(self, noise_arcsec: float, seed: int):
        """Noisy RA/Dec in GCRF [rad], same noise recipe as obs_generator.py."""
        ang = np.degrees(self.noise_free_angles())
        rng = np.random.default_rng(seed)
        s = noise_arcsec / 3600.0
        ra = (ang[:, 0] + rng.normal(0, s, 5)) % 360.0
        dec = ang[:, 1] + rng.normal(0, s, 5)
        return np.radians(ra), np.radians(dec), np.full(5, np.radians(s))

    def to_emi_angles(self, ra, dec):
        from .frames import radec_to_los

        los = radec_to_los(ra, dec) @ self.M                   # rows: M^T u
        return np.arctan2(los[:, 1], los[:, 0]) % (2 * np.pi), np.arcsin(np.clip(los[:, 2], -1, 1))

    def write(self, out_dir: str, noise_arcsec=10.0, seed=1) -> dict:
        """Truth (GCRF, hourly) and observations (GCRF) to CSV, plus the geometry, for the record."""
        os.makedirs(out_dir, exist_ok=True)
        tt = np.arange(0, 8 * DAY + 1, 3600.0)
        r, v = self.truth(tt)
        with open(os.path.join(out_dir, "truth_gcrf.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_days", "x_km", "y_km", "z_km", "vx_m_s", "vy_m_s", "vz_m_s"])
            for i, t in enumerate(tt):
                w.writerow([f"{t / DAY:.6f}", *[f"{x / 1e3:.6f}" for x in r[i]], *[f"{x:.6f}" for x in v[i]]])
        ra, dec, sig = self.observations(noise_arcsec, seed)
        with open(os.path.join(out_dir, "obs_gcrf.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["obs_id", "time_days", "ra_deg", "dec_deg", "sigma_arcsec"])
            for i in range(5):
                w.writerow([i, f"{OBS_DAYS[i]:.6f}", f"{np.degrees(ra[i]):.10f}", f"{np.degrees(dec[i]):.10f}", noise_arcsec])
        info = {"orbit": self.orbit, "epoch_utc": str(self.epoch), "moon_distance_km": self.D / 1e3,
                "moon_rate_rad_s": self.w, "moon_ddot_m_s": self.Ddot, "emi_axes_in_gcrf": self.M.tolist(),
                "truth_forces": "Earth (mu + J2), DE440 Moon, Sun, Venus, Mars, Jupiter, Saturn, SRP "
                                f"(A {AREA_M2} m^2, m {MASS_KG} kg, Cr {CR_TRUE}, Earth+Moon shadow)"}
        with open(os.path.join(out_dir, "scenario.json"), "w") as f:
            json.dump(info, f, indent=2)
        return info


# ----------------------------------------------------------------------------- fitting

def fit(sc: Scenario, variant: str, t_e: float, r0, v0, ra, dec, sigma):
    """Fit one variant from the GCRF guess (r0, v0) at t_e. Returns a dict with the GCRF state and
    covariance at t_e, rms, iterations, estimated parameters, and a function giving GCRF positions."""
    from org.orekit.frames import FramesFactory
    from org.orekit.time import AbsoluteDate

    from .orekit_models import propagator_builder

    if variant.startswith("cr3bp"):
        sys = sc.sys if variant == "cr3bp_nominal" else sc.epoch_sys
        fake_epoch, frame = AbsoluteDate.J2000_EPOCH, FramesFactory.getGCRF()   # the stack's EMI convention
        b = propagator_builder(sys, t_e, sc.M.T @ r0, sc.M.T @ v0)
        if variant.endswith("_emp"):
            add_empirical(b, fake_epoch.shiftedBy(float(t_e)))
        ra_e, dec_e = sc.to_emi_angles(ra, dec)
        meas = radec_measurements(sc.t_obs, ra_e, dec_e, sigma, fake_epoch, frame)
        r, v, cov, rms, it, params, prop = run_bls(b, meas, frame)
        M6 = np.kron(np.eye(2), sc.M)

        def positions(t):
            out = []
            for x in t:
                st = prop.propagate(fake_epoch.shiftedBy(float(x)))
                out.append(sc.M @ _arr(st.getPVCoordinates(frame).getPosition()))
            return np.array(out)

        return {"r": sc.M @ r, "v": sc.M @ v, "cov": None if cov is None else M6 @ cov @ M6.T, "rms": rms,
                "iterations": it, "params": params, "positions": positions}
    level = {"earth_moon": "earth_moon", "sun_moon": "sun_moon", "full_no_srp": "full_no_srp",
             "full_cr_estimated": "full", "full": "full"}[variant]
    est_cr = variant == "full_cr_estimated"
    b = gcrf_builder(sc.epoch, t_e, r0, v0, level, cr=1.0 if est_cr else CR_TRUE, estimate_cr=est_cr)
    meas = radec_measurements(sc.t_obs, ra, dec, sigma, sc.epoch, sc.gcrf)
    r, v, cov, rms, it, params, prop = run_bls(b, meas, sc.gcrf)

    def positions(t):
        return np.array([_arr(prop.propagate(sc.epoch.shiftedBy(float(x))).getPVCoordinates(sc.gcrf).getPosition())
                         for x in t])

    return {"r": r, "v": v, "cov": cov, "rms": rms, "iterations": it, "params": params, "positions": positions}


# ----------------------------------------------------------------------------- experiment

def init_worker(orbit: str):
    import warnings

    from . import parallel

    warnings.filterwarnings("ignore")
    parallel.WORKER["scenario"] = Scenario(orbit)


def do_job(job: dict) -> dict:
    import time

    from . import parallel
    from .convergence import directions

    sc = parallel.WORKER["scenario"]
    row = dict(job)
    t0 = time.time()
    t_e = float(sc.t_obs[2])
    rt, vt = sc.truth(t_e)
    rt, vt = rt[0], vt[0]
    u_r, u_v = directions(job["seed"])
    r0, v0 = rt + job["start_km"] * 1e3 * u_r, vt + job["start_m_s"] * u_v
    ra, dec, sig = sc.observations(job["noise_arcsec"], 10_000 + int(job["seed"]))
    try:
        f = fit(sc, job["variant"], t_e, r0, v0, ra, dec, sig)
    except Exception as e:
        row.update(outcome="failed", error=str(e).splitlines()[0][:160], seconds=time.time() - t0)
        return row
    e = np.r_[f["r"] - rt, f["v"] - vt]
    tc = np.linspace(0, 8 * DAY, 49)
    try:
        err_t = np.linalg.norm(f["positions"](tc) - sc.truth(tc)[0], axis=1) / 1e3
        row["max_position_error_km"] = float(err_t.max())
    except Exception as ex:
        row["max_position_error_km"] = float("nan")
        row["error"] = f"propagation: {str(ex).splitlines()[0][:100]}"
    row.update(position_error_km=float(np.linalg.norm(e[:3]) / 1e3), velocity_error_m_s=float(np.linalg.norm(e[3:])),
               rms_normalized_residual=f["rms"], iterations=f["iterations"],
               passes_gate=bool(f["rms"] <= 3.0), seconds=time.time() - t0,
               **{f"param_{k}": v for k, v in f["params"].items()})
    if f["cov"] is not None:
        try:
            row["nees_6d"] = float(e @ np.linalg.solve(f["cov"], e))
            row["sigma_position_km"] = float(np.sqrt(np.trace(f["cov"][:3, :3])) / 1e3)
        except np.linalg.LinAlgError:
            pass
    row["outcome"] = ("converged" if row["position_error_km"] < 1000 * max(1.0, job["noise_arcsec"] / 10) and row["passes_gate"]
                      else "rejected_by_gate" if not row["passes_gate"] else "wrong_orbit")
    return row


def jobs(variants, noises, seeds, start_km=1000.0, start_m_s=10.0):
    return [{"variant": v, "noise_arcsec": float(n), "seed": int(s), "start_km": float(start_km),
             "start_m_s": float(start_m_s)} for v in variants for n in noises for s in seeds]


def summarize(rows):
    out = {}
    for v in VARIANTS:
        for n in sorted({float(r["noise_arcsec"]) for r in rows}):
            g = [r for r in rows if r["variant"] == v and float(r["noise_arcsec"]) == n]
            if not g:
                continue

            def col(k):
                return np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in g])

            ok = np.array([r.get("outcome") == "converged" for r in g])
            nees = col("nees_6d")
            out.setdefault(v, []).append({
                "noise_arcsec": n, "runs": len(g), "converged": int(ok.sum()),
                "passes_gate": int(sum(str(r.get("passes_gate")) == "True" for r in g)),
                "median_rms": float(np.nanmedian(col("rms_normalized_residual"))),
                "median_epoch_error_km": float(np.nanmedian(col("position_error_km"))),
                "median_max_error_km": float(np.nanmedian(col("max_position_error_km"))),
                "median_sigma_km": float(np.nanmedian(col("sigma_position_km"))),
                "mean_nees": float(np.nanmean(nees)) if np.isfinite(nees).any() else np.nan,
                "inside_95": f"{int(np.nansum(nees < 12.59))}/{int(np.isfinite(nees).sum())}",
                "mean_seconds": float(np.nanmean(col("seconds")))})
    return out


def markdown(summ, orbit):
    lines = [f"## Model mismatch: {orbit} (truth: full model; start 1,000 km / 10 m/s)\n",
             "| Estimator model | Noise (″) | Passes 3σ gate | Converged | Median RMS (σ) | Median epoch error (km) | "
             "Median max error, 8 d (km) | Median 1σ (km) | Mean NEES (expect 6) | Inside 95% | s/run |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, rows in summ.items():
        for a in rows:
            lines.append(f"| {v} | {a['noise_arcsec']:g} | {a['passes_gate']}/{a['runs']} | {a['converged']}/{a['runs']} | "
                         f"{a['median_rms']:.2f} | {a['median_epoch_error_km']:,.1f} | {a['median_max_error_km']:,.1f} | "
                         f"{a['median_sigma_km']:,.1f} | {a['mean_nees']:,.1f} | {a['inside_95']} | {a['mean_seconds']:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="High-fidelity truth and model-mismatch experiment")
    ap.add_argument("which", choices=["build", "run"])
    ap.add_argument("--orbit", default="L1_halo_truth")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--noise", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    out = os.path.join(OUT_DIR, a.orbit)
    if a.which == "build":
        _orekit()
        print(json.dumps(Scenario(a.orbit).write(out), indent=2))
        return 0
    from . import parallel

    rows = parallel.run_jobs(jobs(a.variants, a.noise, range(a.seeds)), os.path.join(out, "runs.csv"),
                             ("variant", "noise_arcsec", "seed", "start_km"), do_job, init=init_worker,
                             init_args=(a.orbit,), workers=a.workers, label=f"hifi {a.orbit}")
    summ = summarize(rows)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    md = markdown(summ, a.orbit)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
