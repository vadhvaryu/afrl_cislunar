"""Sparse direct transcription of Heidrich & Holzinger (2025), Eqs. 15-38.

IPOPT solves the nonlinear program; CasADi differentiates the complete graph,
including the dependence of midpoint states on endpoint controls.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import chi2

from .models import (CR3BPSystem, ObservationArc, SolverOptions, CollocationFailure,
                     cr3bp_rhs, tangent_basis)
from .transcription import hermite_simpson, state_polynomial, control_polynomial


def _symbolic_rhs(x, mu, u):
    r, v = x[:3], x[3:]
    d1, d2 = r + ca.DM([mu, 0, 0]), r - ca.DM([1-mu, 0, 0])
    a = -(1-mu)*d1/ca.dot(d1,d1)**1.5 - mu*d2/ca.dot(d2,d2)**1.5
    return ca.vertcat(v, a + ca.vertcat(r[0]+2*v[1], r[1]-2*v[0], 0) + u)


def _initial_states(arc, mesh, system, options):
    if options.initial_guess == "los":
        rho = np.clip(1., *options.range_bounds)
        spline = CubicSpline(arc.t, arc.observer + rho*arc.los, axis=0)
        return np.column_stack((spline(mesh), spline(mesh, 1)))
    mu = system.mu
    if options.initial_guess == "L4":
        p = [0.5-mu, np.sqrt(3)/2, 0]
    else:
        def ax(x):
            return cr3bp_rhs(np.array([x,0,0,0,0,0]),mu)[3]
        bounds = (-mu+1e-5, 1-mu-1e-5) if options.initial_guess == "L1" else (1-mu+1e-5, 2.)
        p = [brentq(ax, *bounds), 0, 0]
    return np.tile([*p,0,0,0], (len(mesh),1))


@dataclass
class CollocationResult:
    t: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    increments: np.ndarray
    system: CR3BPSystem
    diagnostics: dict

    @property
    def impulses(self):
        """Physical right-minus-left velocity jumps at t[1:] (DU/TU).

        Eq. (29) writes +B*increment in the defect, so jump = -increment.
        """
        return -self.increments

    def _interval(self, value):
        if not np.isfinite(value) or value < self.t[0] or value > self.t[-1]:
            raise ValueError("Evaluation time must be inside the fitted interval")
        return int(np.clip(np.searchsorted(self.t, value, side="right")-1, 0, len(self.t)-2))

    def state_at(self, t, side="right"):
        """Evaluate Eq. (17); at knots choose the pre/post impulse limit explicitly."""
        if side not in ("right", "left"):
            raise ValueError("side must be right or left")
        values = np.asarray(t, dtype=float)
        out = []
        for value in values.ravel():
            k = self._interval(value)
            j = np.searchsorted(self.t, value)
            if j < len(self.t) and value == self.t[j]:
                x = self.states[j].copy()
                if side == "left" and j > 0:
                    k = j-1
                    h = self.t[k+1]-self.t[k]
                    _, _, f0, fm, f1 = hermite_simpson(
                        self.states[k], self.states[k+1], *self.controls[2*k:2*k+3], h, self.system.mu)
                    x = state_polynomial(self.states[k],f0,fm,f1,h,h)
            else:
                h = self.t[k+1]-self.t[k]
                _, _, f0, fm, f1 = hermite_simpson(
                    self.states[k], self.states[k+1], *self.controls[2*k:2*k+3], h, self.system.mu)
                x = state_polynomial(self.states[k], f0, fm, f1, h, value-self.t[k])
            out.append(x)
        return np.asarray(out).reshape(values.shape+(6,))

    def control_at(self, t):
        values = np.asarray(t, dtype=float)
        out = []
        for value in values.ravel():
            k = self._interval(value)
            out.append(control_polynomial(*self.controls[2*k:2*k+3],
                                          self.t[k+1]-self.t[k], value-self.t[k]))
        return np.asarray(out).reshape(values.shape+(3,))

    def to_dict(self):
        return {"time_tau": self.t.tolist(), "states_rotating": self.states.tolist(),
                "controls_rotating": self.controls.tolist(),
                "impulse_times_tau": self.t[1:].tolist(), "impulses_rotating": self.impulses.tolist(),
                "paper_increments": self.increments.tolist(), "diagnostics": self.diagnostics,
                "system": {"mu": self.system.mu, "du_m": self.system.du_m, "tu_s": self.system.tu_s}}


def solve(arc: ObservationArc, system: CR3BPSystem | None = None,
          options: SolverOptions | None = None, *, mesh=None, initial_states=None) -> CollocationResult:
    """Fit a rotating-frame arc without file I/O, Java, or access to a truth trajectory.

    `mesh` contains absolute nondimensional times covering exactly the data span.
    `initial_states`, when supplied, has shape (len(mesh),6). No endpoint states are
    fixed. All maneuver inputs start at zero. Local optimality is not uniqueness.
    """
    if not isinstance(arc, ObservationArc):
        raise TypeError("arc must be an ObservationArc")
    system, options = system or CR3BPSystem(), options or SolverOptions()
    if options.initial_guess == "auto" and initial_states is None:
        # The NLP is nonconvex. Compare two explicitly documented, truth-free
        # starts using the paper's own objective, with a third only as fallback.
        attempts, candidates = [], []
        for start in ("L2", "los", "L4"):
            if start == "L4" and candidates:
                break
            try:
                candidate = solve(arc,system,replace(options,initial_guess=start),mesh=mesh)
                attempts.append({"initial_guess":start,"success":True,
                                 "objective":candidate.diagnostics["objective"],
                                 "elapsed_seconds":candidate.diagnostics["elapsed_seconds"]})
                candidates.append(candidate)
            except CollocationFailure as exc:
                attempts.append({"initial_guess":start,"success":False,"message":str(exc),
                                 "elapsed_seconds":exc.diagnostics.get("elapsed_seconds",0.)})
        if not candidates:
            raise CollocationFailure("All initial guesses failed; inspect attempts, observation geometry, and mesh",
                                     {"success":False,"attempts":attempts})
        best = min(candidates,key=lambda r:r.diagnostics["objective"])
        best.diagnostics["attempts"] = attempts
        best.diagnostics["elapsed_seconds"] = sum(a["elapsed_seconds"] for a in attempts)
        return best
    mesh = np.linspace(arc.t[0], arc.t[-1], options.nodes) if mesh is None else np.asarray(mesh, dtype=float)
    if mesh.ndim != 1 or len(mesh) < 3 or not np.isfinite(mesh).all() or np.any(np.diff(mesh) <= 0):
        raise ValueError("mesh must contain at least three increasing finite times")
    if mesh[0] != arc.t[0] or mesh[-1] != arc.t[-1]:
        raise ValueError("mesh endpoints must equal the first and last observation times")
    n = len(mesh)
    guess = _initial_states(arc, mesh, system, options) if initial_states is None else np.asarray(initial_states, dtype=float)
    if guess.shape != (n,6) or not np.isfinite(guess).all():
        raise ValueError("initial_states must be finite with shape (nodes,6)")
    # Reject singular initial guesses before starting native optimizer code.
    try:
        cr3bp_rhs(guess, system.mu)
    except ValueError as exc:
        if initial_states is not None:
            raise
        raise CollocationFailure(f"{options.initial_guess} initialization reaches a primary singularity") from exc

    exact = options.measurement_mode == "exact" or (arc.covariance is None and np.all(arc.sigma == 0))
    q = 2 if arc.los_rate is None else 4
    if not exact and arc.covariance is None:
        if q == 4:
            raise ValueError("Bounded angle/rate observations require explicit (N,4,4) covariance")
        if np.any(arc.sigma <= 0):
            raise ValueError("Bounded observations require positive sigma (or explicit covariance)")
        covariance = arc.sigma[:,None,None]**2*np.eye(2)[None,:,:]
    else:
        covariance = arc.covariance
    threshold = float(chi2.ppf(options.confidence, q))
    basis = tangent_basis(arc.los)
    x = ca.SX.sym("states",6,n)
    continuous = options.mode in ("continuous", "hybrid")
    impulsive = options.mode in ("impulsive", "hybrid")
    u = ca.SX.sym("controls",3,2*n-1) if continuous else ca.DM.zeros(3,2*n-1)
    dv = ca.SX.sym("increments",3,n-1) if impulsive else ca.DM.zeros(3,n-1)
    variables, lower, upper, initial = [ca.vec(x)], [np.full(6*n,-np.inf)], [np.full(6*n,np.inf)], [guess.ravel()]

    def add_variable(value, lo, hi, start):
        variables.append(ca.vec(value))
        lower.append(np.full(value.numel(),lo))
        upper.append(np.full(value.numel(),hi))
        initial.append(np.full(value.numel(),start))

    if continuous:
        add_variable(u,-options.control_bound,options.control_bound,0.)
    if impulsive:
        add_variable(dv,-options.impulse_bound,options.impulse_bound,0.)
        positive, negative = ca.SX.sym("positive",3,n-1), ca.SX.sym("negative",3,n-1)
        add_variable(positive,0.,options.impulse_bound,0.)
        add_variable(negative,0.,options.impulse_bound,0.)

    constraints, cl, cu = [], [], []
    def constrain(expr, lo=0., hi=0.):
        constraints.append(ca.vec(expr))
        cl.extend([lo]*expr.numel())
        cu.extend([hi]*expr.numel())

    # Equations (15),(16),(29); compressed midpoint is not a decision variable.
    pieces, defect_expressions = [], []
    cost_c, cost_i = 0, 0
    for k,h in enumerate(np.diff(mesh)):
        defect, xm, f0, fm, f1 = hermite_simpson(x[:,k],x[:,k+1],u[:,2*k],u[:,2*k+1],u[:,2*k+2],
                                              h,system.mu,rhs=_symbolic_rhs)
        defect += ca.vertcat(ca.DM.zeros(3),dv[:,k])
        constrain(defect)
        defect_expressions.append(defect)
        pieces.append((f0,fm,f1))
        if continuous:
            cost_c += h*(ca.sumsqr(u[:,2*k])+4*ca.sumsqr(u[:,2*k+1])+ca.sumsqr(u[:,2*k+2]))/6
            # Quadratic Bezier convex hull guarantees the compact control bound
            # between nodes as well as at the collocation samples.
            constrain(2*u[:,2*k+1]-(u[:,2*k]+u[:,2*k+2])/2,
                      -options.control_bound,options.control_bound)
        # Avoid collocating through a primary singularity. Numerical exclusion
        # radius, not an Earth/Moon surface or an inferred physical admissible region.
        for state in (x[:,k],xm):
            for primary in (-system.mu,1-system.mu):
                constrain(ca.sumsqr(state[:3]-ca.DM([primary,0,0])),1e-10,np.inf)
    for primary in (-system.mu,1-system.mu):
        constrain(ca.sumsqr(x[:3,-1]-ca.DM([primary,0,0])),1e-10,np.inf)
    if impulsive:
        constrain(positive-negative-dv)
        cost_i = ca.sum1(ca.vec(positive+negative))
    objective = (options.hybrid_weight*cost_i+(1-options.hybrid_weight)*cost_c if options.mode == "hybrid"
                 else cost_i if impulsive else cost_c)

    measurement_expressions = []
    for i,t in enumerate(arc.t):
        node = np.searchsorted(mesh,t)
        if node < n and mesh[node] == t:
            xi = x[:,node]  # post-impulse at an exact knot
        else:
            k = int(np.clip(node-1,0,n-2))
            xi = state_polynomial(x[:,k],*pieces[k],mesh[k+1]-mesh[k],t-mesh[k])
        relative = xi[:3]-arc.observer[i]
        distance = ca.sqrt(ca.sumsqr(relative))
        direction = relative/distance
        constrain(distance,*options.range_bounds)
        constrain(ca.dot(direction,arc.los[i]),0.,1.00000001)
        residual = ca.mtimes(basis[i],direction-arc.los[i])
        if q == 4:
            relative_v = xi[3:]-arc.observer_velocity[i]
            direction_rate = (relative_v-direction*ca.dot(direction,relative_v))/distance
            residual = ca.vertcat(residual,ca.mtimes(basis[i],direction_rate-arc.los_rate[i]))
        if exact:
            constrain(residual)
            measurement_expressions.append(residual)
        else:
            whiten = np.linalg.inv(np.linalg.cholesky(covariance[i]))
            mahal = ca.sumsqr(ca.mtimes(whiten,residual))
            constrain(mahal/threshold,-np.inf,1.)
            measurement_expressions.append(mahal)

    z, g = ca.vertcat(*variables),ca.vertcat(*constraints)
    solver_options = {"print_time":False,"error_on_fail":False,
                      "ipopt.print_level":options.print_level,"ipopt.sb":"yes",
                      "ipopt.max_iter":options.max_iterations,"ipopt.tol":options.tolerance,
                      "ipopt.constr_viol_tol":min(options.tolerance,options.feasibility_tolerance),
                      "ipopt.acceptable_iter":0,"ipopt.bound_relax_factor":0.,
                      "ipopt.mu_strategy":"adaptive"}
    started = time.perf_counter()
    solver = ca.nlpsol("collocation","ipopt",{"x":z,"f":objective,"g":g},solver_options)
    try:
        solution = solver(x0=np.concatenate(initial),lbx=np.concatenate(lower),ubx=np.concatenate(upper),lbg=cl,ubg=cu)
    except RuntimeError as exc:
        raise CollocationFailure(f"IPOPT could not solve the collocation problem: {exc}") from exc
    stats = solver.stats()
    zv, gv = np.asarray(solution["x"]).ravel(),np.asarray(solution["g"]).ravel()
    violation = max(float(np.max(np.maximum(np.array(cl)-gv,0))),
                    float(np.max(np.maximum(gv-np.array(cu),0))),
                    float(np.max(np.maximum(np.concatenate(lower)-zv,0))),
                    float(np.max(np.maximum(zv-np.concatenate(upper),0))))
    diagnostics = {"success":bool(stats.get("success",False)),"status":str(stats.get("return_status","unknown")),
                   "iterations":int(stats.get("iter_count",0)),"objective":float(solution["f"]),
                   "max_constraint_violation":violation,"elapsed_seconds":time.perf_counter()-started,
                   "mode":options.mode,"measurement_mode":"exact" if exact else "bounded",
                   "initial_guess":"supplied" if initial_states is not None else options.initial_guess,
                   "nodes":n,"observations":len(arc.t),"decision_variables":int(z.numel()),
                   "constraints":int(g.numel()),"chi_square_threshold":None if exact else threshold}
    history = stats.get("iterations",{})
    for key in ("inf_pr","inf_du","mu"):
        if key in history and len(history[key]):
            diagnostics[key] = float(history[key][-1])
    if (not diagnostics["success"] or not np.isfinite(zv).all() or not np.isfinite(gv).all()
            or not np.isfinite(diagnostics["objective"]) or violation > options.feasibility_tolerance):
        diagnostics["success"] = False
        raise CollocationFailure(f"Collocation failed: {diagnostics['status']}; constraint violation {violation:.3g}",diagnostics)

    evaluate = ca.Function("extract",[z],[x,u,dv,ca.vertcat(*defect_expressions),
                                        ca.vertcat(*measurement_expressions),cost_c,cost_i])
    xs,us,dvs,defects,measures,jc,ji = evaluate(zv)
    diagnostics.update(max_collocation_defect=float(np.max(np.abs(np.asarray(defects)))),
                       continuous_cost=float(jc),impulsive_l1_cost=float(ji))
    if exact:
        diagnostics["max_measurement_residual"] = float(np.max(np.abs(np.asarray(measures))))
    else:
        diagnostics["max_measurement_mahalanobis_squared"] = float(np.max(np.asarray(measures)))
    result = CollocationResult(mesh.copy(),np.asarray(xs).T,np.asarray(us).T,np.asarray(dvs).T,system,diagnostics)
    diagnostics["impulsive_delta_v_m_s"] = float(np.linalg.norm(result.impulses,axis=1).sum()*system.du_m/system.tu_s)
    # Report angular residuals and an interior mesh diagnostic independently of the
    # NLP equalities. A small collocation defect alone does not establish accuracy.
    predicted = result.state_at(arc.t)[:,:3]-arc.observer
    predicted /= np.linalg.norm(predicted,axis=1)[:,None]
    angles = np.arctan2(np.linalg.norm(np.cross(predicted,arc.los),axis=1),np.sum(predicted*arc.los,axis=1))
    diagnostics["max_angular_residual_rad"] = float(angles.max())
    interior, scaled, sample_times = [], [], []
    for k,h in enumerate(np.diff(mesh)):
        _,_,f0,fm,f1 = hermite_simpson(result.states[k],result.states[k+1],*result.controls[2*k:2*k+3],h,system.mu)
        for s in (.25,.5,.75):
            xp = state_polynomial(result.states[k],f0,fm,f1,h,s*h)
            derivative = f0+(-3*f0+4*fm-f1)*s+(2*f0-4*fm+2*f1)*s*s
            up = control_polynomial(*result.controls[2*k:2*k+3],h,s*h)
            try:
                residual = np.abs(derivative-cr3bp_rhs(xp,system.mu,up))
            except ValueError as exc:
                diagnostics["success"] = False
                diagnostics["mesh_check_passed"] = False
                raise CollocationFailure("Interpolated trajectory reaches a primary singularity; refine the mesh", diagnostics) from exc
            interior.append(np.max(residual))
            scaled.append(np.max(h*residual/(1+np.abs(xp))))
            sample_times.append(float(mesh[k]+s*h))
    diagnostics["max_interior_dynamics_residual"] = float(max(interior))
    diagnostics["max_scaled_mesh_error"] = float(max(scaled))
    diagnostics["worst_mesh_sample_tau"] = sample_times[int(np.argmax(scaled))]
    diagnostics["mesh_tolerance"] = options.mesh_tolerance
    diagnostics["mesh_check_passed"] = bool(np.isfinite(scaled).all() and max(scaled) <= options.mesh_tolerance)
    if not diagnostics["mesh_check_passed"]:
        diagnostics["success"] = False
        raise CollocationFailure(
            f"Collocation mesh is unresolved (sampled error {max(scaled):.3g} exceeds "
            f"{options.mesh_tolerance:.3g}). Increase nodes, refine mesh, or change the initial guess.", diagnostics)
    return result
