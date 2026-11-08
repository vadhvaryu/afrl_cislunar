# Paper-to-code mapping

This package implements the direct-transcription method described by Casey Heidrich
and Marcus Holzinger, **Initial Orbit Determination for Cislunar Objects with
Unknown Maneuvers via Collocation and Nonlinear Programming**, *The Journal of the
Astronautical Sciences* 72, 38 (2025),
[doi:10.1007/s40295-025-00513-7](https://doi.org/10.1007/s40295-025-00513-7).
The paper is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
This is an independent implementation, not the authors' software.

## Dynamics and units: Equations 1–7

The core uses the barycentric Earth–Moon rotating frame and nondimensional units.
The state is `x = [rx, ry, rz, vx, vy, vz]`. Earth is at `[-mu, 0, 0]` and the Moon
at `[1-mu, 0, 0]`. The CR3BP acceleration includes gravity, centrifugal, and
Coriolis terms. Continuous controls add directly to acceleration.

The defaults are the paper's constants: `mu = 0.0121505`,
`DU = 384338000 m`, and `TU = 375099 s`. Therefore velocity, acceleration, and
impulse units are respectively `DU/TU`, `DU/TU**2`, and `DU/TU`.

Observations supply a unit line-of-sight vector `l`, observer position `o`, and time
at each observation. The predicted line of sight is
`d = (r-o)/norm(r-o)`. Optional line-of-sight rates are modeled as

```text
d_dot = (I - d d^T) (v - o_dot) / norm(r-o).
```

This is the vector form of the position/velocity relations in Equations 6–7. Rate
data are supplied by the caller; the solver does not infer measured rates by
differencing sparse angles.

The stack adapter converts Earth-centered inertial (EMI) measurements into this
rotating frame. The rotation angle is the **absolute** `t/TU` in the stack's time
convention. Selecting an observation subset does not reset this phase. EMI here is
the simulation frame defined by `od_stack`, not automatically a real astronomical
GCRF frame.

## Compressed Hermite–Simpson transcription: Equations 14–18

There are `N` state nodes and `N-1` intervals of widths `h_k`. State midpoints are
computed rather than added as independent decision variables:

```text
f_k     = f(x_k, u_k)
x_mid   = (x_k + x_next)/2 + h_k*(f_k - f_next)/8                 (16)
f_mid   = f(x_mid, u_mid)
g_k     = x_next - x_k - h_k*(f_k + 4*f_mid + f_next)/6          (15)
```

For a continuous-control problem, `g_k = 0`. Controls at every node and midpoint
are independent variables, allowing the quadratic control interpolation in
Equation 18. The ordering is
`[u_0, u_mid_0, u_1, u_mid_1, ..., u_last]`.

For an observation inside an interval, `s = t_obs - t_k`, Equation 17 gives

```text
x(s) = x_k + f_k*s
       + (-3*f_k + 4*f_mid - f_next)*s**2/(2*h_k)
       + (2*f_k - 4*f_mid + 2*f_next)*s**3/(3*h_k**2).
```

Thus an observation need not coincide with a mesh node. The same interpolation is
used when evaluating the fitted trajectory. A custom mesh's endpoints must equal
the first and last observation times.

## Continuous maneuvers: Equations 19–22

The objective is Simpson quadrature of squared continuous acceleration:

```text
J_C = sum_k h_k/6 * (dot(u_k,u_k) + 4*dot(u_mid,u_mid)
                     + dot(u_next,u_next)).
```

This is a minimum-control-energy objective, not a least-squares observation
objective. Observation agreement is imposed as constraints. `J_C` is also not
total delta-v: integrating acceleration magnitude would be a different functional.

## Impulsive maneuvers: Equations 29–35

The package follows the **printed sign** of Equation 29:

```text
g_k + B*increment_k = 0,    B = vertical_stack(zeros(3,3), eye(3)).
```

For each interval there is a three-component increment and nonnegative slack
vectors `s_plus`, `s_minus` satisfying

```text
s_plus - s_minus - increment = 0
J_I = sum(s_plus + s_minus).
```

At an optimum this is the componentwise L1 norm of the increments. It is not the
sum of their Euclidean magnitudes. No absolute-value smoothing is substituted.

### Impulse sign, timing, and interpolation caveat

With the printed plus sign, Equation 17 evaluated at the right end of the interval
gives `x_next + B*increment`. The stored next-node state is `x_next`. Consequently
the **physical right-minus-left velocity change is `-increment`**. The result
reports these physical changes as `impulses`, at the right endpoint of each
interval, while retaining the paper's decision variables as `increments`.

At a mesh node, `state_at(t, side="right")` returns the stored node state;
`side="left"` returns the preceding interval's endpoint. Position is continuous
up to the solver feasibility tolerance; velocity can jump. There is no separate
impulse variable at the first node because its state is free.

The paper modifies the defect equation but does not fully rederive the midpoint
and interpolation formulas for discontinuous endpoints. This implementation uses
Equations 16–17 literally together with Equation 29. In particular, endpoint
dynamics in those formulas are evaluated from stored node states. **This discrete
transcription must not be represented as an exactly integrated sequence of physical
impulsive burns.** Mesh refinement and independent propagation with jumps are
needed before making physical reconstruction accuracy claims. A formulation with
separate pre-burn and post-burn endpoint states would be a distinct extension.

## Hybrid maneuvers: Equation 36

Hybrid mode includes both the continuous controls and impulsive increments. Its
objective is

```text
J = w*J_I + (1-w)*J_C,    0 < w < 1.
```

The weight compares nondimensional costs of different forms. It is a modeling
choice, not an inferred property of the object. Use continuous or impulsive mode
for the endpoint cases rather than leaving a set of controls unpenalized. The
paper proposes this hybrid extension but does not include a hybrid numerical
example.

## Measurements and uncertainty: Equations 11 and 37–38

Exact mode enforces zero measurement residual. Bounded mode enforces, separately
for each observation,

```text
residual.T @ inverse(covariance) @ residual <= chi2.ppf(confidence, q).
```

Two angular degrees of freedom are represented in a tangent plane at the measured
LOS. The basis points east and north relative to the rotating frame's z axis.
The residual consists of the two components of the predicted-minus-measured LOS
in that basis. A forward-hemisphere constraint excludes the antipodal LOS solution.
This avoids angular wrap discontinuities and the singular 3-by-3 covariance of a
unit direction with only two independent components.

When rates are supplied, the tangent components of predicted-minus-measured LOS
rate are appended. The covariance order is
`[east LOS, north LOS, east LOS_rate, north LOS_rate]`, and `q=4` instead of `q=2`.
Rates are derivatives with respect to nondimensional time, not seconds. A full
positive-definite covariance is required for bounded measurements with rates;
the solver does not invent a rate uncertainty. Observer velocity must also be
supplied, including explicitly supplying zeros for a stationary observer.

For angles-only core input, a scalar or per-observation `sigma` is the tangent
angular standard deviation. An explicitly supplied covariance takes precedence.
This tangent-plane representation is an implementation choice for the generic
measurement model in the paper. It approximates angular noise locally; it is not
an exact wrapped-angle likelihood for arbitrarily large errors. For RA/Dec
measurements, east uncertainty includes the `cos(dec)` factor; the stack adapter
transforms the measurement covariance into the rotating tangent basis.

The default confidence is 0.95. The paper does not state the confidence threshold
used for its collocation numerical examples (its explicitly mentioned 95% value
concerns the EKF comparison), so this default is not claimed to reproduce those
examples. All-zero `sigma` without a covariance selects exact measurements, even
if bounded mode was requested. Mixed zero/nonzero fallback sigmas are rejected.

## Bounds, initialization, and optimization

The paper uses generic admissible-region bounds but does not provide numerical
range, state, thrust, or impulse bounds for its examples. The package's range and
control bounds are exposed numerical/modeling settings. They are not published
spacecraft specifications or guarantees of a complete admissible region. Range
bounds are enforced at observations. A numerical exclusion radius of `1e-5 DU`
is enforced around each primary at mesh nodes and compressed midpoints. This
radius excludes singularities; it does not model an Earth or Moon surface.
Checking these locations does not prove continuous-time collision avoidance.

For each quadratic continuous-control segment, the package additionally bounds
its middle Bezier control point `2*u_mid - (u_left+u_right)/2`. Together with
the endpoint bounds, the convex-hull property guarantees that every component of
the interpolated control stays within its configured bound throughout the
interval. This is a conservative extension to the paper's displayed transcription:
some bounded quadratic curves have Bezier control points outside their own range
and are therefore excluded by this sufficient condition.

The default uses 80 mesh nodes and `initial_guess="auto"`. This solves from two
initializations, stationary L2 and LOS-based, with all controls initially zero.
The LOS guess places the target one DU along each measured sight line and
interpolates those positions to the mesh. Of the attempts that pass the optimizer,
feasibility, and mesh checks, the lowest control-objective solution is returned.
If both attempts fail, stationary L4 is tried as a fallback. Diagnostics retain
the attempts. No truth trajectory is read to initialize or select a result.

This small set of initializations reduces dependence on a single local basin;
it does not establish global optimality or identifiability. A minimum objective
can still correspond to an incorrect orbit when the observations are insufficient.
The strategy addresses sensitivity to initialization; it is not a reproduction
of the paper's experiments or a timing/performance claim. The paper's examples
use L4 for continuous thrust and L2 for impulsive cases. Explicit `"los"`,
`"L1"`, `"L2"`, or `"L4"`, or supplied initial mesh states, request a single solve.

[CasADi](https://web.casadi.org/docs/) constructs sparse derivatives of the full
constraints and objective, and [IPOPT](https://www.coin-or.org/Ipopt/) solves the
NLP. Automatic differentiation includes the dependence of the
midpoint state on the endpoint controls. This avoids treating the abbreviated
Appendix A derivative expressions as if those indirect dependencies were absent.

A successful solve means a local numerical solution passed the configured
feasibility and sampled mesh-resolution checks. It does not prove global
optimality, uniqueness, physical orbit accuracy, or statistical calibration.
Diagnostics distinguish optimizer status, objective, and constraint violation.
No state covariance is fabricated from a minimum-control solution.

### Mesh-resolution acceptance heuristic

The polynomial derivative is evaluated at fractions `1/4`, `1/2`, and `3/4` of
each interval and compared with `f(x_poly(t), u_poly(t))`. The acceptance
indicator is the maximum over intervals, sample points, and state components of

```text
h_k * abs(x_poly_dot - f(x_poly, u_poly)) / (1 + abs(x_poly)).
```

It must not exceed `mesh_tolerance` (default `1e-3`). The interval width converts
the derivative discrepancy into a rough local accumulated-error scale. This is
an implementation safeguard against grossly under-resolved curves; it is not an
algorithm or threshold supplied by the paper. Three samples can miss excursions,
and this indicator does not account for subsequent dynamical error amplification.
It is neither an SI position-error guarantee nor a certified residual bound.

In the literal impulsive transcription, the interior residual also reflects the
midpoint ambiguity described above. Its amplitude can be proportional to an
increment even when the algebraic defect vanishes; multiplying by interval width
allows that localized contribution to shrink as the burn interval is refined.
Independent propagation and a mesh convergence study remain required to establish
physical trajectory accuracy.

## Reproduction limits

The paper supplies its CR3BP constants and several starting orbit states, but not
complete machine-readable observation schedules, truth control histories, or
solver settings. The data availability statement directs requests for raw files
to the corresponding author. Its reported CPU times and trajectory errors are
therefore **not claimed as reproduced** by this package.

Additional ambiguities in the published examples include:

- The short-arc observer is called L2 northern halo in the text and L1 northern
  halo in its figure caption.
- The continuous example's stated number of free variables does not match a
  direct count of all six state components and three endpoint/midpoint controls
  for its stated 40 nodes.
- The short-arc noise description does not fully specify the angle/rate covariance
  convention. It lists 10 arcseconds and `1E-9 rad/TU` without a complete matrix.
- Sections 4.2 and 4.3 use angles **and angle rates**. The existing repository CSV
  contains angles only. It cannot validate those examples as written.

The included tests instead use independently generated trajectories and verify
the implemented transcription, measurement handling, frame conversions, solver
feasibility, and adapter behavior. Independent reconstruction experiments and
mesh convergence studies are still necessary for a scientific performance claim.

Finally, the existing downstream Orekit batch estimator fits an uncontrolled
CR3BP orbit. It does not consume the reconstructed control history. Passing a
single fitted state into that estimator is interface-compatible, but reproducing
a maneuvering arc requires downstream forces or maneuver models that actually
apply the returned controls and impulses.
