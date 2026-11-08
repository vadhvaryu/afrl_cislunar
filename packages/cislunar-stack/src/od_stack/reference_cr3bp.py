import numpy as np


def nondim_cr3bp(t, Y, pi_2):
    """CR3BP equations of motion in nondimensional rotating coordinates.

    Y = [x, y, z, vx, vy, vz]. Use with solve_ivp(..., args=(pi_2,)).
    """
    x, y, z = Y[:3]
    xdot, ydot, zdot = Y[3:]

    Ydot = np.zeros_like(Y)
    Ydot[:3] = Y[3:]                                   # d(position)/dt = velocity

    sigma = np.sqrt((x + pi_2) ** 2 + y**2 + z**2)     # distance to Earth
    psi = np.sqrt((x - 1 + pi_2) ** 2 + y**2 + z**2)   # distance to Moon
    Ydot[3] = 2 * ydot + x - (1 - pi_2) * (x + pi_2) / sigma**3 - pi_2 * (x - 1 + pi_2) / psi**3
    Ydot[4] = -2 * xdot + y - (1 - pi_2) * y / sigma**3 - pi_2 * y / psi**3
    Ydot[5] = -(1 - pi_2) * z / sigma**3 - pi_2 * z / psi**3
    return Ydot


def jacobi(r, v, pi_2):
    """Jacobi-type energy along a trajectory. r, v are (N, 3) arrays. Should stay constant."""
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    sigma = np.sqrt((x + pi_2) ** 2 + y**2 + z**2)
    psi = np.sqrt((x - 1 + pi_2) ** 2 + y**2 + z**2)
    return (0.5 * np.sum(v**2, axis=1) - (1 - pi_2) / sigma - pi_2 / psi
            - 0.5 * (x**2 + y**2))


# ---------------------------------------------------------------------------
# State transition matrix (STM)
#
# The STM Phi(t) is a 6x6 matrix: if the starting state changes by a tiny
# amount dY0, the state at time t changes by about Phi(t) @ dY0.
# It starts as the identity and evolves as dPhi/dt = A(t) @ Phi, where A is
# the Jacobian of the equations of motion along the trajectory.
# ---------------------------------------------------------------------------

def cr3bp_jacobian(Y, pi_2):
    """6x6 Jacobian A = d(Ydot)/dY of the CR3BP equations at state Y."""
    x, y, z = Y[:3]
    m1, m2 = 1 - pi_2, pi_2
    dx1, dx2 = x + pi_2, x - 1 + pi_2                  # x-offsets from Earth and Moon
    r1 = np.sqrt(dx1**2 + y**2 + z**2)                 # distance to Earth (sigma)
    r2 = np.sqrt(dx2**2 + y**2 + z**2)                 # distance to Moon (psi)
    a1, b1 = m1 / r1**3, 3 * m1 / r1**5
    a2, b2 = m2 / r2**3, 3 * m2 / r2**5

    # Second derivatives of the effective potential
    Uxx = 1 - a1 - a2 + b1 * dx1**2 + b2 * dx2**2
    Uyy = 1 - a1 - a2 + b1 * y**2 + b2 * y**2
    Uzz = -a1 - a2 + b1 * z**2 + b2 * z**2
    Uxy = b1 * dx1 * y + b2 * dx2 * y
    Uxz = b1 * dx1 * z + b2 * dx2 * z
    Uyz = b1 * y * z + b2 * y * z

    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)                              # d(position rate)/d(velocity)
    A[3:, :3] = [[Uxx, Uxy, Uxz],
                 [Uxy, Uyy, Uyz],
                 [Uxz, Uyz, Uzz]]
    A[3, 4], A[4, 3] = 2, -2                           # Coriolis terms
    return A


def nondim_cr3bp_stm(t, Y, pi_2):
    """CR3BP equations plus the STM. Y has 42 entries: the 6 states, then the
    36 STM entries (row by row). Use with solve_ivp(..., args=(pi_2,)).
    Start from: np.hstack((Y_0, np.eye(6).ravel()))
    """
    Phi = Y[6:].reshape(6, 6)
    Ydot = np.empty(42)
    Ydot[:6] = nondim_cr3bp(t, Y[:6], pi_2)
    Ydot[6:] = (cr3bp_jacobian(Y[:6], pi_2) @ Phi).ravel()
    return Ydot
