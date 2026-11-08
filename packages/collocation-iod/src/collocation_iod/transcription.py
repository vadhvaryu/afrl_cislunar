"""Hermite-Simpson equations (15)-(19), usable with NumPy or CasADi values."""
from .models import cr3bp_rhs


def hermite_simpson(x0, x1, u0, um, u1, h, mu, rhs=cr3bp_rhs):
    f0, f1 = rhs(x0, mu, u0), rhs(x1, mu, u1)
    xm = (x0 + x1)/2 + h*(f0 - f1)/8
    fm = rhs(xm, mu, um)
    return x1 - x0 - h*(f0 + 4*fm + f1)/6, xm, f0, fm, f1


def state_polynomial(x0, f0, fm, f1, h, dt):
    return x0 + f0*dt + (-3*f0 + 4*fm - f1)*dt**2/(2*h) + (2*f0 - 4*fm + 2*f1)*dt**3/(3*h**2)


def control_polynomial(u0, um, u1, h, dt):
    s = dt / h
    return (2*s*s - 3*s + 1)*u0 + (4*s - 4*s*s)*um + (2*s*s - s)*u1
