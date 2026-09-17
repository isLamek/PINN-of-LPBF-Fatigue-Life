"""
Threshold-corrected fatigue life integral.

POSED PROBLEM
-------------
Murakami stress intensity factor range for a crack of depth a:

    (1)   dK(a) = Y * ds * sqrt(pi * a)

Paris crack growth law WITH the threshold retained:

    (2)   da/dN = C * (dK - dK_th)^m ,   valid only where dK > dK_th

Cycles to grow from the initial flaw a0 to the collapse depth a_f:

    (3)   N = integral from a0 to a_f of  da / [ C * (dK(a) - dK_th)^m ]

Equation (3) has no closed form for dK_th > 0, so it is evaluated numerically.
Setting dK_th = 0 recovers the closed form, which is used as the verification.

    (4)   N0 = [ a_f^(1-m/2) - a0^(1-m/2) ] / [ (1 - m/2) * C * (Y*ds*sqrt(pi))^m ]

SYMBOLS
    a       crack depth                                mm
    a0      initial flaw size, taken as sqrt(area)     mm
    a_f     crack depth at net-section collapse        mm
    dK      stress intensity factor range              MPa*sqrt(m)
    dK_th   threshold stress intensity factor range    MPa*sqrt(m)
    Y       geometry factor                            -
    ds      stress range                               MPa
    C       Paris coefficient                          mm/cycle
    m       Paris exponent                             -
    N       cycles to failure                          cycles

NOTE ON UNITS. dK is formed with a in METRES, because MPa*sqrt(m) is the
conventional unit, while the integration variable and C are in MILLIMETRES. The
factor of 1e-3 inside the square root is the only place the two meet.

A specimen whose initial flaw sits below threshold cannot grow a crack at all, so
equation (3) returns infinity. That is a physical prediction, not a failure of the
method: the test would have been stopped at the run-out limit.
"""
import numpy as np

Y_SURFACE = 0.65
C_PARIS   = 6.25e-10          # mm/cycle
M_PARIS   = 3.94
DKTH      = 5.0               # MPa*sqrt(m)
RUNOUT    = 1.0e7             # cycles, the ORNL removal point


def dK(a_mm, ds_mpa, Y=Y_SURFACE):
    """Stress intensity factor range at crack depth a, in MPa*sqrt(m)."""
    return Y * ds_mpa * np.sqrt(np.pi * np.asarray(a_mm) * 1e-3)


def life_numeric(a0_mm, af_mm, ds_mpa, dkth=DKTH, Y=Y_SURFACE,
                 C=C_PARIS, m=M_PARIS, n=20001):
    """
    Equation (3) by Simpson's rule on a logarithmic grid in a.

    A log grid is used because the integrand is dominated by the lower limit when
    m > 2: on a linear grid almost every sample would sit where the integrand is
    negligible. Returns np.inf when the initial flaw is at or below threshold.
    """
    a0_mm = float(a0_mm); af_mm = float(af_mm)
    if af_mm <= a0_mm:
        return 0.0
    if dK(a0_mm, ds_mpa, Y) <= dkth:
        return np.inf                       # cannot grow: below threshold
    if n % 2 == 0:
        n += 1                              # Simpson needs an odd count
    a = np.logspace(np.log10(a0_mm), np.log10(af_mm), n)
    f = 1.0 / (C * (dK(a, ds_mpa, Y) - dkth) ** m)
    # Simpson on a non-uniform grid: integrate over the substitution u = ln a,
    # which is uniform, with da = a du.
    u = np.log(a)
    h = u[1] - u[0]
    g = f * a
    return float(h / 3.0 * (g[0] + g[-1] + 4 * g[1:-1:2].sum() + 2 * g[2:-2:2].sum()))


def life_closed_form(a0_mm, af_mm, ds_mpa, Y=Y_SURFACE, C=C_PARIS, m=M_PARIS):
    """Equation (4). Threshold-free, exact. Used to verify the numerical integral."""
    k = C * (Y * ds_mpa * np.sqrt(np.pi * 1e-3)) ** m
    p = 1.0 - m / 2.0
    return float((af_mm ** p - a0_mm ** p) / (p * k))


def verify(report=True):
    """
    VERIFICATION. With dK_th = 0 the numerical integral must reproduce the closed
    form. Refining the grid must drive the relative error down at the expected
    rate for Simpson's rule.
    """
    a0, af, ds = 0.4, 2.0, 405.0
    exact = life_closed_form(a0, af, ds)
    out = []
    for n in (101, 201, 401, 801, 1601, 3201):
        approx = life_numeric(a0, af, ds, dkth=0.0, n=n)
        out.append(dict(n=n, value=approx, rel_err=abs(approx - exact) / exact))
    for i in range(1, len(out)):
        if out[i]["rel_err"] > 0:
            out[i]["rate"] = np.log2(out[i - 1]["rel_err"] / out[i]["rel_err"])
    if report:
        print("VERIFICATION 3  numerical life integral against the closed form")
        print(f"  closed form, dK_th = 0:  {exact:,.0f} cycles".replace(",", " "))
        print(f"  {'intervals':>10} {'numerical':>16} {'rel. error':>12} {'rate':>7}")
        for d in out:
            print(f"  {d['n']:10d} {d['value']:16,.0f} {d['rel_err']:12.2e} "
                  f"{d.get('rate', float('nan')):7.2f}".replace(",", " "))
    return dict(exact=exact, table=out)


def physical_life(a0_um, af_mm, ds_mpa, dkth=DKTH, cap=RUNOUT):
    """
    Physics baseline used as a model feature: cycles predicted by equations (1)
    to (3) alone, with no fitting. Capped at the run-out limit, because a
    sub-threshold specimen would have been removed at 10^7 cycles rather than
    recorded as infinite.
    """
    n = life_numeric(a0_um * 1e-3, af_mm, ds_mpa, dkth=dkth)
    return min(n, cap), np.isinf(n)


if __name__ == "__main__":
    verify()
