"""
trajectory.py
-------------
Reference trajectory generators. The controller only ever sees
(x_des, z_des) at each timestep - it has no idea whether that came from
a step, a circle, or a figure-8. This separation (trajectory generation
vs. tracking control) is standard in robotics: swap the trajectory without
touching the controller at all.
"""

import numpy as np

"""
Every trajectory function now returns (pos, vel, accel) instead of just
pos. This is what makes feedforward control possible: the controller can
add the reference's own acceleration directly into the control law instead
of relying purely on feedback (PD) to "discover" that acceleration after
the fact, which is what caused the lag you'd see with feedback-only control
on any moving (non-step) target. This pattern - closed-form derivatives of
a trajectory feeding a feedforward term - is standard in trajectory-tracking
control and is closely related to "differential flatness", a concept you'll
see a lot in quadrotor trajectory generation literature (Mellinger & Kumar).
"""


def step_trajectory(t, x0=0.0, z0=0.0, x1=2.0, z1=2.0, t_step=1.0):
    """Simple step: hold (x0,z0) then jump to (x1,z1) at t_step.
    Velocity/accel are zero everywhere except the (physically unrealizable)
    instant of the jump, so feedforward = 0 here and it's a pure feedback
    problem - which is exactly why this is a good baseline case."""
    if t < t_step:
        return (x0, z0), (0.0, 0.0), (0.0, 0.0)
    return (x1, z1), (0.0, 0.0), (0.0, 0.0)


def circle_trajectory(t, radius=1.5, center=(0.0, 2.0), omega=0.5):
    """Circular trajectory in the x-z plane, with analytic vel/accel."""
    cx, cz = center
    pos = (cx + radius * np.cos(omega * t), cz + radius * np.sin(omega * t))
    vel = (-radius * omega * np.sin(omega * t), radius * omega * np.cos(omega * t))
    accel = (-radius * omega**2 * np.cos(omega * t), -radius * omega**2 * np.sin(omega * t))
    return pos, vel, accel


def figure8_trajectory(t, a=1.5, center=(0.0, 2.0), omega=0.4):
    """Figure-8 (lemniscate) trajectory - a good stress test since it
    requires continuous sign changes in both position and pitch.
    x(t) = a*sin(wt), z(t) = a*sin(wt)*cos(wt) = (a/2)*sin(2wt)"""
    cx, cz = center
    s, c = np.sin(omega * t), np.cos(omega * t)
    x = cx + a * s
    z = cz + (a / 2) * np.sin(2 * omega * t)
    vx = a * omega * c
    vz = a * omega * np.cos(2 * omega * t)
    ax = -a * omega**2 * s
    az = -2 * a * omega**2 * np.sin(2 * omega * t)
    return (x, z), (vx, vz), (ax, az)
