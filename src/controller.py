"""
controller.py
-------------
Cascaded control architecture: an OUTER position loop and an INNER attitude
loop, running at the same rate here for simplicity (in real flight
controllers the inner loop typically runs 5-10x faster, e.g. 500Hz attitude
vs 50-100Hz position, because attitude dynamics are faster and more safety
critical - if attitude diverges the vehicle literally falls out of the sky,
whereas position error just means you're a few cm off course).

WHY CASCADED AND NOT A SINGLE MIMO CONTROLLER?
Because of the physics in dynamics.py: you cannot directly command x
acceleration. You can only command thrust magnitude and torque. The vehicle
must tilt (change theta) to redirect thrust sideways. So position control
is fundamentally a two-step process:
    1. "I want x-acceleration A" -> outer loop computes "tilt to theta_des"
    2. "I am at theta, want theta_des" -> inner loop computes torque

This mirrors exactly what you did in the AAS PID/PD assignments and the
cascaded control architecture assignment - this file is that architecture
made explicit and runnable end-to-end.

OUTER LOOP (position -> desired attitude + thrust):
    z-axis (altitude) is control DIRECTLY through thrust:
        T = m * (g + PD_z(z_des - z))
        (g feedforward cancels gravity at hover - without it the PD term
         alone would have to fight gravity constantly, causing large
         steady-state error unless ki is added)

    x-axis (lateral) is controlled INDIRECTLY through desired pitch:
        theta_des = -PD_x(x_des - x) / g
        (small-angle approx: xdd ≈ -g * theta for small theta, so solving
         for theta that would produce the desired xdd gives this. We also
         saturate theta_des - a real vehicle cannot pitch 90 degrees, and
         our small-angle approximation breaks down at large angles anyway)

INNER LOOP (attitude -> torque):
        tau = PD_theta(theta_des - theta)
"""

import numpy as np


class PDController:
    """Simple PD controller with output saturation."""
    def __init__(self, kp, kd, ki=0.0, out_limits=(-np.inf, np.inf)):
        self.kp, self.kd, self.ki = kp, kd, ki
        self.out_min, self.out_max = out_limits
        self.integral = 0.0

    def compute(self, error, error_dot, dt):
        self.integral += error * dt
        out = self.kp * error + self.kd * error_dot + self.ki * self.integral
        return np.clip(out, self.out_min, self.out_max)


class CascadedController:
    def __init__(self, mass, gravity=9.81,
                 kp_z=8.0, kd_z=4.5,
                 kp_x=9.0, kd_x=4.2,
                 kp_theta=6.0, kd_theta=0.5,
                 theta_max=np.radians(25)):
        self.m = mass
        self.g = gravity
        self.theta_max = theta_max

        # Outer loop gains.
        # NOTE: pd_x's output is a desired ACCELERATION (m/s^2), not an
        # angle - it gets converted to theta_des and clipped to theta_max
        # further down. Clipping it here to (-theta_max, theta_max) would
        # be a units bug (radians vs m/s^2), so we leave it unclipped and
        # bound accel to something physically sane instead.
        accel_limit = theta_max * gravity  # max accel achievable at theta_max
        self.pd_z = PDController(kp_z, kd_z)
        self.pd_x = PDController(kp_x, kd_x, out_limits=(-accel_limit, accel_limit))

        # Inner loop gains - deliberately much higher bandwidth (larger kp)
        # than the outer loop, per the cascaded-control design rule: each
        # inner loop must be significantly faster than the loop that
        # commands it, or the cascade becomes unstable / sluggish.
        self.pd_theta = PDController(kp_theta, kd_theta)

    def compute(self, state, x_des, z_des, dt,
                vx_des=0.0, vz_des=0.0, ax_ff=0.0, az_ff=0.0):
        """
        vx_des, vz_des: feedforward reference velocities (m/s).
        ax_ff, az_ff: feedforward reference accelerations (m/s^2).

        Both matter, for different reasons:
        - ax_ff/az_ff cancel the reference's own acceleration so feedback
          only has to correct *deviations* from the trajectory, not
          generate the whole motion from scratch.
        - vx_des/vz_des fix a subtler issue: our derivative term is
          "derivative on measurement" (-actual_velocity) rather than
          "derivative on error" (ref_velocity - actual_velocity), which is
          the standard trick to avoid derivative kick on setpoint changes.
          But for a *moving* reference this leaves a residual term
          proportional to the reference's own velocity uncancelled in the
          closed-loop error dynamics - you can derive this directly from
          the error ODE. Passing vx_des/vz_des here restores true
          derivative-on-error and removes that residual. For static
          setpoints both default to 0, which correctly reduces to plain
          derivative-on-measurement PD.
        """
        x, z, theta, vx, vz, omega = state

        # ---- Outer loop: position -> desired thrust & desired pitch ----
        z_err = z_des - z
        # feedback correction + feedforward of the reference's own accel
        T_vertical_needed = self.m * (self.g + az_ff + self.pd_z.compute(z_err, vz_des - vz, dt))
        # Thrust-tilt coupling compensation: only T*cos(theta) of total
        # thrust acts vertically. Without dividing by cos(theta) here, any
        # lateral maneuver (which requires tilting) "steals" lift and the
        # vehicle sags in altitude - a very real, very common coupling
        # effect on real quadrotors, not a numerical artifact. Dividing by
        # cos(theta) decouples the two loops so each behaves as if the
        # other weren't there (valid as long as theta stays well away from
        # +/-90 deg, which theta_max already guarantees).
        T = T_vertical_needed / max(np.cos(theta), 0.1)
        T = max(T, 0.0)  # thrust can't be negative

        x_err = x_des - x
        # feedback correction + feedforward of the reference's own accel
        xdd_des = ax_ff + self.pd_x.compute(x_err, vx_des - vx, dt)
        theta_des = np.clip(-xdd_des / self.g, -self.theta_max, self.theta_max)

        # ---- Inner loop: attitude -> torque ----
        theta_err = theta_des - theta
        tau = self.pd_theta.compute(theta_err, -omega, dt)

        return T, tau, theta_des
