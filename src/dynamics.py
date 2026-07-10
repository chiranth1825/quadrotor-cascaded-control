"""
dynamics.py
-----------
Planar (2D, x-z plane) quadrotor rigid-body dynamics.

Why planar and not full 3D?
The full 3D quadrotor (6DOF: x,y,z,roll,pitch,yaw) decouples almost perfectly
into two independent planar problems (x-z plane controlled by pitch, y-z plane
controlled by roll) plus a yaw channel. Building the planar model first is the
standard way to *understand* cascaded control before generalizing to 3D -
the physics and the control architecture is identical, just with more axes.

STATE VECTOR: [x, z, theta, vx, vz, omega]
    x, z    - position in the vertical plane (m)
    theta   - pitch angle, positive = nose up in our convention (rad)
    vx, vz  - linear velocities (m/s)
    omega   - angular velocity (rad/s)

INPUTS: [T, tau]
    T   - total thrust from both rotors, acts along the body z-axis (N)
    tau - net torque about the body y-axis, from differential thrust (N*m)

EQUATIONS OF MOTION (Newton-Euler, standard quadrotor derivation):
    m * xdd = -T * sin(theta)
    m * zdd =  T * cos(theta) - m * g
    I * thetadd = tau

Notice the coupling: to accelerate in x, the vehicle must PITCH first
(tilt the thrust vector). This is *why* cascaded control is the natural
architecture here - position control necessarily happens through attitude.
"""

import numpy as np


class PlanarQuadrotor:
    def __init__(self, mass=1.0, inertia=0.02, gravity=9.81, arm_length=0.2):
        self.m = mass          # kg
        self.I = inertia       # kg*m^2, moment of inertia about pitch axis
        self.g = gravity       # m/s^2
        self.L = arm_length    # m, distance from center to each rotor

        # state: [x, z, theta, vx, vz, omega]
        self.state = np.zeros(6)

    def set_state(self, x=0, z=0, theta=0, vx=0, vz=0, omega=0):
        self.state = np.array([x, z, theta, vx, vz, omega], dtype=float)

    def dynamics(self, state, u):
        """
        Continuous-time equations of motion.
        state: [x, z, theta, vx, vz, omega]
        u: [T, tau]  (thrust, torque)
        returns state_dot
        """
        x, z, theta, vx, vz, omega = state
        T, tau = u

        xdd = -(T / self.m) * np.sin(theta)
        zdd =  (T / self.m) * np.cos(theta) - self.g
        thetadd = tau / self.I

        return np.array([vx, vz, omega, xdd, zdd, thetadd])

    def step(self, u, dt):
        """
        Advance the simulation by dt seconds using RK4 integration.

        Why RK4 and not simple Euler? Euler integration accumulates
        significant error for oscillatory/rotational systems like this one
        over hundreds of timesteps - the pitch dynamics in particular will
        visibly drift with Euler at dt=0.01s. RK4 is the standard choice
        for physics simulation because it gets 4th-order accuracy for
        roughly 4x the compute of Euler - a very good trade.
        """
        s = self.state
        k1 = self.dynamics(s, u)
        k2 = self.dynamics(s + 0.5 * dt * k1, u)
        k3 = self.dynamics(s + 0.5 * dt * k2, u)
        k4 = self.dynamics(s + dt * k3, u)

        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.state

    def rotor_thrusts_from_T_tau(self, T, tau):
        """
        Convert total thrust + torque into individual rotor thrusts (T1, T2).
        This is the "mixer" / allocation step present on every real flight
        controller (translates high-level commands into motor PWM signals).

        T = T1 + T2
        tau = (T2 - T1) * L   (T2 on the right produces positive/nose-up torque)
        """
        T1 = (T - tau / self.L) / 2.0
        T2 = (T + tau / self.L) / 2.0
        return T1, T2
