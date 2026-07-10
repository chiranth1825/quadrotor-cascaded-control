"""
simulate.py
-----------
Ties dynamics + controller + trajectory together, runs the simulation loop,
and produces plots. Run this directly: `python src/simulate.py`

The simulation loop is the same pattern used everywhere in robotics control:
    for each timestep:
        1. get reference (x_des, z_des) from trajectory
        2. controller computes (T, tau) from current state + reference
        3. dynamics integrates forward one step using (T, tau)
        4. log everything for plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from dynamics import PlanarQuadrotor
from controller import CascadedController
from trajectory import step_trajectory, circle_trajectory, figure8_trajectory


def run_simulation(traj_fn, traj_name, t_final=15.0, dt=0.01):
    quad = PlanarQuadrotor(mass=1.0, inertia=0.02, gravity=9.81)
    quad.set_state(x=0.0, z=0.0, theta=0.0)
    ctrl = CascadedController(mass=quad.m, gravity=quad.g)

    n_steps = int(t_final / dt)
    log = {k: np.zeros(n_steps) for k in
           ['t', 'x', 'z', 'theta', 'x_des', 'z_des', 'theta_des', 'T', 'tau']}

    for i in range(n_steps):
        t = i * dt
        (x_des, z_des), (vx_des, vz_des), (ax_ff, az_ff) = traj_fn(t)

        T, tau, theta_des = ctrl.compute(quad.state, x_des, z_des, dt,
                                          vx_des=vx_des, vz_des=vz_des,
                                          ax_ff=ax_ff, az_ff=az_ff)
        quad.step((T, tau), dt)

        log['t'][i] = t
        log['x'][i], log['z'][i], log['theta'][i] = quad.state[0], quad.state[1], quad.state[2]
        log['x_des'][i], log['z_des'][i], log['theta_des'][i] = x_des, z_des, theta_des
        log['T'][i], log['tau'][i] = T, tau

    plot_results(log, traj_name)
    rmse_x = np.sqrt(np.mean((log['x'] - log['x_des']) ** 2))
    rmse_z = np.sqrt(np.mean((log['z'] - log['z_des']) ** 2))
    print(f"[{traj_name}] tracking RMSE -> x: {rmse_x:.4f} m, z: {rmse_z:.4f} m")
    return log


def plot_results(log, traj_name):
    os.makedirs('plots', exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Quadrotor Cascaded Control — {traj_name}', fontsize=14, fontweight='bold')

    # Path in x-z plane
    ax = axs[0, 0]
    ax.plot(log['x_des'], log['z_des'], 'r--', label='reference', linewidth=1.5)
    ax.plot(log['x'], log['z'], 'b-', label='actual', linewidth=1.5)
    ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')
    ax.set_title('Path tracking (x-z plane)')
    ax.legend(); ax.axis('equal'); ax.grid(alpha=0.3)

    # x and z vs time
    ax = axs[0, 1]
    ax.plot(log['t'], log['x_des'], 'r--', label='x_des')
    ax.plot(log['t'], log['x'], 'b-', label='x')
    ax.plot(log['t'], log['z_des'], 'g--', label='z_des')
    ax.plot(log['t'], log['z'], 'm-', label='z')
    ax.set_xlabel('t (s)'); ax.set_ylabel('position (m)')
    ax.set_title('Position tracking vs time')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # theta vs theta_des (inner loop tracking)
    ax = axs[1, 0]
    ax.plot(log['t'], np.degrees(log['theta_des']), 'r--', label='theta_des')
    ax.plot(log['t'], np.degrees(log['theta']), 'b-', label='theta')
    ax.set_xlabel('t (s)'); ax.set_ylabel('pitch (deg)')
    ax.set_title('Inner loop: attitude tracking')
    ax.legend(); ax.grid(alpha=0.3)

    # control inputs
    ax = axs[1, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(log['t'], log['T'], 'g-', label='Thrust T (N)')
    l2, = ax2.plot(log['t'], log['tau'], 'orange', label='Torque tau (N.m)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('T (N)', color='g'); ax2.set_ylabel('tau (N.m)', color='orange')
    ax.set_title('Control effort')
    ax.legend(handles=[l1, l2], loc='upper right', fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = f'plots/{traj_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"Saved {fname}")


if __name__ == '__main__':
    run_simulation(lambda t: step_trajectory(t, x1=2.0, z1=2.5), 'Step Response')
    run_simulation(lambda t: circle_trajectory(t), 'Circle Trajectory', t_final=15.0)
    run_simulation(lambda t: figure8_trajectory(t), 'Figure8 Trajectory', t_final=20.0)
