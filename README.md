# Quadrotor Cascaded Control (Planar Model)

A from-scratch simulation of a planar (2D) quadrotor tracking position
trajectories using **cascaded PD control** — an outer position loop and an
inner attitude loop — with **trajectory feedforward** to eliminate lag on
moving references. Written in pure Python/NumPy (RK4 integration), no
control-toolbox dependency, so every line of the control law is visible.

## Why cascaded control?

A quadrotor cannot directly accelerate sideways — thrust only acts along
its own body axis. To move in `x`, it must first **tilt** (change pitch
`theta`) to redirect part of its thrust sideways, then use that tilt to
accelerate. This physical fact forces a two-loop structure:

```
 x_des, z_des ──▶ [Outer loop: position PD] ──▶ theta_des, T
                                                     │
     theta_des ──▶ [Inner loop: attitude PD] ──▶ tau (torque)
```

The outer loop treats "desired tilt angle" as its control output; the inner
loop treats that tilt angle as its setpoint and drives the actual torque.
This mirrors real flight-controller firmware (e.g. Betaflight/PX4 rate →
angle → position cascades), just with 2 axes instead of 4.

## What's in this repo

| File | What it does |
|---|---|
| `src/dynamics.py` | Nonlinear planar rigid-body equations of motion + RK4 integrator |
| `src/controller.py` | Cascaded PD controller with feedforward and thrust-tilt decoupling |
| `src/trajectory.py` | Reference trajectories (step, circle, figure-8) with analytic velocity/acceleration |
| `src/simulate.py` | Runs the sim, produces the plots in `plots/` |

Run it:
```bash
pip install -r requirements.txt
cd src && python simulate.py
```

## Key Engineering Decisions

**1. Why RK4 instead of Euler integration?**
Euler integration accumulates visible error on oscillatory/rotational
systems like this one over hundreds of timesteps. RK4 gets 4th-order
accuracy for ~4x the compute of Euler — standard for physics simulation.

**2. The inner loop must be much faster (and correctly damped) than the
outer loop, or the cascade doesn't work.**
Early version used `kp_theta=40, kd_theta=8` with `I=0.02` — this gives
an effective proportional gain of 2000 rad/s² per radian of error, which
is numerically unstable at `dt=0.01`. Tuning to a physically sane natural
frequency and damping ratio (`ζ≈0.7`, critically damped) fixed it. I
initially over-corrected to `kd_theta=1.5` (`ζ≈2.2`, heavily overdamped) —
stable, but *sluggish enough that it silently defeated the feedforward
term below*. Getting the inner loop both fast *and* correctly damped
mattered more than either alone.

**3. Thrust-tilt coupling: tilting "steals" lift.**
Only `T·cos(θ)` of total thrust acts vertically — the rest goes sideways
(that's the whole point). So any lateral maneuver naturally saps altitude
unless compensated. This isn't a simulation artifact; it's the same
coupling a real quadrotor experiences. Fixed with `T = T_needed / cos(θ)`.

**4. Feedback-only PD tracks a step to ~zero error, but always lags a
*moving* target.**
This is a clean, derivable control-theory result: for a plain PD position
loop with no reference feedforward, the closed-loop error transfer
function has non-zero DC gain to the reference's own velocity/acceleration.
You can compute the exact steady-state tracking error from the closed-loop
transfer function `E(s)/Zdes(s) = (s² + kd·s)/(s² + kd·s + kp)`, evaluated
at the trajectory's frequency — and it matched the measured RMSE almost
exactly during development (~0.42 m predicted vs ~0.42 m measured on the
uncompensated circle trajectory). The fix is standard: feed the reference's
own velocity and acceleration into the control law directly
(**feedforward**), so feedback only has to correct *deviations*, not
generate the whole motion from scratch. This is closely related to
*differential flatness*, a concept used a lot in quadrotor trajectory
generation (Mellinger & Kumar).

**5. A subtler feedforward bug: derivative-on-measurement vs
derivative-on-error.**
The derivative term was implemented as `-actual_velocity` (a common trick
to avoid "derivative kick" on setpoint changes). For a *moving* reference
this leaves a residual term proportional to the reference's own velocity
uncancelled in the closed-loop error ODE — adding acceleration feedforward
alone wasn't enough to fully remove the lag. Passing the reference velocity
through and using `(v_des - v_actual)` as the derivative term (true
derivative-on-error) closed the gap. Figure-8 x-axis RMSE went from
0.41 m → 0.01 m after this fix.

## Results (RMSE tracking error, post-transient)

| Trajectory | x RMSE | z RMSE | Notes |
|---|---|---|---|
| Step (0,0)→(2, 2.5) | 0.36 m | 0.40 m | Dominated by physical approach time, not a tuning issue |
| Circle (r=1.5m) | 0.26 m | 0.34 m | Residual from x-z coupling through tilt angle |
| Figure-8 | **0.01 m** | 0.29 m | x-axis near-perfect with full feedforward; z lag is coupling from x maneuvering |

## Known limitation / possible extension

Isolated altitude tracking (no lateral maneuvering) achieves ~0.005 m RMSE
— essentially perfect. The residual z error during combined x-z maneuvers
comes from real coupling through the tilt angle that the current
`cos(theta)` compensation doesn't fully cancel (it uses the *current*
theta, which itself lags theta_des slightly). A next step would be full
nonlinear feedback linearization of the coupled x-z dynamics, or increasing
altitude-loop bandwidth further — both are natural "future work" talking
points.
