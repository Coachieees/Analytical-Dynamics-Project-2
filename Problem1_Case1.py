import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Global Font Settings ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['mathtext.it'] = 'Cambria:italic'
plt.rcParams['mathtext.bf'] = 'Cambria:bold'

# --- Parameters (Group 1 Setup) ---
g = 9.81
r = 1.0  # Constant Radius (m)
t_max = 20
dt = 0.01
t = np.arange(0, t_max, dt)

def equations(y, t, v0, g, r):
    theta, theta_dot = y
    if abs(theta - np.pi/2) < 1e-6:
        theta = np.pi/2 - 1e-6
    term_gravity = (g / r) * np.cos(theta)
    term_centrifugal = (v0**2 * np.sin(theta)) / (r**2 * np.cos(theta)**3)
    theta_ddot = term_gravity - term_centrifugal
    return [theta_dot, theta_ddot]

# Varying Initial Velocity
v0_cases = [1.0, 2.5, 5.0, 7.5, 10.0]
results_theta = {}
results_theta_dot = {}

for v0 in v0_cases:
    y0 = [0.0, 0.0] 
    sol = odeint(equations, y0, t, args=(v0, g, r))
    results_theta[v0] = sol[:, 0]
    results_theta_dot[v0] = sol[:, 1]

def get_coords(theta_arr, v0, r, dt):
    phi = 0
    x, y, z = [], [], []
    for th in theta_arr:
        phi_dot = v0 / (r * np.cos(th)**2)
        phi += phi_dot * dt
        x.append(r * np.cos(th) * np.cos(phi))
        y.append(r * np.cos(th) * np.sin(phi))
        z.append(-r * np.sin(th))
    return np.array(x), np.array(y), np.array(z)

# --- FIGURE 1: Trajectories (GridSpec Layout) ---
fig1 = plt.figure(figsize=(12, 10)) 

# Create a 2x2 grid where the top row is 1.5 times taller
gs = fig1.add_gridspec(2, 2, height_ratios=[1.8, 1])

# Top row, spanning both columns
ax3d = fig1.add_subplot(gs[0, :], projection='3d')

# Bottom row, left column
ax_top = fig1.add_subplot(gs[1, 0])

# Bottom row, right column
ax_side = fig1.add_subplot(gs[1, 1])

for v0 in v0_cases:
    xs, ys, zs = get_coords(results_theta[v0], v0, r, dt)
    lbl = f'v₀ = {v0} m/s'
    
    # Plot Trajectories
    ax3d.plot(xs, ys, zs, label=lbl)
    ax_top.plot(xs, ys, label=lbl)
    ax_side.plot(xs, zs, label=lbl)
    
    # Start and End points
    ax3d.scatter(xs[0], ys[0], zs[0], color='green', marker='o', s=50, zorder=10)
    ax3d.scatter(xs[-1], ys[-1], zs[-1], color='red', marker='x', s=50, zorder=10)

    ax_top.plot(xs[0], ys[0], 'go', markersize=8, zorder=10)
    ax_top.plot(xs[-1], ys[-1], 'rx', markersize=8, zorder=10)

    ax_side.plot(xs[0], zs[0], 'go', markersize=8, zorder=10)
    ax_side.plot(xs[-1], zs[-1], 'rx', markersize=8, zorder=10)

# Formatting Figure 1
ax3d.set_title(f"3D View (r = {r} m)", fontsize=14)
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")

ax_top.set_title(f"Top View (XY Plane, r = {r} m)", fontsize=14)
ax_top.set_xlabel("X (m)")
ax_top.set_ylabel("Y (m)")
ax_top.grid(True, linestyle=':', alpha=0.7)

ax_side.set_title(f"Side View (XZ Plane, r = {r} m)", fontsize=14)
ax_side.set_xlabel("X (m)")
ax_side.set_ylabel("Z (m)")
ax_side.grid(True, linestyle=':', alpha=0.7)

ax_top.axis('equal')
ax_side.axis('equal')

# Custom Legend for Start/End
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor='g', markersize=8),
                   Line2D([0], [0], marker='x', color='w', label='End', markeredgecolor='r', markersize=8)]
ax3d.legend(handles=legend_elements + ax3d.get_legend_handles_labels()[0], loc='upper left')

fig1.tight_layout(pad=3.0)

# --- FIGURE 2: Angular Dynamics ---
fig2, (ax_th, ax_thd) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for v0 in v0_cases:
    ax_th.plot(t, np.degrees(results_theta[v0]), label=f'v₀ = {v0} m/s')
    ax_thd.plot(t, results_theta_dot[v0], label=f'v₀ = {v0} m/s')

# Reference Line at theta = 90
ax_th.axhline(90, color='red', linestyle='--', linewidth=1.5, label='Bottom (90°)')

# Formatting Figure 2
ax_th.set_ylabel('Angle θ (degrees)', fontsize=12)
ax_th.set_title(f'Angular Position (θ) vs Time (r = {r} m)', fontsize=14)
ax_th.grid(True, linestyle=':', alpha=0.7)
ax_th.legend(loc='upper right')

ax_thd.set_ylabel('Velocity dθ/dt (rad/s)', fontsize=12)
ax_thd.set_xlabel('Time (s)', fontsize=12)
ax_thd.set_title(f'Angular Velocity (dθ/dt) vs Time (r = {r} m)', fontsize=14)
ax_thd.grid(True, linestyle=':', alpha=0.7)

fig2.tight_layout(pad=3.0)
plt.show()