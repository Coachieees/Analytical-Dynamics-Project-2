import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- [Existing Physics Logic Kept for Context] ---
g = 9.81
r = 1.0
t_max = 10
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

v0_cases = [2]
results = {}

# Solve the ODEs
for v0 in v0_cases:
    y0 = [0.0, 0.0] 
    sol = odeint(equations, y0, t, args=(v0, g, r))
    results[v0] = sol[:, 0]

def get_trajectory(theta_arr, v0, r, dt):
    phi = 0
    x_vals, y_vals, z_vals = [], [], []
    for theta in theta_arr:
        phi_dot = v0 / (r * np.cos(theta)**2)
        phi += phi_dot * dt
        x_vals.append(r * np.cos(theta) * np.cos(phi))
        y_vals.append(r * np.cos(theta) * np.sin(phi))
        z_vals.append(-r * np.sin(theta))
    return np.array(x_vals), np.array(y_vals), np.array(z_vals)

# --- NEW VISUALIZATION SECTION ---

fig = plt.figure(figsize=(15, 5))

# 1. 3D Trajectory (Existing)
ax3d = fig.add_subplot(131, projection='3d')
# 2. XY Plane - Top View (New)
ax_xy = fig.add_subplot(132)
# 3. XZ Plane - Side View (New)
ax_xz = fig.add_subplot(133)

# Pre-calculate bowl wireframe for the projections
u_bowl = np.linspace(0, 2 * np.pi, 100)
v_bowl = np.linspace(0, np.pi / 2, 50)

for v0 in v0_cases:
    theta_data = results[v0]
    xs, ys, zs = get_trajectory(theta_data, v0, r, dt)
    
    label = f'$v_0={v0} m/s$'
    ax3d.plot(xs, ys, zs, label=label)
    ax_xy.plot(xs, ys, label=label)
    ax_xz.plot(xs, zs, label=label)

# --- Refine 3D Plot ---
ax3d.set_title("3D Trajectory")
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")

# --- Refine XY Plot (Top View) ---
# Draw the rim of the bowl
rim_x = r * np.cos(u_bowl)
rim_y = r * np.sin(u_bowl)
ax_xy.plot(rim_x, rim_y, color='gray', linestyle='--', alpha=0.5)
ax_xy.set_title("Top View (XY Plane)")
ax_xy.set_xlabel("X")
ax_xy.set_ylabel("Y")
ax_xy.axis('equal')
ax_xy.grid(True)

# --- Refine XZ Plot (Side View) ---
# Draw the bowl profile (cross-section)
profile_v = np.linspace(-np.pi/2, np.pi/2, 100)
profile_x = r * np.cos(profile_v)
profile_z = -r * np.abs(np.sin(profile_v))
ax_xz.plot(profile_x, profile_z, color='gray', linestyle='--', alpha=0.5)
ax_xz.set_title("Side View (XZ Plane)")
ax_xz.set_xlabel("X")
ax_xz.set_ylabel("Z")
ax_xz.axis('equal')
ax_xz.grid(True)

plt.tight_layout()
ax3d.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
plt.show()