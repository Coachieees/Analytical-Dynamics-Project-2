import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Define System Parameters ---
m = 1.0  # mass (kg)
k = 10000.0 # spring stiffness (N/m)

# --- 2. Define the Differential Equations ---
def equations(t, state):
    x, x_dot, y, y_dot = state
    
    # Derived equations of motion
    x_ddot = -(k / (3 * m)) * (x - y)
    y_ddot = (k / m) * (x - y)
    
    return [x_dot, x_ddot, y_dot, y_ddot]

# --- 3. Initial Conditions ---
x0 = 0
x_dot0 = 0
y0 = -0.1
y_dot0 = 0
initial_state = [x0, x_dot0, y0, y_dot0]

# Time span for simulation (0 to 5 seconds)
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# --- 4. Solve the ODE ---
solution = solve_ivp(equations, t_span, initial_state, t_eval=t_eval)

t = solution.t
x = solution.y[0]
x_dot = solution.y[1]
y = solution.y[2]
y_dot = solution.y[3]

# --- 5. Calculate Required Plot Quantities ---
# Center of Mass (CM) Displacement calculation:
# Mass A moves down by x (z = -x). Mass B moves up by x (z = +x). Mass C moves up by y (z = +y).
# z_cm = [2m(-x) + m(x) + m(y)] / (2m + m + m) = (-2mx + mx + my) / 4m = (y - x) / 4
cm_position = (y - x) / 4.0

kinetic_energy = 1.5 * m * x_dot**2 + 0.5 * m * y_dot**2
potential_energy = 0.5 * k * (x - y)**2
total_energy = kinetic_energy + potential_energy
lagrangian = kinetic_energy - potential_energy

# --- 6. Generate Plots ---
fig, axs = plt.subplots(5, 1, figsize=(8, 9), sharex=True)

# Plot x vs time
axs[0].plot(t, x, 'b', label='$x$ (Mass B)')
axs[0].set_ylabel('Position (m)')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Plot y vs time
axs[1].plot(t, y, 'r', label='$y$ (Mass C)')
axs[1].set_ylabel('Position (m)')
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Plot CM position vs time
axs[2].plot(t, cm_position, 'g', label='Center of Mass $\\Delta z$')
axs[2].set_ylabel('CM Position (m)')
axs[2].legend(loc='upper right')
axs[2].grid(True)

# Plot Total Energy vs time
axs[3].plot(t, total_energy, 'm', label='Total Energy ($E$)')
axs[3].set_ylabel('Energy (J)')
# Set y-axis limits slightly above and below the constant energy value to show it's flat
axs[3].set_ylim(0, max(total_energy) * 1.5) 
axs[3].legend(loc='upper right')
axs[3].grid(True)

# Plot Lagrangian vs time
axs[4].plot(t, lagrangian, 'c', label='Lagrangian ($L = T - V$)')
axs[4].set_xlabel('Time (s)')
axs[4].set_ylabel('Lagrangian (J)')
axs[4].legend(loc='upper right')
axs[4].grid(True)

plt.suptitle('Dynamics of the Pulley-Mass-Spring System\nCASE-3: x(0)=0  |  x_dot(0)=0  |  y(0)=-0.1  |  y_dot(0)=0 ', fontsize=14)
plt.tight_layout()
plt.show()