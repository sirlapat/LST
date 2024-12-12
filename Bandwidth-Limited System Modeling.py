Bandwidth.ph
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, lsim
from scipy.linalg import solve_continuous_are
import math  # Python's math module for factorial

# Custom Padé approximation function
def pade_approximation(tau, order):
    """
    Compute the coefficients for the Padé approximation of the delay.
    Parameters:
        tau (float): Delay time.
        order (int): Order of the Padé approximation.
    Returns:
        num (list): Numerator coefficients of the approximation.
        den (list): Denominator coefficients of the approximation.
    """
    n = order
    a = [math.factorial(n + k) // (math.factorial(n - k) * math.factorial(k)) * (-tau / 2)**k for k in range(n + 1)]
    b = [math.factorial(n + k) // (math.factorial(n - k) * math.factorial(k)) * (tau / 2)**k for k in range(n + 1)]
    return np.array(a[::-1]), np.array(b[::-1])

# Parameters
m = 500  # Mass (kg)
b = 30   # Damping coefficient (Ns/m)
k = 3000 # Spring constant (N/m)
tau = 0.2  # Communication delay (seconds)

# State-space matrices
A = np.array([[0, 1], [-k/m, -b/m]])
B = np.array([[0], [1/m]])
C = np.array([[1, 0]])
D = np.array([[0]])

# LQR design
Q = np.eye(2)  # State weighting matrix
R = np.array([[1]])  # Control weighting matrix
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R).dot(B.T).dot(P)

print("LQR Gain Matrix K:")
print(K)

# Custom Padé approximation for delay
pade_num, pade_den = pade_approximation(tau, 4)  # 4th order approximation

# System transfer function without delay
num_sys = [1/m]
den_sys = [1, b/m, k/m]

# Combine system dynamics with delay approximation
num_combined = np.polymul(num_sys, pade_num)
den_combined = np.polymul(den_sys, pade_den)

# Create the delayed system transfer function
delayed_sys_tf = TransferFunction(num_combined, den_combined)

# Time simulation
t = np.linspace(0, 10, 1000)
u = np.ones_like(t)  # Step input
_, y, _ = lsim(delayed_sys_tf.to_ss(), U=u, T=t)

# Plot response
plt.figure(figsize=(10, 6))
plt.plot(t, y, label="System Response with Delay")
plt.title("Bandwidth-Limited Sensor Communication System")
plt.xlabel("Time (s)")
plt.ylabel("Response")
plt.grid()
plt.legend()

# Save the graph as a PNG file
plt.savefig("system_response_with_delay.png", dpi=300, bbox_inches='tight')  # High resolution and tight layout
plt.show()
