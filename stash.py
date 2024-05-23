import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fig, axs = plt.subplots()


########################################################################
"""Extra Functions"""
########################################################################
def system_properties(sys):
    """
    Check controllability, observability, and stability of a given state-space system.
    
    Args:
    - sys: State-space system (control.StateSpace object)

    Returns:
    - controllable: Boolean indicating controllability
    - observable: Boolean indicating observability
    - stable: Boolean indicating stability
    """
    # Get the state-space matrices
    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    # Controllability test
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    controllability_matrix = np.zeros((n_states, n_states * n_inputs))
    for i in range(n_states):
        controllability_matrix[:, i*n_inputs:(i+1)*n_inputs] = np.linalg.matrix_power(A, i+1) @ B
    controllable = np.linalg.matrix_rank(controllability_matrix) == n_states

    # Observability test
    n_outputs = C.shape[0]
    observability_matrix = np.zeros((n_states * n_outputs, n_states))
    for i in range(n_states):
        observability_matrix[i*n_outputs:(i+1)*n_outputs, :] = C @ np.linalg.matrix_power(A.T, i+1)
    observable = np.linalg.matrix_rank(observability_matrix) == n_states

    # Stability test
    eigenvalues, _ = np.linalg.eig(A)
    stable = all(np.real(eigenvalues) < 0)
    print("Controllability:", "Controllable" if controllable else "Not Controllable")
    print("Observability:", "Observable" if observable else "Not Observable")
    print("Stability:", "Stable" if stable else "Unstable")


def compute_N(A, B, C, K):
    """
    Compute the N matrix using the given matrices A, B, C, and K.
    
    Parameters:
        A (numpy.ndarray): State matrix of the system.
        B (numpy.ndarray): Input matrix of the system.
        C (numpy.ndarray): Output matrix of the system.
        K (numpy.ndarray): State feedback gain matrix.
        
    Returns:
        numpy.ndarray: N matrix.
    """
    # Compute A - BK
    A_minus_BK = A - np.dot(B, K)
    
    # Compute C(A - BK)^-1B
    CA_minus_BK_inv_B = np.dot(C, np.dot(np.linalg.inv(A_minus_BK), B))
    
    # Compute N = -(C(A - BK)^-1B)^-1
    N = -np.linalg.inv(CA_minus_BK_inv_B)
    
    return N

########################################################################
"""Inverted Pendulum"""
########################################################################
def invertedPendulum(poles = np.array([-1,-1,-1,-1]), LQR_gains = ([1, 1, 1, 1, 1])):
    # Given parameters
    Mp = 0.215  # Pendulum mass
    Mc = 1.608  # Cart mass
    L = 0.314  # Pendulum half-length
    I = 7.06e-3  # Pendulum moment of inertia
    R = 0.16  # Motor terminal resistance
    r = 0.0184  # Pinion radius
    k_g = 3.71  # Gearing ratio
    k_m = 0.0168  # Motor back EMF constant
    g = 9.81 #Gravity
    C = 0 #Damping on cart
    t = np.linspace(0, 20, 1000)  #time span


    # State space matrices
    A = np.array([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, -(Mp**2*L**2*g)/((Mc + Mp)*I + Mc*Mp*L**2), ((I + Mp*L**2)*(C*R*r**2 + k_m**2*k_g**2))/(((Mc+Mp)*I + Mc*Mp*L**2)*R*r**2), 0],
                [0, ((Mc+Mp)*Mp*L*g)/((Mc+Mp)*I+(Mc*Mp*L**2)), (-Mp*L*(C*R*r**2+k_m**2*k_g**2))/((((Mc+Mp)*I)+(Mc*Mp*L**2))*R*r**2), 0]])
    B = np.array([[0], [0], [-((I+Mp*L**2)*k_m*k_g)/(((Mc+Mp)*I+Mc+Mp*L**2)*R*r)], [(Mp*L*k_m*k_g)/((((Mc+Mp)*I)+(Mc*Mp*L**2))*R*r)]])
    C = np.array([[1, 0, 0, 0]])
    D = np.array([[0]])  # Create a zero matrix with the same number of columns as B


    # Create the state-space model
    # Display the state-space model


    # sys = ctrl.ss(A, B, C, D)
    # print("Original Sys (Pendulum)")
    # system_properties(sys)

    # Pole placement method
    K = ctrl.acker(A, B, poles)
    N = compute_N(A,B,C,K) # Tracking Gain
    print(f"Tracking Gain: {N}")
    A_CL = A - np.dot(B, K)
    sys_CL = ctrl.ss(A_CL, N*B, np.eye(4), np.zeros_like(B))
    t_vals,y_vals = ctrl.forced_response(sys_CL, T=t, U=0.1)
    
    # LQR method
    Q = np.array([[LQR_gains[0],0,0,0],
                 [0,LQR_gains[1],0,0],
                 [0,0,LQR_gains[2],0],
                 [0,0,0,LQR_gains[3]]])
    R = 1  # Control input weighting matrix
    K_LQR, _, Eigenvalues = ctrl.lqr(A, B, Q, R)


    N_LQR = compute_N(A,B,C,K_LQR) # Tracking Gain
    print(f"Tracking Gain (LQR): {N_LQR}")
    A_CL = A - np.dot(B, K_LQR)
    sys_CL_LQR = ctrl.ss(A_CL, N_LQR*B, np.eye(4), np.zeros_like(B))
    t_vals,y_vals_LQR = ctrl.forced_response(sys_CL_LQR, T=t, U=0.1)

    # sys = scipy.signal.lti(A, B, C, D)
    # sys_CL_LQR = scipy.signal.lti(A_CL, B, C, D)
    
    # Display the control gains
    print("Pole Placement Control Gains (K):", K)
    print("LQR Control Gains (K_lqr):", K_LQR)
    print(f"Eignenvalues: {Eigenvalues}")

    # Find Control effort (voltage and slew)
    voltage_pp = - np.dot(K, y_vals) + N * 0.1
    voltage_LQR = - np.dot(K_LQR, y_vals_LQR) + N_LQR * 0.1
    slew_pp = np.diff(voltage_pp) 
    slew_LQR = np.diff(voltage_LQR)
    slew_pp = np.abs(slew_pp)
    slew_LQR = np.abs(slew_LQR)

    #Pole Placement
    label_poles = ", ".join([f"{pole:.2f}" for pole in poles])
    LQR_gains = ", ".join([f"{gain}" for gain in LQR_gains])

    # Plot displacement response
    axs.plot( t , y_vals[0], label=f'Pole Placement: {label_poles}')
    axs.plot(t, y_vals_LQR[0], label=f'LQR: {LQR_gains}')
    axs.set_xlabel('Time')
    axs.set_ylabel('Displacement')
    axs.set_title('Displacement Response')
    axs.legend()
    axs.grid(True)

    # Plot angle response
    axs[1].plot(t, y_vals[1], label=f'Pole Placement: {label_poles}')
    axs[1].plot(t, y_vals_LQR[1], label=f'LQR: {LQR_gains}')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Angle')
    axs[1].set_title('Angle Response')
    axs[1].legend()
    axs[1].grid(True)

    # Plot angle response
    axs[2].plot(t , voltage_pp[0], label=f'Pole Placement: {label_poles}')
    axs[2].plot(t, voltage_LQR[0], label=f'LQR: {LQR_gains}')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Volts')
    axs[2].set_title('Voltage (MAX 10)')
    axs[2].legend()
    axs[2].grid(True)

    # Plot angle response
    axs[3].plot(t[:-1] , slew_pp[0], label=f'Pole Placement: {label_poles}')
    axs[3].plot(t[:-1], slew_LQR[0], label=f'LQR: {LQR_gains}')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('V/s')
    axs[3].set_title('Slew (MAX 30)')
    axs[3].legend()
    axs[3].grid(True)


########################################################################
"""Three Carts"""
########################################################################
def threeCartController(poles=np.array([-1,-1,-1,-1,-1,-1]), LQR_gains = ([1,1,1,1,1,1,1])):
    # Given parameters
    m1 = 1.608  # Cart 1 mass
    m2 = m3 = 0.75  # Cart 2/3 mass (unloaded)
    m2_loaded = m3_loaded = 1.25  # Cart 2/3 mass (loaded)
    k = 175  # Spring constant
    C1 = 0  # Cart 1 damping
    C2 = C3 = 3.68  # Cart 2/3 damping
    alpha = 12.45  # Fiddle factor
    R = 1.4  # Motor terminal resistance
    r = 0.0184  # Pinion radius
    k_g = 3.71  # Gearing ratio
    k_b = 0.00176  # Motor back EMF constant
    t = np.linspace(0, 2, 1000)  #time span

    # State space matrices
    A = np.array([
        [0, 0, 0, 1, 0, 0],    # dv1/dt = v1'
        [0, 0, 0, 0, 1, 0],    # dv2/dt = v2'
        [0, 0, 0, 0, 0, 1],    # dv3/dt = v3'
        [-k/m1, k/m1, 0, -C1/m1, 0, 0],      # dv1'/dt
        [k/m2, -2*k/m2, k/m2, 0, -C2/m2, 0], # dv2'/dt
        [0, k/m3, -k/m3, 0, 0, -C3/m3]       # dv3'/dt
    ])

    B = np.array([
    [0],     
    [0],    
    [0],   
    [1/m1],  # Force influence on v1'
    [0],  
    [0]    
    ])

    C = np.array([[0, 0, 1, 0, 0, 0]])
    D = np.array([[0]]) 




    # sys = ctrl.ss(A, B, C, D)
    # print("Original Sys")
    # system_properties(sys)

    # Create the state-space model
    # Display the state-space model

    # Pole placement method (updated)
    K = ctrl.acker(A, B, poles)
    N = compute_N(A,B,C,K) # Tracking Gain
    print(f"Triple-Cart - Tracking Gain: {N}")
    A_CL = A - np.dot(B, K)
    sys_CL = ctrl.ss(A_CL, N*B*0.5, np.eye(6), np.zeros_like(B))
    t_vals,y_vals = ctrl.forced_response(sys_CL, T=t, U=0.5) #step size 500mm
    

    # LQR method
    print(LQR_gains)
    Q = np.array([[LQR_gains[0], 0, 0, 0, 0, 0],
                [0, LQR_gains[1], 0, 0, 0, 0],
                [0, 0, LQR_gains[2], 0, 0, 0],
                [0, 0, 0, LQR_gains[3], 0, 0],
                [0, 0, 0, 0, LQR_gains[4], 0],
                [0, 0, 0, 0, 0, LQR_gains[5]]])  # State weighting matrix
    R = 1  # Control input weighting matrix
    K_LQR, _, Eigenvalues = ctrl.lqr(A, B, Q, R)
    print(Eigenvalues)

    N_LQR = compute_N(A,B,C,K_LQR) # Tracking Gain
    print(f"Triple-Cart - Tracking Gain (LQR): {N_LQR}")
    A_CL = A - B @ K_LQR
    sys_CL_LQR = ctrl.ss(A_CL, N_LQR*B, np.eye(6), np.zeros_like(B))
    t_vals,y_vals_LQR = ctrl.forced_response(sys_CL_LQR, T=t, U=0.5)

    voltage_LQR = - np.dot(K_LQR, y_vals_LQR) + N_LQR * 0.5
    slew_LQR = np.diff(voltage_LQR)
    slew_LQR = np.abs(slew_LQR)

    # # Display the control gains
    print("Pole Placement Control Gains (K):", K)
    print("LQR Control Gains (K_lqr):", K_LQR)
    
    axs.plot(t, y_vals_LQR[0], label=f'Cart1: LQR')
    axs.plot(t, y_vals_LQR[1], label=f'Cart2: LQR')
    axs.plot(t, y_vals_LQR[2], label=f'Cart3: LQR') 

    # axs.plot(t, y_vals[0], label=f'Cart1: PP')
    # axs.plot(t, y_vals[1], label=f'Cart2: PP')
    # axs.plot(t, y_vals[2], label=f'Cart3: PP') 
    # print(f"y_vals: {y_vals_LQR[1]}")

    # axs.plot(t, y_vals_LQR[1], label=f'LQR: {LQR_gains}')
    axs.set_xlabel('Time')
    axs.set_ylabel('Displacement (m)')
    axs.set_title('Cart 3 - Response')
    axs.legend()
    axs.grid(True)

    # axs[2].plot(t, voltage_LQR[0], label=f'LQR: {LQR_gains}')
    # axs[2].set_xlabel('Time [s]')
    # axs[2].set_ylabel('Volts [V]')
    # axs[2].set_title('Voltage (MAX 10)')
    # axs[2].legend()
    # axs[2].grid(True)

    # axs[3].plot(t[:-1], slew_LQR[0], label=f'LQR: {LQR_gains}')
    # axs[3].set_xlabel('Time [s]')
    # axs[3].set_ylabel('Slew [V/s]')
    # axs[3].set_title('Slew (MAX 30)')
    # axs[3].legend()
    # axs[3].grid(True)
    # axs.plot(t, y_vals_LQR[0], label='LQR')




########################################################################
"""UI STUFF"""
########################################################################

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Assuming invertedPendulum and threeCartController are defined elsewhere
# and they plot something on the current figure based on poles and lqr_gains

# Initial parameters
poles = [-1, -2, -3, -4,-5,-6]  # Example pole values
lqr_gains = [190, 190, 190, 1,1,1]  # Example LQR gain values

# Setup the figure and axis
# fig, axs = plt.plot()

# Plot the function directly
threeCartController(poles, lqr_gains)

plt.show()