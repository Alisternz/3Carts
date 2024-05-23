# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:01:59 2024

@author: mgp57
"""

import numpy as np
import h5py
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import control as ctrl



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

def threeCartController(PC, weighted, poles=np.array([-1,-1,-1,-1,-1,-1]), LQR_gains = ([1,1,1,1,1,1,1])):
    # Given parameters
    m1 = 1.608  # Cart 1 mass
    if weighted == True:
        m2 = m3 = 1.25
    else:
        m2 = m3 = 0.75
    # m2 = m3 = 1.25  # Cart 2/3 mass (unloaded)
    m2_loaded = m3_loaded = 1.25  # Cart 2/3 mass (loaded)
    k = 175  # Spring constant
    C1 = 0  # Cart 1 damping
    C2 = C3 = 3.68  # Cart 2/3 damping
    alpha = 12.45  # Fiddle factor
    R = 1.4  # Motor terminal resistance
    r = 0.0184  # Pinion radius
    k_g = 3.71  # Gearing ratio
    k_b = 0.00176  # Motor back EMF constant
    t = np.linspace(0, len(PC)/1000, len(PC))  #time span

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
    #print(f"Triple-Cart - Tracking Gain: {N}")
    A_CL = A - np.dot(B, K)
    sys_CL = ctrl.ss(A_CL, N*B*0.5, np.eye(6), np.zeros_like(B))
    t_vals,y_vals = ctrl.forced_response(sys_CL, T=t, U=PC) #step size 500mm
    

    # LQR method
    #print(LQR_gains)
    Q = np.array([[LQR_gains[0], 0, 0, 0, 0, 0],
                [0, LQR_gains[1], 0, 0, 0, 0],
                [0, 0, LQR_gains[2], 0, 0, 0],
                [0, 0, 0, LQR_gains[3], 0, 0],
                [0, 0, 0, 0, LQR_gains[4], 0],
                [0, 0, 0, 0, 0, LQR_gains[5]]])  # State weighting matrix
    R = 1  # Control input weighting matrix
    K_LQR, _, Eigenvalues = ctrl.lqr(A, B, Q, R)
    #print(Eigenvalues)

    N_LQR = compute_N(A,B,C,K_LQR) # Tracking Gain
    #print(f"Triple-Cart - Tracking Gain (LQR): {N_LQR}")
    A_CL = A - B @ K_LQR
    sys_CL_LQR = ctrl.ss(A_CL, N_LQR*B, np.eye(6), np.zeros_like(B))
    t_vals,y_vals_LQR = ctrl.forced_response(sys_CL_LQR, T=t, U=PC)

    voltage_LQR = - np.dot(K_LQR, y_vals_LQR) + N_LQR * 0.5
    slew_LQR = np.diff(voltage_LQR)
    slew_LQR = np.abs(slew_LQR)

    # # Display the control gains
    print(f"Current Pole-Placement Control Gains: {K}")
    print(f"Current LQR Gains (K_lqr): {K_LQR}")

    return t, y_vals, y_vals_LQR, voltage_LQR, slew_LQR

#THE FOLLOWING IS OLD PLOTTING STUFF, USE FOR REFERENCE 
    
    # axs.plot(t, y_vals_LQR[0], label=f'Cart1: LQR')
    # axs.plot(t, y_vals_LQR[1], label=f'Cart2: LQR')
    # axs.plot(t, y_vals_LQR[2], label=f'Cart3: LQR') 

    # axs.plot(t, y_vals[0], label=f'Cart1: PP')
    # axs.plot(t, y_vals[1], label=f'Cart2: PP')
    # axs.plot(t, y_vals[2], label=f'Cart3: PP') 
    # print(f"y_vals: {y_vals_LQR[1]}")

    # axs.plot(t, y_vals_LQR[1], label=f'LQR: {LQR_gains}')
    # axs.set_xlabel('Time')
    # axs.set_ylabel('Displacement (m)')
    # axs.set_title('Cart 3 - Response')
    # axs.legend()
    # axs.grid(True)

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


def extract_ENME403_triplecart_data(filename): # Type filename in here, as a string and WITHOUT the .mat suffix. See the example usage in main() below:

    mat_file = scipy.io.loadmat(filename + '.mat')
    data = mat_file[filename]
    time =      data[0][0][0][0][0][2][0]
    C1P =       data[0][0][1][0][0][2][0]
    C1P_GAIN =  data[0][0][1][0][1][2][0]
    C1V =       data[0][0][1][0][2][2][0]
    C1V_GAIN =  data[0][0][1][0][3][2][0]
    C2P =       data[0][0][1][0][4][2][0]
    C2P_GAIN =  data[0][0][1][0][5][2][0]
    C2V =       data[0][0][1][0][6][2][0]
    C2V_GAIN =  data[0][0][1][0][7][2][0]
    C3P =       data[0][0][1][0][8][2][0]
    C3P_GAIN =  data[0][0][1][0][9][2][0]
    C3V =       data[0][0][1][0][10][2][0]
    C3V_GAIN =  data[0][0][1][0][11][2][0]
    CMV =       data[0][0][1][0][12][2][0]
    PC =        data[0][0][1][0][13][2][0]
    RMV =       data[0][0][1][0][14][2][0]
    
    return time, C1P, C1P_GAIN, C1V, C1V_GAIN, C2P, C2P_GAIN, C2V, C2V_GAIN, C3P, C3P_GAIN, C3V, C3V_GAIN, CMV, PC, RMV

"""****************************EXAMPLE USAGE******************************"""
   
def plot(weighted, time, C1P, C1P_GAIN, C1V, C1V_GAIN, C2P, C2P_GAIN, C2V, C2V_GAIN, C3P, C3P_GAIN, C3V, C3V_GAIN, CMV, PC, RMV, t, y_vals, y_vals_LQR, voltage_LQR, slew_LQR, min_index=70000, max_index=73000):
    # Slicing the data arrays between min_index and max_index
   
    #print(f"C1 PosGain: {C1P_GAIN[100]},C2 PosGain: {C2P_GAIN[0]},C3 PosGain: {C3P_GAIN[0]},C1 VelGain: {C1V_GAIN[0]},C2 VelGain: {C2V_GAIN[0]},C3 VelGain: {C3V_GAIN[0]}, N: {RMV[1]}")
    time = time[min_index:max_index]
    C1P = C1P[min_index:max_index]
    C1V = C1V[min_index:max_index]
    C2P = C2P[min_index:max_index]
    C2V = C2V[min_index:max_index]
    C3P = C3P[min_index:max_index]
    C3V = C3V[min_index:max_index]
    PC = PC[min_index:max_index]
    t = t[min_index:max_index]
    y_vals = y_vals[:, min_index:max_index]
    y_vals_LQR = y_vals_LQR[:, min_index:max_index]
    voltage_LQR = voltage_LQR[:, min_index:max_index]  
    slew_LQR = slew_LQR[:, min_index:max_index]

    # CMV and RMV can be sliced similarly if they are to be plotted
    plt.subplot(2, 1, 1)
    plt.plot(t, PC, label='Step Input', color='black', linestyle='--')
    plt.plot(t, C1P, label='Cart1: LAB', color='orange')
    plt.plot(t, C2P, label='Cart2: LAB', color='maroon')
    plt.plot(t, C3P, label='Cart3: LAB', color='crimson')
    if weighted == False:
        plt.plot(t, y_vals_LQR[0], label=f'Cart1: LQR', color='olive')
        plt.plot(t, y_vals_LQR[1], label=f'Cart2: LQR', color='yellowgreen')
        plt.plot(t, y_vals_LQR[2], label=f'Cart3: LQR', color='seagreen') 
        
    else:
        # plt.plot(time, y_vals_LQR[0], label=f'Cart1: 1.25kg')
        # plt.plot(time, y_vals_LQR[1], label=f'Cart2: 1.25kg')
        plt.plot(time, y_vals_LQR[2], label=f'Cart3: 1.25kg') 
    
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid()
    plt.yticks([-0.25, -0.125, 0, 0.125, 0.25])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, C1V, label='C1V')
    plt.plot(time, C2V, label='C2V')
    plt.plot(time, C3V, label='C3V')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (ms^-1)')
    plt.legend()

    
    #plt.plot(time, CMV, label='CMV')
    
    #plt.plot(time, RMV, label='RMV')

def calculateRiseTime(t, y_vals, PC):
    # Calculate the rise time of the system
    # Find the time when the system first reaches 90% of the final value
    # Find the time when the system first reaches 10% of the final value
    # Calculate the difference between the two times
    rise_time = []
    for i in range(3):
        final_value = PC[-1]
        final_value_90 = 0.9 * final_value
        final_value_10 = 0.1 * final_value
        for j in range(len(y_vals[i])):
            if y_vals[i][j] >= final_value_90:
                time_90 = t[j]
                break
        for j in range(len(y_vals[i])):
            if y_vals[i][j] >= final_value_10:
                time_10 = t[j]
                break
        rise_time.append(time_90 - time_10)
    return rise_time

def print_C3P_values(C3P): #56319, 56972
        for i in range(53000, 53500):
            print(f"C3P[{i}]: {C3P[i]}")

def main():
    
    poles = [-1, -2, -3, -4,-5,-6]  # Example pole values
    lqr_gains = [190,190,190,1,1,1,1]  # Example LQR gain values
    weighted = False
    time, C1P, C1P_GAIN, C1V, C1V_GAIN, C2P, C2P_GAIN, C2V, C2V_GAIN, C3P, C3P_GAIN, C3V, C3V_GAIN, CMV, PC, RMV = extract_ENME403_triplecart_data('agi73_set1')
    t, y_vals, y_vals_LQR, voltage_LQR, slew_LQR = threeCartController(PC, weighted, poles, lqr_gains)
    plot(weighted, time, C1P,C1P_GAIN,C1V,C1V_GAIN,C2P,C2P_GAIN,C2V,C2V_GAIN,C3P,C3P_GAIN,C3V,C3V_GAIN,CMV,PC,RMV, t, y_vals, y_vals_LQR, voltage_LQR, slew_LQR, 50000, 60300)
    weighted = True
    #t, y_vals, y_vals_LQR, voltage_LQR, slew_LQR = threeCartController(PC, weighted, poles, lqr_gains)
    #plot(weighted, time, C1P,C1P_GAIN,C1V,C1V_GAIN,C2P,C2P_GAIN,C2V,C2V_GAIN,C3P,C3P_GAIN,C3V,C3V_GAIN,CMV,PC,RMV, t, y_vals, y_vals_LQR, voltage_LQR, slew_LQR, 30000, 38100)
    #print(f"Current Dataset Gains: {C1P_GAIN[0], C2P_GAIN[0], C3P_GAIN[0], C1V_GAIN[0], C2V_GAIN[0], C3V_GAIN[0]}")
    print(print_C3P_values(C3P))
    plt.grid()
    plt.show()
    print(57247- 56319)
    
main()