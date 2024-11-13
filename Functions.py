"""
Function usefull for the NTD project Homework:
Reported here are the properly built function that return the crucial parameters we need
for the design of the reactor component, such as axal and radial temperature profiles and
the main quantities we can report fo the cooling system:
"""
import numpy as np
import Correlations as co

# We begin by building the axial temperature profile in the coolant keeping the suggested
# subdivision of the 10 main power node spanning the whole height of the fuel column
def Th_Analysis(q1_vec, mdot, T_in, p, R_co, R_ci, R_fo, R_fi, H_node, Por, Pu, Bu, f, n):
    # First we find the geometrial values wich can depend on temperature (will be counted in the input),
    # when we are in cold geometry the value along z will not be constant and D_h will become an array:
    A_ch = (np.sqrt(3) / 4) * (p**2) - (np.pi * (R_co**2)) / 2
    D_h = (4 * A_ch) / (np.pi * R_co)
    
    # Then we inizializing by creating the empty temperature vector wich will be filled with the temperatures 
    # along z at each subsequent iteration; we provide the fisrt given value:
    T_z_cool = np.zeros(np.size(q1_vec) + 1)
    T_z_cool[0] = T_in
    # And here we build the other initialized vectors to be filled at each iteration of the cycle 
    empty_vec = [np.zeros(np.size(q1_vec)) for _ in range(4)]
    u_cool, h_cool, T_z_co, T_node_avg = empty_vec
    # While here we build the empty matrixes that will be filled at each i-th iteration
    T_map = np.zeros((np.size(q1_vec), 3*n))
    
    for i, (q, D, A) in enumerate(zip(q1_vec, D_h, A_ch)): # i is the node index
        # At each iteration we find the upper tempertaure extreme of each node for the one fo the node before,
        # (for the first node we need the lower tempertaure extreme wich is given: T[i = 0] = T_in )
        T_z_cool[i + 1] = T_z_cool[i] + (q * H_node * f) / (mdot * co.cp_cool(T_z_cool[i]))
        T_node_avg[i] = np.mean([T_z_cool[i], T_z_cool[i + 1]]) 
        
        # While now at each iteration we find the outer tempertaure of the cladding at each nodal mid-point:
        u_cool[i] = mdot / (co.rho_cool(T_node_avg[i]) * A)
        h_cool[i] = (co.Nu_cool(co.Pe_cool(T_node_avg[i], u_cool[i], D)) * co.k_cool(T_node_avg[i])) / D
        T_z_co[i] = T_node_avg[i] + q / (2 * np.pi * R_co[i] * h_cool[i])
        
        # And from there we can continue going back of tempertaure radial direction comuping, using the thermal resistences, 
        # the tempertaure profiles at each interface: r = [R_fi, R_fo, R_ci, R_co]. We have found the tempertaure profiles for
        # each zone and we will use them discretizing the radial depndence:
        R_cl = np.linspace(R_co[i], R_ci[i], n + 1)
        R_gap = np.linspace(R_ci[i], R_fo[i], n + 1)
        R_fu = np.linspace(R_fo[i], R_fi[i], n)
        R = np.concatenate((R_cl[:-1], R_gap[:-1], R_fu))
        
        # We initialize the temperature matrix with the cladding outer vector in the first column:
        T_map[i, 0] = T_z_co[i]
        
        # we run at each node the vector of radial positions just built:
        for j in range(np.size(R) - 1): 
            # And according to the subsection of system we are we use the proper shape of temperature profile:
            if R[j] in R_cl[:-1]:
                T_map[i, j + 1] = T_map[i, j] + (f * q * np.log(R[j] / R[j + 1])) / (2 * np.pi * co.k_cl(T_map[i, j]))
                
            elif R[j] in R_gap[:-1]:
                T_map[i, j + 1] = T_map[i, j] + (f * q * np.log(R[j] / R[j + 1])) / (2 * np.pi * co.k_gap(T_map[i, j]))

            elif R[j] in R_fu[:-1]:
                T_map[i, j + 1] = T_map[i, j] + (f * q * ((R[j]**2) - (R[j + 1]**2))) / (4 * np.pi * (R_fo[i])**2 * co.k_fuel(Bu, Por, Pu, T_map[i, j]))

    # And we pack the data in a more convenient way:
    T_ax_vec = np.array((T_map[:, 0], T_map[:, n - 1], T_map[:, 2*n - 1], T_map[:, - 1]))

    return [T_node_avg, T_map, T_ax_vec, R, u_cool, h_cool]

def Th_Expansion(R_fo_cold, T_fo, R_ci_cold, T_ci, R_co_cold, T_co, alpha_l, T_ref):
    # We must keep the axial subdivision into nodes, again [i] is the i-th node index
    empty_vec = [np.zeros(np.size(R_fo_cold)) for _ in range(3)]
    R_fo_new, R_co_new, R_ci_new = empty_vec
    
    for i, (R_fo, R_ci, R_co) in enumerate(zip(R_fo_cold, R_ci_cold, R_co_cold)):
        # Th. expansion of the fuel external radius: 
        R_fo_new[i] = R_fo * (1 + alpha_l * (T_fo[i] - T_ref))
        
        # Th. expansion of the clading inner radius:
        R_ci_new[i] = R_ci * (1 + co.epsilon_cl(T_ci[i]))
        
        # Th. expansion of the cladding outer radius found my impling mass conservation
        eps_cl_cold = co.epsilon_cl(T_ref)
        eps_cl_hot = co.epsilon_cl(np.mean((T_ci[i], T_co[i])))
        R_co_new[i] = np.sqrt((R_ci_new[i]**2) + (co.rho_cl(eps_cl_cold) / co.rho_cl(eps_cl_hot)) * ((R_co**2) - (R_ci**2)))
        
    return [R_fo_new, R_ci_new, R_co_new]
           
        
        
        
        
        