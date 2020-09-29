# RPCRISPR-StochasticModel
# This file implements the direct Gillespie algorithm to do a stochastic simulation of the dCas9 repressilator system
# where a dCas9 - sgRNA pair replaces TetR (with sgRNA transcribed from the sponge plasmid).
# Here dCas9-sgRNA bind irreversibly to the target DNA

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from datetime import datetime
from time import time
import random
import pandas as pd
from scipy.interpolate import interp1d
###############################################################################
time_init = datetime.now()
twait = 0.5

###############################################################################
# System parameters and initial values
###############################################################################

t_d = 53*60 # division time, measured in experiments in sec
l_dil = np.log(2) / (t_d) # degradation rate due to dilution in 1/sec
l_sgRNA = 0.02 # rate of sgRNA transcription in 1/seconds from Friedmann et al. 2006
l_deg = 0.01 # RNA degradation rate Friedmann et al., 2006
l = np.array([0.0189, 0.0189, 0.0189])  # gene expression rate from Potvin-Trottier et al., 2016
l_tag = 0.0002  # rate of degradation with deg. tag (1/sec) from Andersen et al., 1998 
l_dCas9_sgRNA = 6.1109*10**6 # dCas9 - sgRNA complex formation rate (1/s) from Westbrook et al. 2019
b_mean = 10 #   # protein burst size from Potvin-Trottier et al., 2016
l_mat = 3.33*10**-3 # GFP maturation rate in 1/ seconds from Iizuka et al., 2011
l_plasmid = l_dil # plasmid replication rate to maintain the copy number
K = np.array([5.9, 2.3])# Michaelis-Menten constants of repression in nM from Niederholtmeyer et al. eLife [cI, TetR]
n = np.array([1.9, 1.2])  # Hill coefficients from Niederholtmeyer et al. eLife [cI, TetR]
l_target= 1.191*10**7 # dCas9 - sgRNA binding rate (1/s) to DNA from Westbrook et al. 2019

###############################################################################
# Initial molecule numbers
###############################################################################
num_bind_sites = np.array([2.0, 2.0]) # number of binding sites per promoter
N_repressilator_mean = 5.0  # Mean repressilator plasmid copy number
N_sponge_mean = 20.0  # Mean sponge plasmids copy numbers
N_repressilator = N_repressilator_mean # Define initial repressilator plasmid copy number
N_sponge = N_sponge_mean  # Define initial sponge plasmids copy numbers
p0 = 1000.0 # intital cI and GFP numbers
p = np.array([p0, 0.0])   # Define initial repressor numbers
#                                  p[0]=cI
#                                  p[1]=TeTR
dCas9 = 0.0 # Number of dCas9 molecules
sgRNA = 0.0  # Number of sgRNA molecules
dCas9_sgRNA = 0.0  # Number of mature dCas9-sgRNA complexes
Nrep_dCas9 = 0.0 # repressilator plasmids with dCas9-sgRNA bound to them
Nspon_1dCas9 = 0.0 # sponge plasmids with 1 dCas9-sgRNA bound to them
Nspon_2dCas9 = 0.0  # sponge plasmids with 2 dCas9-sgRNA bound to them
Nspon_3dCas9 = 0.0 # sponge plasmids with 3 dCas9-sgRNA bound to them

sgRNA1 = 0.0  # number of dummy sgRNA molecules
dCas9_sgRNA1 = 0.0 # number of dummy-sgRNA molecules complexed with dCas9

# This vector is updated after each reaction of the gillespie simulation
p_free = np.array([0.0, 0.0])  # Number of unbound repressors.
# This are the repressors that participate in promoter binding
# and this implements the effect of the sponge. The first element of this array
# corresponds to the dCas9-sgRNA complex and NOT to dCas9 alone

GFP = 0.0 # GFP molecules

GFPmat = p0 # intital number of maturated GFP molecules

Nspon_total = N_sponge # initital number of sponge molecules.

###############################################################################
# General parameters for stochastic simulation
###############################################################################
time_window_of_simulation = 30*60*60  # 50 hours

# Total simulated time (in seconds), updated after each reaction takes place.
T = 0

t_sim = np.array([])  # array storing simulation time points
p_1_sim = np.array([])  # array storing simulation protein cI time series
p_2_sim = np.array([])  # array storing simulation protein TeTR time series
p_1_free_sim = np.array([]) # array storing simulation unbound protein cI time series

# array storing simulation dCas9-sgRNA complex time series
dCas9_sim = np.array([])
sgRNA_sim = np.array([])
dCas9_sgRNA_sim = np.array([])
sgRNA1_sim = np.array([])
dCas9_sgRNA1_sim = np.array([])

# GFP arrays
GFP_sim = np.array([])
GFPmat_sim = np.array([])

# arrays for repressilator copy numbers
N_repressilator_sim = np.array([])
Nrep_dCas9_sim = np.array([])

# arrays for sponge plasmids
N_sponge_sim = np.array([])
Nspon_1dCas9_sim = np.array([])
Nspon_2dCas9_sim = np.array([])
Nspon_3dCas9_sim = np.array([])
Nspon_total_sim = np.array([])

# save initial values
N_repressilator_sim = np.append(N_repressilator_sim, N_repressilator)
Nrep_dCas9_sim = np.append(Nrep_dCas9_sim, Nrep_dCas9)

N_sponge_sim = np.append(N_sponge_sim, N_sponge)
Nspon_1dCas9_sim = np.append(Nspon_1dCas9_sim, Nspon_1dCas9)
Nspon_2dCas9_sim = np.append(Nspon_2dCas9_sim, Nspon_2dCas9)
Nspon_3dCas9_sim = np.append(Nspon_3dCas9_sim, Nspon_3dCas9)
Nspon_total_sim= np.append(Nspon_total_sim, Nspon_total)

t_sim = np.append(t_sim, T)
p_1_sim = np.append(p_1_sim, p[0]) #cI
p_2_sim = np.append(p_2_sim, p[1]) #TetR

p_1_free_sim = np.append(p_1_free_sim, p_free[0])

dCas9_sim = np.append(dCas9_sim, dCas9)
sgRNA_sim = np.append(sgRNA_sim, sgRNA)
dCas9_sgRNA_sim = np.append(dCas9_sgRNA_sim, dCas9_sgRNA)

sgRNA1_sim = np.append(sgRNA1_sim, sgRNA1)
dCas9_sgRNA1_sim = np.append(dCas9_sgRNA1_sim, dCas9_sgRNA1)

GFP_sim = np.append(GFP_sim, GFP)
GFPmat_sim = np.append(GFPmat_sim, GFPmat)


# Define function of which the roots determine the number of unbound
# repressors p_free
def root_function(
        free_repressor,
        total_repressor,
        number_of_binding_sites,
        number_of_plasmids,
        hill_coeff,
        hill_constant):
    # The number of free repressors is given by the roots of this function.
    # The function is negative for free_repressor = 0
    # and crosses the x-axis only once for positive values of free_repressor (for 1< n < 4)
    # This is important for the function which will look for the roots during the runs of the gillespie algorithm.
    # Parameters:
    # number_of_binding_sites: number of repressor molecules which bind per
    # promoter
    f = free_repressor - total_repressor + number_of_binding_sites * number_of_plasmids * \
        free_repressor ** hill_coeff / (free_repressor ** hill_coeff + hill_constant ** hill_coeff)
    return f

# free repressor function
def calculate_free_repressors():
    N = np.array([2 * (N_repressilator + Nrep_dCas9) + (N_sponge + Nspon_1dCas9 + Nspon_2dCas9 + Nspon_3dCas9),
                  2 * (N_repressilator + Nrep_dCas9) + N_sponge + Nspon_1dCas9 + Nspon_2dCas9 + Nspon_3dCas9])
    for index in range(0, 2):
        p_free[index] = brentq(
            root_function,
            0.0,
            p[index],
            args=(
                p[index],
                num_bind_sites[index],
                N[index],
                n[index],
                K[index]))
    return

###############################################################################
# Propensity vector
###############################################################################

# Define propensity vector - all the chemical reactions that can happen
propensity = np.array([lambda l, K, N_repressilator, Nrep_dCas9, n, p_free:
                       l[0] * (N_repressilator + Nrep_dCas9) *
                       (K[0] ** n[0]) / (K[0] ** n[0] + p_free[0] ** n[0]),
                       # p1 dCas9 synthesis, repressed by cI
                       lambda l, K, N_repressilator, Nrep_dCas9, n, p_free:
                       l[1] * (N_repressilator + Nrep_dCas9) * \
                           (K[1] ** n[1]) / (K[1] ** n[1] + p_free[1] ** n[1]),
                       # p2 cI synthesis, repressed by TetR
                       lambda l, N_repressilator:
                       l[2] * N_repressilator,
                       # p3 TetR synthesis, repressed by dCas9-sgRNA
                       lambda l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free:
                       l_sgRNA * (N_sponge + Nspon_1dCas9 + Nspon_2dCas9 + Nspon_3dCas9) * \
                           (K[0] ** n[0]) / (K[0] ** n[0] + p_free[0] ** n[0]),
                       # p4 sgRNA transcription from sponge (high copy) plasmid
                       # and repressed by cI
                       lambda l_dCas9_sgRNA, sgRNA, dCas9:
                       l_dCas9_sgRNA * sgRNA * dCas9,
                       # p5 Formation of dCas9-sgRNA complex
                       lambda l_dil, dCas9:
                       l_dil * dCas9,
                       # p6 dilution of dCas9
                       lambda l_dil, l_tag, p:
                       (l_dil + l_tag) * p[0],
                       # p7 dilution of cI
                       lambda l_dil, l_tag, p:
                       (l_dil + l_tag) * p[1],
                       # p8 dilution of TetR
                       lambda l_dil, dCas9_sgRNA:
                       l_dil * dCas9_sgRNA,
                       # p9 dilution of dCas9-sgRNA complex
                       lambda l_deg, sgRNA:
                       l_deg * sgRNA,
                       # p10 degradation of sgRNA
                       lambda l_plasmid, N_repressilator_mean:
                       l_plasmid * N_repressilator_mean,
                       # p11 Repressilator plasmid replication
                       lambda l_plasmid, N_sponge_mean:
                       l_plasmid * N_sponge_mean,
                       # p12 Sponge plasmid replication
                       lambda l_dil, N_repressilator, Nrep_dCas9:
                       l_dil * (N_repressilator + Nrep_dCas9),
                       # p13 Repressilator plasmid dilution
                       lambda l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9:
                       l_dil * (N_sponge + Nspon_1dCas9 + Nspon_2dCas9 + Nspon_3dCas9),
                       # p14 sponge plasmid dilution
                       lambda l_target, dCas9_sgRNA, N_repressilator:
                       l_target * dCas9_sgRNA * N_repressilator,
                       # p15 binding of dCas9-sgRNA on N_rep
                       lambda l_target, dCas9_sgRNA, N_sponge:
                       l_target * dCas9_sgRNA * N_sponge,
                       # p16 binding of dCas9-sgRNA on N_spon              
                       lambda l_target, dCas9_sgRNA, Nspon_1dCas9:
                       l_target * dCas9_sgRNA * Nspon_1dCas9,         
                       #p17
                       lambda l_target, dCas9_sgRNA, Nspon_2dCas9:
                       l_target * dCas9_sgRNA * Nspon_2dCas9,    
                       #p18
                       lambda l, K, N_repressilator, Nrep_dCas9, n, p_free:
                       l[1] * (N_repressilator + Nrep_dCas9) * \
                           (K[1] ** n[1]) / (K[1] ** n[1] + p_free[1] ** n[1]),
                       # p19 GFP synthesis, repression by TetR
                       lambda l_dil, GFP:
                       l_dil * GFP,
                       # p20 GFP dilution
                       lambda l_mat, GFP:
                       l_mat * GFP,
                       # p21 GFP maturation
                       lambda l_dil, GFPmat:
                       l_dil * GFPmat,
                       # p22 GFPmat dilution
                       lambda l_sgRNA, K, N_repressilator, Nrep_dCas9, n, p_free:
                       l_sgRNA * (N_repressilator + Nrep_dCas9) * \
                           (K[0] ** n[0]) / (K[0] ** n[0] + p_free[0] ** n[0]),
                       # p23 transcription of sgRNA1 - dummy sgRNA
                       lambda l_dCas9_sgRNA, sgRNA1, dCas9:
                       l_dCas9_sgRNA * sgRNA1 * dCas9,
                       # p24 Formation of dCas9-sgRNA1 complex
                       lambda l_dil, dCas9_sgRNA1:
                       l_dil * dCas9_sgRNA1,
                       # p25 dilution of dCas9-sgRNA1 complex
                       lambda l_deg, sgRNA1:
                       l_deg * sgRNA1
                       # p26 degradation of sgRNA1                       
                       ])

###############################################################################
# Total free repressor
###############################################################################

# Calculate number of free repressors for initial conditions
calculate_free_repressors()

# calculate total_propensity: the sum of reaction rates, total rate for
# any reaction to happen
total_propensity = 0

for i in range(0, 2):
    total_propensity += propensity[i](l, K, N_repressilator, Nrep_dCas9, n, p_free)
    
total_propensity += propensity[2](l, N_repressilator)

total_propensity += propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)

total_propensity += propensity[4](l_dCas9_sgRNA, sgRNA, dCas9)
    
total_propensity += propensity[5](l_dil, dCas9)

total_propensity += propensity[6](l_dil, l_tag, p)

total_propensity += propensity[7](l_dil, l_tag, p)

total_propensity += propensity[8](l_dil, dCas9_sgRNA)

total_propensity += propensity[9](l_deg, sgRNA)

total_propensity += propensity[10](l_plasmid, N_repressilator_mean)

total_propensity += propensity[11](l_plasmid, N_sponge_mean)

total_propensity += propensity[12](l_dil, N_repressilator, Nrep_dCas9)

total_propensity += propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9)

total_propensity += propensity[14](l_target, dCas9_sgRNA, N_repressilator)

total_propensity += propensity[15](l_target, dCas9_sgRNA, N_sponge)
total_propensity += propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9)
total_propensity += propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9)

total_propensity += propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)
total_propensity += propensity[19](l_dil, GFP)
total_propensity += propensity[20](l_mat, GFP)
total_propensity += propensity[21](l_dil, GFPmat)

total_propensity += propensity[22](l_sgRNA, K, N_repressilator, Nrep_dCas9, n, p_free)
total_propensity += propensity[23](l_dCas9_sgRNA, sgRNA1, dCas9)
total_propensity += propensity[24](l_dil, dCas9_sgRNA1)
total_propensity += propensity[25](l_deg, sgRNA1)

###############################################################################
# Gillespie Direct Algorithm
###############################################################################

start = time()

while T <= time_window_of_simulation:

    elapsed_time = time()
    
    if elapsed_time - start < 60*twait:
    
        # Detemine first time step drawing a random number from an exponential distribution with mean 1/total_propensity
        # generate random number from a homogeneous distribution in [0,1[
        rho = np.random.random()
    
        # generate random number from a homogeneous distribution in [0,1[
        rho2 = np.random.random()
    
        tau = (1 / total_propensity) * np.log(1 / rho2)
    
    ###############################################################################
    # Calculate propensities
    
        r1 = propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) / total_propensity
    
        r2 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)) / total_propensity
    
        r3 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
              + propensity[2](l, N_repressilator)) / total_propensity
    
        r4 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
              + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)) \
            / total_propensity
    
        r5 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9,n, p_free)
              + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
              + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9)) / total_propensity
    
        r6 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
              + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
              + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9)) / total_propensity
    
        r7 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
              + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
              + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p)) / total_propensity
    
        r8 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
              + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
              + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)) \
            / total_propensity
    
        r9 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
              + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
              + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
              + propensity[8](l_dil, dCas9_sgRNA)) / total_propensity
    
        r10 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA)) / total_propensity
    
        r11 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)) \
              / total_propensity
    
        r12 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean)) / total_propensity
    
        r13 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9)) / total_propensity
        
        r14 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9)) / total_propensity
            
        r15 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9)
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator)) / total_propensity
    
        r16 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9)
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9)
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)) / total_propensity
 
        r17 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9) 
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9)) / total_propensity
    
        r18 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9) 
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9) 
               + propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9)) / total_propensity    
        
        r19 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9)
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9) 
               + propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9) 
               + propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)) / total_propensity    
        
        r20 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9) 
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9) 
               + propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9) 
               + propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[19](l_dil, GFP)) / total_propensity
        
        r21 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9)
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9) 
               + propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9) 
               + propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[19](l_dil, GFP) + propensity[20](l_mat, GFP)) / total_propensity
        
        r22 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9)
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9) 
               + propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9) 
               + propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[19](l_dil, GFP) + propensity[20](l_mat, GFP)
               + propensity[21](l_dil, GFPmat)) / total_propensity      
    
        r23 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9) 
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9) 
               + propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9) 
               + propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[19](l_dil, GFP) + propensity[20](l_mat, GFP)
               + propensity[21](l_dil, GFPmat)
               + propensity[22](l_sgRNA, K, N_repressilator, Nrep_dCas9, n, p_free)) / total_propensity
        
        r24 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9) 
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9) 
               + propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9) 
               + propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[19](l_dil, GFP) + propensity[20](l_mat, GFP)
               + propensity[21](l_dil, GFPmat)
               + propensity[22](l_sgRNA, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[23](l_dCas9_sgRNA, sgRNA1, dCas9)) / total_propensity 
        
        r25 = (propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free) + propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[2](l, N_repressilator) + propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
               + propensity[4](l_dCas9_sgRNA, sgRNA, dCas9) + propensity[5](l_dil, dCas9) + propensity[6](l_dil, l_tag, p) + propensity[7](l_dil, l_tag, p)
               + propensity[8](l_dil, dCas9_sgRNA) + propensity[9](l_deg, sgRNA) + propensity[10](l_plasmid, N_repressilator_mean)
               + propensity[11](l_plasmid, N_sponge_mean) + propensity[12](l_dil, N_repressilator, Nrep_dCas9) 
               + propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9) 
               + propensity[14](l_target, dCas9_sgRNA, N_repressilator) + propensity[15](l_target, dCas9_sgRNA, N_sponge)
               + propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9) 
               + propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9) 
               + propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[19](l_dil, GFP) + propensity[20](l_mat, GFP)
               + propensity[21](l_dil, GFPmat)
               + propensity[22](l_sgRNA, K, N_repressilator, Nrep_dCas9, n, p_free)
               + propensity[23](l_dCas9_sgRNA, sgRNA1, dCas9)
               + propensity[24](l_dil, dCas9_sgRNA1)) / total_propensity  
              
###############################################################################
    # Determine which reaction will happen
###############################################################################

  # determine size of translational burst b, which is geometrically
  # distributed with mean <b>=1/p
        b = np.random.geometric(1 / b_mean)
    
        if 0 < rho <= r1:
            dCas9 += b
            # print('reaction 1')
    
        elif r1 < rho <= r2:
            p[0] += b
            # print('reaction 2')
    
        elif r2 < rho <= r3:
            p[1] += b
        # print('reaction 3')
    
        elif r3 < rho <= r4:
            sgRNA += 1
        # print('reaction 4')
    
        elif r4 < rho <= r5:
            if dCas9 >= 1 and sgRNA >= 1:
                dCas9_sgRNA += 1
                dCas9 -= 1
                sgRNA -= 1
            # print('reaction 5')
    
        elif r5 < rho <= r6:
            if dCas9 >= 1:
                dCas9 -= 1
            # print('reaction 6')
    
        elif r6 < rho <= r7:
            if p[0] >= 1:
                p[0] -= 1
            # print('reaction 7')
    
        elif r7 < rho <= r8:
            if p[1] >= 1:
                p[1] -= 1
                # print('reaction 8')
    
        elif r8 < rho <= r9:
            if dCas9_sgRNA >= 1:
                dCas9_sgRNA -= 1
                # print('reaction 9')
    
        elif r9 < rho <= r10:
            if sgRNA >= 1:
                sgRNA -= 1
                # print('reaction 10')
    
        elif r10 < rho <= r11:
            N_repressilator += 1
    
        elif r11 < rho <= r12:
            N_sponge += 1
    
        elif r12 < rho <= r13:
            RepSum = N_repressilator + Nrep_dCas9    
            rep = [N_repressilator/RepSum, Nrep_dCas9/RepSum]
            rn0 = random.random()
            if 0 < rn0 <= rep[0]:
                N_repressilator -= 1
                    
            else:
                Nrep_dCas9 -= 1
                # print('reaction 13')
    
        elif r13 < rho <= r14:
            SponSum = N_sponge + Nspon_1dCas9 + Nspon_2dCas9 + Nspon_3dCas9
            spon = [N_sponge/SponSum, Nspon_1dCas9/SponSum, Nspon_2dCas9/SponSum, Nspon_3dCas9/SponSum]
            rn0 = random.random()
            
            if 0 < rn0 <= spon[0]:
                N_sponge -= 1
                
            elif spon[0] < rn0 <= spon[0]+spon[1]:
                Nspon_1dCas9 -= 1
                
            elif spon[0]+spon[1] < rn0 <= spon[0]+spon[1]+spon[2]:
                Nspon_2dCas9 -= 1
                
            else:
                Nspon_3dCas9 -= 1
                # print('reaction 14')
                
        elif r14 < rho < r15:
            if dCas9_sgRNA >= 1 and N_repressilator >= 1:
                Nrep_dCas9 += 1
                N_repressilator -= 1
                dCas9_sgRNA -= 1
                # print('reaction 17')
                

        elif r15 < rho <= r16:
            if dCas9_sgRNA >= 1 and N_sponge >= 1:
                Nspon_1dCas9 += 1
                N_sponge -= 1
                dCas9_sgRNA -= 1
                # print('reaction 18')
                
        elif r16 < rho <= r17:
            if dCas9_sgRNA >= 1 and Nspon_1dCas9 >= 1:
                Nspon_2dCas9 += 1
                Nspon_1dCas9 -= 1
                dCas9_sgRNA -= 1
                # print('reaction 19')        
                
        elif r17 < rho <= r18:
            if dCas9_sgRNA >= 1 and Nspon_2dCas9 >= 1:
                Nspon_3dCas9 += 1
                Nspon_2dCas9 -= 1
                dCas9_sgRNA -= 1
                # print('reaction 20')  
      
        elif r18 < rho <= r19:
            GFP += b
        
        elif r19 < rho <= r20:
            if GFP >= 1:
                GFP -= 1        
        
        elif r20 < rho <= r21:
            if GFP >= 1:
                GFP -= 1
                GFPmat += 1
                
        elif r21 < rho <= r22:
            if GFPmat >= 1:
                GFPmat -= 1   
        # update total time
 
        elif r22 < rho < r23:
            sgRNA1 += 1
        # update total time        
        
        elif r23 < rho <= r24:
            if dCas9 >= 1 and sgRNA1 >= 1:
                dCas9_sgRNA1 += 1
                dCas9 -= 1
                sgRNA1 -= 1
        
        elif r24 < rho <= r25:        
            if dCas9_sgRNA1 >= 1:
                dCas9_sgRNA1 -= 1        
        
        elif r25 < rho <= 1:  
           if sgRNA1 >= 1:
                sgRNA1 -= 1               

              
        # update total time    
    
        T += tau
    
        # save simulated values
    
        t_sim = np.append(t_sim, T)
        p_1_sim = np.append(p_1_sim, p[0])
        p_2_sim = np.append(p_2_sim, p[1])

    
        dCas9_sgRNA_sim = np.append(dCas9_sgRNA_sim, dCas9_sgRNA)
        dCas9_sim =np.append(dCas9_sim, dCas9)
        sgRNA_sim = np.append(sgRNA_sim, sgRNA)
        
        GFP_sim = np.append(GFP_sim, GFP)
        GFPmat_sim = np.append(GFPmat_sim, GFPmat)
        
        sgRNA1_sim = np.append(sgRNA1_sim, sgRNA)
        dCas9_sgRNA1_sim = np.append(dCas9_sgRNA1_sim, dCas9_sgRNA1)
        
        N_repressilator_sim = np.append(N_repressilator_sim, N_repressilator)
        N_sponge_sim = np.append(N_sponge_sim, N_sponge)
        
        Nrep_dCas9_sim = np.append(Nrep_dCas9_sim, Nrep_dCas9)
        Nspon_1dCas9_sim = np.append(Nspon_1dCas9_sim, Nspon_1dCas9)
        Nspon_2dCas9_sim = np.append(Nspon_2dCas9_sim, Nspon_2dCas9)
        Nspon_3dCas9_sim = np.append(Nspon_3dCas9_sim, Nspon_3dCas9)
        
        Nspon_total = N_sponge + Nspon_1dCas9 + Nspon_2dCas9 + Nspon_3dCas9
        
        Nspon_total_sim= np.append(Nspon_total_sim, Nspon_total)
        
        # update number of free repressors
    
        calculate_free_repressors()
        
        p_1_free_sim = np.append(p_1_free_sim, p_free[0])
    
        # update total_propensity
    
        total_propensity = 0
    
        total_propensity += propensity[0](l, K, N_repressilator, Nrep_dCas9, n, p_free)
        
        total_propensity += propensity[1](l, K, N_repressilator, Nrep_dCas9, n, p_free)
        
        total_propensity += propensity[2](l, N_repressilator)
        
        total_propensity += propensity[3](l_sgRNA, K, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9, n, p_free)
    
        total_propensity += propensity[4](l_dCas9_sgRNA, sgRNA, dCas9)
        
        total_propensity += propensity[5](l_dil, dCas9)
    
        total_propensity += propensity[6](l_dil, l_tag, p)

        total_propensity += propensity[7](l_dil, l_tag, p)
    
        total_propensity += propensity[8](l_dil, dCas9_sgRNA)
    
        total_propensity += propensity[9](l_deg, sgRNA)
    
        total_propensity += propensity[10](l_plasmid, N_repressilator_mean)
    
        total_propensity += propensity[11](l_plasmid, N_sponge_mean)
    
        total_propensity += propensity[12](l_dil, N_repressilator, Nrep_dCas9)
    
        total_propensity += propensity[13](l_dil, N_sponge, Nspon_1dCas9, Nspon_2dCas9, Nspon_3dCas9)
    
        total_propensity += propensity[14](l_target, dCas9_sgRNA, N_repressilator)

        total_propensity += propensity[15](l_target, dCas9_sgRNA, N_sponge)
        total_propensity += propensity[16](l_target, dCas9_sgRNA, Nspon_1dCas9)
        total_propensity += propensity[17](l_target, dCas9_sgRNA, Nspon_2dCas9)
        
        total_propensity += propensity[18](l, K, N_repressilator, Nrep_dCas9, n, p_free)
        total_propensity += propensity[19](l_dil, GFP)
        total_propensity += propensity[20](l_mat, GFP)
        total_propensity += propensity[21](l_dil, GFPmat)
        
        total_propensity += propensity[22](l_sgRNA, K, N_repressilator, Nrep_dCas9, n, p_free)
        total_propensity += propensity[23](l_dCas9_sgRNA, sgRNA1, dCas9)
        total_propensity += propensity[24](l_dil, dCas9_sgRNA1)
        total_propensity += propensity[25](l_deg, sgRNA1)           
        
    else:
        break

time_end = datetime.now()

###############################################################################

def interp_vector(vector_sim, t_sim, t_interp):
     interp_func = interp1d(t_sim, vector_sim, fill_value='extrapolate',
kind='linear')
     return interp_func(t_interp) # vector interpreted at timepoints according to new time vector

# make sure t_sim is strictly monotonic
cond = np.append(True, ~(np.diff(t_sim) <= 0))

t_end = t_sim[cond][-1]//60 # in full minutes
t_interp = 60*np.arange(0, t_end+1, 1) # new time vector (in seconds), 1 datapoint per minute

# create dataframe with desired vectors
df = pd.DataFrame({'Time': t_interp,
                    'cI': interp_vector(p_1_sim[cond], t_sim[cond], t_interp),
                    'dCas9-sgRNA': interp_vector(dCas9_sgRNA_sim[cond], t_sim[cond], t_interp),
                    'sgRNA': interp_vector(sgRNA_sim[cond], t_sim[cond], t_interp),
                    'TetR': interp_vector(p_2_sim[cond], t_sim[cond], t_interp),
                    'GFP': interp_vector(GFPmat_sim[cond], t_sim[cond], t_interp)})

filename = 'sim_irr.csv'
df.to_csv(filename, index=False) # save to csv

###############################################################################

# print simulated time interval
print('Time for simulating: T=', time_end - time_init)
print('Total simulation time in sec: T=', T)
print('Total simulation time in hours: T=', T / 3600)

# Plot simulation
plt.figure(1)
plt.plot(t_sim / 3600, p_1_sim, label='cI', linewidth=5)
plt.plot(t_sim / 3600, dCas9_sgRNA_sim, label='dCas9-sgRNA', linewidth=5)
plt.plot(t_sim / 3600, p_2_sim, label='TetR', linewidth=5)
plt.plot(t_sim / 3600, dCas9_sim, label='dCas9', linewidth=5)
maxim = max([np.amax(p_1_sim), np.amax(p_2_sim), np.amax(dCas9_sgRNA_sim)])
plt.axis([0, T / 3600, 0, maxim + 0.1 * maxim])
plt.tick_params(labelsize=20)
plt.xlabel('t[hours]', fontsize=20)
plt.ylabel('Number of proteins', fontsize=20)
plt.legend(fontsize=10)

plt.figure(2)
plt.plot(t_sim / 3600, sgRNA_sim, label='sgRNA', linewidth=5)
plt.plot(t_sim / 3600, sgRNA1_sim, label='sgRNA1', linewidth=5)
maxim = max([np.amax(sgRNA_sim), np.amax(sgRNA1_sim)])
plt.axis([0, T / 3600, 0, maxim + 0.1 * maxim])
plt.tick_params(labelsize=20)
plt.xlabel('t[hours]', fontsize=20)
plt.ylabel('Number of sgRNA molecules', fontsize=20)

plt.figure(3)
plt.plot(t_sim / 3600, GFPmat_sim, label='GFP', linewidth=5)
# plt.plot(t_sim / 3600, N_spon_sim, label='Rep', linewidth=5)
maxim = max([np.amax(GFPmat_sim)])
plt.axis([0, T / 3600, 0, maxim + 0.1 * maxim])
plt.tick_params(labelsize=20)
plt.xlabel('t[hours]', fontsize=20)
plt.ylabel('Number of Proteins', fontsize=20)
plt.legend(fontsize=10)

plt.figure(4)
plt.plot(t_sim / 3600, p_1_sim, label='cI', linewidth=5)
plt.plot(t_sim / 3600, dCas9_sgRNA_sim, label='dCas9-sgRNA', linewidth=5)
plt.plot(t_sim / 3600, dCas9_sgRNA1_sim, label='dCas9-sgRNA1', linewidth=5)
plt.plot(t_sim / 3600, p_2_sim, label='TetR', linewidth=5)
maxim = max([np.amax(p_1_sim), np.amax(p_2_sim), np.amax(dCas9_sgRNA_sim)])
plt.axis([0, T / 3600, 0, maxim + 0.1 * maxim])
plt.tick_params(labelsize=20)
plt.xlabel('t[hours]', fontsize=20)
plt.ylabel('Number of proteins', fontsize=20)
plt.legend(fontsize=10)

plt.show()