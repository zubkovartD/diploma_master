
#%% Plots
import json
import math, cmath
import numpy as np
import scipy.io as sio
from scipy.signal import chirp
from scipy.fft import fft, fftfreq
from deap import creator
from deap import tools
from deap import base
import random
import matplotlib.pyplot as plt
import seaborn as sns
from transducer_class import TransducerNonlin,Transducer

import elitism

fs = 48000 #Sampling frequency

T = 1 #Signal duration
A = 1 #Signal amplitude
phi = 270 # Signal phase
dc_shift = 0 # Constant DC shift to the input signal

t = np.linspace(0, T, T*fs) #Time vector

f0 = 10
f1 = 3100

#sig = A*chirp(t, f0=f0, f1=f1, t1=T, method='logarithmic', phi = phi) + dc_shift

f_res_meas = 140  # Hz
R_e_meas = 3.54  # Ohm
L_e_meas = 1.45e-4  # H
Bl_meas = 2.49  # N./A
M_ms_meas = 2.8e-3  # kg
C_ms_meas = 4.65e-4  # m/N
K_ms_meas = 1 / C_ms_meas
R_ms_meas = 0.65  # kg/s

f_res_meas = 1/(2*math.pi*math.sqrt(M_ms_meas/K_ms_meas))

trans = TransducerNonlin(R_e_meas, [L_e_meas], [0.0013400, -0.0070605, -0.064718, 0.022497, 2.5193], M_ms_meas, [C_ms_meas], R_ms_meas, fs, stateless = True) #Create transducer object

spectra = []
dc_shift_arr = []
bl_gen_arr = []
best_ind_arr = []
dc_shift_arr_result_params = []
for dc_shift in np.arange(-1, 1.1, 0.1):
    result_params = {}
    result_params['dc_shift'] = dc_shift
    dc_shift_arr.append(dc_shift)
    sig = A*chirp(t, f0=f0, f1=f1, t1=T, method='logarithmic', phi = phi) + dc_shift
    
    resp = trans.get_response(sig,'training_data')

    trans.reset_states() # Return to zero initial conditions

    disp = resp[:,2] # Membrane displacement 
    curr = resp[:,1] # Voice-coil current

    f = fftfreq(len(sig), 1/fs)[1:len(sig)//2]
    w = 2*np.pi*f
    jw = 1j*w

    U = (2*fft(sig)/len(sig))[1:len(sig)//2]
    I = (2*fft(curr)/len(sig))[1:len(sig)//2]
    X = (2*fft(disp)/len(sig))[1:len(sig)//2]
    V = X*jw

    f_low = 20
    f_high = 3000

    i_start = int(np.where(f == f_low)[0])
    i_end = int(np.where(f == f_high)[0])

    f = f[i_start:i_end]
    f = f[:, np.newaxis]

    jw = jw[i_start:i_end]
    jw = jw[:, np.newaxis]

    U = U[i_start:i_end]
    U = U[:, np.newaxis]

    I = I[i_start:i_end]
    I = I[:, np.newaxis]

    X = X[i_start:i_end]
    X = X[:, np.newaxis]

    V = V[i_start:i_end]
    V = V[:, np.newaxis]

    Z_tot_mod = U/I # Derived using transducer model
    Z_tot_meas = R_e_meas + jw * L_e_meas + Bl_meas ** 2 / (jw * M_ms_meas + K_ms_meas / jw + R_ms_meas)# derived from linear model using added mass
    
    DIMENSIONS = 4 
    BOUND_LOW, BOUND_UP = 0, 10.0  # boundaries for all dimensions

    # Genetic Algorithm constants:
    POPULATION_SIZE = 300
    P_CROSSOVER = 0.9  # probability for crossover
    P_MUTATION = 0.5   # (try also 0.5) probability for mutating an individual
    MAX_GENERATIONS = 100
    HALL_OF_FAME_SIZE = 30
    CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

    # set the random seed:
    RANDOM_SEED = 40
    random.seed(RANDOM_SEED)

    toolbox = base.Toolbox()

    # define a single objective, minimizing fitness strategy:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # create the Individual class based on list:
    IND_SIZE = 4
    creator.create("Individual", list, fitness=creator.FitnessMin)


    # helper function for creating random real numbers uniformly distributed within a given range [low, up]
    # it assumes that the range is the same for every dimension
    def randomFloat(low, up):
        return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]


    # create an operator that randomly returns a float in the desired range and dimension:
    toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)

    # create the individual operator to fill up an Individual instance:
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

    # create the population operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


    def myFunc(individual):
        
        f_res = 139.4
        R_e = 3.54
        
        L_e = individual[0] * 1e-4
        Bl = individual[1]
        M_ms = individual[2] * 1e-3
        R_ms = individual[3] * 1e-1

        
        K_ms = M_ms*(f_res*2*math.pi)**2
        
        f = ((abs(np.divide(U,I) - (R_e + jw * L_e + np.divide(Bl ** 2,(jw * M_ms + K_ms / jw + R_ms)))) ** 2).mean()) + 500*(((abs(np.divide(V,I) - np.divide(Bl,(jw * M_ms + K_ms / jw + R_ms))) ** 2).mean()))


        return f,  # return a tuple


    toolbox.register("evaluate", myFunc)


    # genetic operators:
    toolbox.register("select", tools.selTournament, tournsize=10)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/DIMENSIONS)
        
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                            ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Calculated = ",L_e_meas*1e4,Bl_meas,M_ms_meas*1e3,R_ms_meas*1e1)
    print("-- Best Fitness = ", best.fitness.values[0])
    # np.append(bl_gen_arr,Bl_meas)
    result_params['best'] = best
    
    bl_gen_arr.append(best[1])
    best_ind_arr.append(best)
    f_res = 139.4
    R_e = 3.54

    L_e = best[0] * 1e-4
    Bl = best[1]
    M_ms = best[2]*1e-3
    R_ms = best[3] * 1e-1


    K_ms = M_ms*(f_res*2*math.pi)**2
        
    Z_tot_gen = R_e + jw * L_e + Bl ** 2 / (jw * M_ms + K_ms / jw + R_ms) # Genetically derived total input impedance

    f_res_gen = 1/(2*math.pi*math.sqrt(M_ms/K_ms))

    print('f_res_meas = ',f_res_meas)
    print('f_res_gen = ',f_res_gen)

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    plt.rc('font', size=20) 
    # Statistics
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.xlim([0,25])
    plt.ylim([-2,12])
    plt.show()

    # Magnitude
    plt.figure(figsize=(8, 6.5), dpi=80)
    plt.grid()
    plt.title('Input electrical impedance (Magnitude)')
    plt.xscale('log')
    plt.grid()

    plt.xlabel('Frequency, Hz')
    plt.ylabel('Magnituge, Ohm')
    plt.plot(f, abs(Z_tot_mod), label="State-space model")
    plt.plot(f, abs(Z_tot_meas), label="Added mass")
    plt.plot(f, abs(Z_tot_gen), label="Genetic")

    plt.legend()
    plt.show()

    # Phase
    plt.figure(figsize=(8, 6.5), dpi=80)
    plt.grid()
    plt.title('Input electrical impedance (Phase)')
    plt.xscale('log')
    plt.grid()

    plt.xlabel('Frequency, Hz')
    plt.ylabel('Phase, rad')
    plt.plot(f, np.angle(Z_tot_mod), label="State-space model")
    plt.plot(f, np.angle(Z_tot_meas), label="Added mass")
    plt.plot(f, np.angle(Z_tot_gen), label="Genetic")

    plt.legend()
    plt.show()
    dc_shift_arr_result_params.append(result_params)

for i in range(len(dc_shift_arr)):
    print('dc_shift = ',dc_shift_arr[i])
    print('BL = ',bl_gen_arr[i])
    print('Best individual = ',best_ind_arr[i])
    
#Save dc_shift_arr_result_params in json file
with open('dc_shift_arr_result_params.json', 'w') as fp:
    json.dump(dc_shift_arr_result_params, fp)


# %%

#Read dc_shift_arr_result_params from json file
with open('dc_shift_arr_result_params.json') as json_file:
    dc_shift_arr_result_params = json.load(json_file)

# Get dc_shift and bl_meas arrays from dc_shift_arr_result_params
dc_shift_arr = []
bl_gen_arr = []
for i in range(len(dc_shift_arr_result_params)):
    dc_shift_arr.append(dc_shift_arr_result_params[i]['dc_shift'])
    bl_gen_arr.append(dc_shift_arr_result_params[i]['best'][1])
  
  
#BL to dc_shift ratio
plt.figure(figsize=(8, 6.5), dpi=80)
plt.grid()
plt.title('BL to dc_shift ratio')
plt.grid()

plt.xlabel('dc_shift')
plt.ylabel('BL, Nm/A')

plt.plot(dc_shift_arr, bl_gen_arr , label="Measured")

plt.legend()
plt.show()

trans.show_Bl(dc_shift_arr, bl_gen_arr, xlim = [-3,3]) 
