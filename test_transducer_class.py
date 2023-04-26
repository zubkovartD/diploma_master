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
A = 1/10 #Signal amplitude
phi = 270 # Signal phase
dc_shift = 0 # Constant DC shift to the input signal

t = np.linspace(0, T, T*fs) #Time vector

f0 = 10
f1 = 3100

f_res_meas = 140  # Hz
R_e_meas = 3.54  # Ohm
L_e_meas = 0.1334e-3  # H
Bl_meas = 2.49  # N./A
M_ms_meas = 2.8e-3  # kg
C_ms_meas = 0.85690e-3  # m/N
K_ms_meas = 1 / C_ms_meas
R_ms_meas = 0.65  # kg/s

f_res_meas = 1/(2*math.pi*math.sqrt(M_ms_meas/K_ms_meas))

Bl = [0.0013400e12,-0.0070605e9,-0.064718e6,0.022497e3,2.5193]

Bl_last = Bl[-1]

trans = TransducerNonlin(Re = 3.33, Rms = 1.033, Mms = 2.667e-3, Le = [L_e_meas], Bl=Bl, Cms = [C_ms_meas]) #Create transducer object

dc_volt_dc_disp_object = {}
dc_volt_arr = []
dc_disp_arr = []

bl_GA_arr_for_graphic = []

for dc_volt in np.arange(-1, 1.1, 0.2):
    sig = np.ones(T*fs) * dc_volt
    disp = trans.get_response(sig, 'displacement')
    dc_disp = np.mean(disp[len(disp)//2:])
    trans.reset_states()
    dc_volt_dc_disp_object[dc_volt] = dc_disp
    dc_volt_arr.append(dc_volt)
    dc_disp_arr.append(dc_disp)

spectra = []
dc_shift_arr = []
bl_GA_arr = []
best_ind_arr = []
dc_shift_arr_result_params = []
for dc_shift in np.arange(-1, 1.1, 0.2):
    result_params = {}
    result_params['dc_shift'] = dc_shift
    dc_shift_arr.append(dc_shift)

    static_samples = np.zeros(1000) 
    
    sig = np.concatenate((static_samples, A*chirp(t, f0=f0, f1=f1, t1=T, method='logarithmic', phi = phi)), axis=None) + dc_shift

    resp = trans.get_response(sig,'training_data')

    trans.reset_states() # Return to zero initial conditions

    resp = resp[1000:,:]

    volt = resp[:,0]
    trans.reset_states()
    curr = resp[:,1]
    trans.reset_states()
    disp = resp[:,2]
    trans.reset_states()

    f = fftfreq(len(volt), 1/fs)[1:len(volt)//2]
    w = 2*np.pi*f
    jw = 1j*w

    U = (2*fft(volt)/len(volt))[1:len(volt)//2]
    I = (2*fft(curr)/len(volt))[1:len(volt)//2]
    X = (2*fft(disp)/len(volt))[1:len(volt)//2]
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
    
    DIMENSIONS = 1 
    BOUND_LOW, BOUND_UP = 0, 10.0  # boundaries for all dimensions

    # Genetic Algorithm constants:
    POPULATION_SIZE = 100
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
    IND_SIZE = 1
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
        
        L_e = L_e_meas
        Bl = individual[0]
        M_ms = M_ms_meas
        R_ms = R_ms_meas

        
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
    print("-- Calculated = ",Bl_meas)
    print("-- Best Fitness = ", best.fitness.values[0])
    result_params['best'] = best
    bl_GA_arr_for_graphic.append(best[0])
    bl_GA_arr.append(best[0])
    best_ind_arr.append(best)
    f_res = 139.4
    R_e = 3.54

    #L_e = best[0] * 1e-4
    Bl = best[0]
    # M_ms = best[2]*1e-3
    # R_ms = best[3] * 1e-1

    K_ms = M_ms_meas*(f_res*2*math.pi)**2
        
    Z_tot_gen = R_e + jw * L_e_meas + Bl ** 2 / (jw * M_ms_meas + K_ms / jw + R_ms_meas) # Genetically derived total input impedance

    f_res_gen = 1/(2*math.pi*math.sqrt(M_ms_meas/K_ms))

    print('f_res_meas = ',f_res_meas)
    print('f_res_gen = ',f_res_gen)

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    dc_shift_arr_result_params.append(result_params)

for i in range(len(dc_shift_arr)):
    print('dc_shift = ',dc_shift_arr[i])
    print('BL = ',bl_GA_arr[i])
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
bl_GA_arr = []
for i in range(len(dc_shift_arr_result_params)):
    dc_shift_arr.append(dc_shift_arr_result_params[i]['dc_shift'])
    bl_GA_arr.append(dc_shift_arr_result_params[i]['best'][0])

with open('bl_GA_array.txt', 'w') as f:
    # Записать массив в файл
    for item in bl_GA_arr:
        f.write("%s\n" % item)

with open('dc_disp_GA_array.txt', 'w') as f:
    # Записать массив в файл
    for item in dc_disp_arr:
        f.write("%s\n" % item)  

with open('bl_GA_arr_for_graphic_array.txt', 'w') as f:
    # Записать массив в файл
    for item in bl_GA_arr_for_graphic:
        f.write("%s\n" % item)

#BL to dc_shift ratio
plt.figure(figsize=(8, 6.5), dpi=80)
plt.grid()
plt.title('BL to dc_shift ratio')
plt.grid()

plt.xlabel('dc_shift')
plt.ylabel('BL, Nm/A')

plt.plot(dc_shift_arr, bl_GA_arr , label="Measured")

plt.legend()
plt.show()
#Show the best individual for each dc_shift in table

with open('dc_volt_dc_disp_object.json', 'w') as f:
    json.dump(dc_volt_dc_disp_object, f)

plt.figure()
plt.grid()
plt.title('BL to dc_disp ratio')
plt.grid()
plt.ylim([0, 10])
plt.xlabel('dc_disp')
plt.ylabel('BL, Nm/A')

print(dc_disp_arr)
print(bl_GA_arr_for_graphic)
plt.plot(dc_disp_arr, bl_GA_arr_for_graphic, label="Measured by GA")

plt.legend()
plt.show()

trans.show_Bl([-0.01, 0.01], dc_arr =dc_disp_arr, bl_arr = bl_GA_arr_for_graphic)
