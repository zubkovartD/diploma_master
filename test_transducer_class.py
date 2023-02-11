
import numpy as np
from scipy.signal import chirp, sweep_poly
import matplotlib.pyplot as plt
from transducer_class import TransducerNonlin
from scipy.fft import fft, fftfreq
from deap import creator
from deap import tools
from deap import base
import math
import random
import elitism
import scipy.io as sio


#%% Construct test signal

fs = 48000 #Sampling frequency

T = 3 #Signal duration
A = 1 #Signal amplitude
phi = 270 # Signal phase
dc_shift = 0 # Constant DC shift to the input signal

t = np.linspace(0, T, T*fs) #Time vector

f0 = 10
f1 = 3100

sig = A*chirp(t, f0=f0, f1=f1, t1=T, method='logarithmic', phi = phi) + dc_shift

#%% Construct LS object

trans = TransducerNonlin() #Create transducer object with default parameters

resp = trans.get_response(sig,'training_data')

trans.reset_states() # Return to zero initial conditions

disp = resp[:,2] # Membrane displacement 
curr = resp[:,1] # Voice-coil current

#%% Find signal spectrums

f = fftfreq(len(sig), 1/fs)[1:len(sig)//2]
w = 2*np.pi*f
jw = 1j*w

U = (2*fft(sig)/len(sig))[1:len(sig)//2]
I = (2*fft(curr)/len(sig))[1:len(sig)//2]
X = (2*fft(disp)/len(sig))[1:len(sig)//2]
V = X*jw

#%% Cut only to low frequencies around resonance

f_low = 20
f_high = 3000

i_start = int(np.where(f == f_low)[0])
i_end = int(np.where(f == f_high)[0])

f = f[i_start:i_end]
f = f[:, np.newaxis]

U = U[i_start:i_end]
U = U[:, np.newaxis]

I = I[i_start:i_end]
I = I[:, np.newaxis]

X = X[i_start:i_end]
X = X[:, np.newaxis]

V = V[i_start:i_end]
V = V[:, np.newaxis]

#%% Impedances

Z_tot = U / I
Z_mec = V / I

DIMENSIONS = 4  # number of dimensions
BOUND_LOW, BOUND_UP = 0, 10.0  # boundaries for all dimensions

POPULATION_SIZE = 10
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # (try also 0.5) probability for mutating an individual
MAX_GENERATIONS = 10
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 1.0  # crowding factor for crossover and mutation

RANDOM_SEED = 1
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

IND_SIZE = 4
creator.create("Individual", list, fitness=creator.FitnessMin)

def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]

toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)

toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

print('aaa')

def myFunc(individual):
    
    #f_res = 139.4
    R_e = 3.54
    K_ms = 2.04e3
    
    L_e = individual[0] * 1e-4
    Bl = individual[1]
    M_ms = individual[2] * 1e-3
    R_ms = individual[3] * 1e-1
    
    print('bbbbb')
    f = ((abs(U / I - (R_e + jw * L_e + Bl ** 2 / (jw * M_ms + K_ms / jw + R_ms))) ** 2).mean()) + ((abs(V / I - Bl / (jw * M_ms + K_ms / jw + R_ms)) ** 2).mean())

    return f, 

toolbox.register("evaluate", myFunc)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/DIMENSIONS)
    
population = toolbox.populationCreator(n=POPULATION_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

best = hof.items[0]
print("-- Best Individual = ", best)
# print("-- Calculated = ",L_e_meas*1e4,Bl_meas,M_ms_meas*1e3,R_ms_meas*1e1)
print("-- Best Fitness = ", best.fitness.values[0])

#%% Time plots

plt.figure()
plt.title("Input signal (Voltage)")
plt.plot(t,sig)

plt.figure()
plt.title("Membrane displacement")
plt.plot(t,disp)

plt.figure()
plt.title("Voice-coil current")
plt.plot(t,curr)

#%% Spectrum plots

plt.figure()
plt.title('Displacement spectrum')
plt.semilogx(f, 20*np.log10(np.abs(X)))
plt.xlabel('Frequency, Hz')

plt.figure()
plt.title('Current spectrum')
plt.semilogx(f, 20*np.log10(np.abs(I)))
plt.xlabel('Frequency, Hz')

plt.figure()
plt.title('Input signal spectrum')
plt.semilogx(f, 20*np.log10(np.abs(U)))
plt.xlabel('Frequency, Hz')

plt.figure()
plt.title('Velocity spectrum')
plt.semilogx(f, 20*np.log10(np.abs(V)))
plt.xlabel('Frequency, Hz')


#%% Impedance plots

plt.figure()
plt.title('Total input impedance')
plt.semilogx(f, 20*np.log10(np.abs(Z_tot)))
plt.xlabel('Frequency, Hz')

plt.figure()
plt.title('Mechanical impedance')
plt.semilogx(f, 20*np.log10(np.abs(Z_mec)))
plt.xlabel('Frequency, Hz')

plt.show()