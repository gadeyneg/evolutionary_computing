# voyageur de commerce

#%%
from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt

solution_save = []
#%%
f = open(r'c:\Users\gadey\Documents\ITMO\evolutionnary_computing\TSP\data1.txt', 'r')

def get_points(f):
    points = []
    lines = []
    for line in f:
        lines.append(line.replace('\n', ''))
    lines = lines[8:-1]
    for line in lines:
        a,b,c = line.split()
        points.append((int(b), int(c)))
    return(points)

def distance(a, b):
    return( ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)  )

points = get_points(f)

#%%
mutation_probability = 0.20
SIZE = len(points)
POPULATION_SIZE = 800
N_gen = 2000

distances = np.zeros((SIZE, SIZE))

for i in range(SIZE):
    for j in range(SIZE):
        distances[i, j] = distance(points[i], points[j])

#%%
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)

IND_SIZE = SIZE

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# %%

def evaluate(individual):
    res = 0
    for i in range(len(individual)-1):
        res += distances[individual[i], individual[i+1]]
    return res,

toolbox.register('evaluate', evaluate)

 # mutation
def shuffle2(individual):
    a, b = random.randrange(len(individual)), random.randrange(len(individual))
    while a == b :
        a = random.randrange(len(individual))
    individual[a], individual[b] = individual[b], individual[a]
    return(individual)

def switchpart(individual):
    n = random.randrange(10)
    a = random.randrange(len(individual) - (n+1))
    b = a + n
    return individual[:a] + individual[a:b][::-1] + individual[b:]

def shufflelot(individual):
    c = random.randrange(len(individual))
    for i in range(c):
        a, b = random.randrange(len(individual)), random.randrange(len(individual))
        while a == b :
            a = random.randrange(len(individual))
        individual[a], individual[b] = individual[b], individual[a]
    return(individual)

toolbox.register('mutate_3', shufflelot)
toolbox.register('mutate_2', switchpart)
toolbox.register('mutate', shuffle2)

def orderedCrossover(p1, p2):
    a, b = random.randrange(len(p1)), random.randrange(len(p1))
    while a == b :
        a = random.randrange(len(p1))
    a, b = min((a,b)), max((a,b))
    
    child1_core = p1[a:b]
    child1_rest = [value for value in p2 if value not in child1_core]
    child1 = child1_rest[:a] + child1_core + child1_rest[a:]

    child2_core = p2[a:b]
    child2_rest = [value for value in p1 if value not in child2_core]
    child2 = child2_rest[:a] + child2_core + child2_rest[a:]

    return child1, child2

# toolbox.register('mate_2', tools.cxPartialyMatched)
toolbox.register('mate', orderedCrossover)
toolbox.register('select', tools.selBest)

#%%

def main():
    pop = toolbox.population(n = POPULATION_SIZE * 2)
    CXPB, MUTPB, NGEN = 0.5, 0.25, N_gen

    fitness = []

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, int(len(pop)*0.4)) + pop[-int(len(pop)*0.1):]
        
        if g%100 == 0:
            print("génération : ", g, ", score : ", offspring[0].fitness.values)
        
        fitness.append(offspring[0].fitness.values)
        
        random.shuffle(offspring)
        # Clone the selected individuals
        offspring_copy = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            offspring_copy.append(child1)
            offspring_copy.append(child2)

        if g < 1300 :
            for mutant in offspring_copy:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
        else :
            for mutant in offspring_copy:
                if random.random() < 0.2:
                    toolbox.mutate_3(mutant)
                    del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring_copy if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring_copy

    return toolbox.select(pop, len(pop)), fitness

L, fit = main()

print(type(L))

print([ind.fitness.values for ind in L][:5])


# %%

solution = L[0]
solution_save.append(solution)

positions = points
N = len(points)

#%%

fig, ax = plt.subplots(2, sharex=True, sharey=True)         # Prepare 2 plots
ax[0].set_title('Raw nodes')
ax[1].set_title('Optimized tour')
ax[0].scatter([positions[i][0] for i in range(SIZE)], [positions[i][1] for i in range(SIZE)])             # plot A
ax[1].scatter([positions[i][0] for i in range(SIZE)], [positions[i][1] for i in range(SIZE)])             # plot B
totDistance = 0.
for i in range(N-1):
    start_node = solution[i]
    start_pos = positions[start_node]
    next_node = solution[i+1]
    end_pos = positions[next_node]
    ax[1].annotate("",
            xy=start_pos, xycoords='data',
            xytext=end_pos, textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            connectionstyle="arc3"))
    totDistance += distances[start_node, next_node]

textstr = "N nodes: %d\nTotal length: %.3f" % (N, totDistance)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14, Textboxverticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
#%%

plt.plot(range(len(fit)), fit)