import numpy as np
import deap
from deap import tools, gp, creator, algorithms, base
import operator
import math
import pandas as pd
import matplotlib.pyplot as mp
import random
import pandas as pd
from statistics import mean
colors = ['red', 'blue', 'green', 'darkgreen', 'orange', 'yellow', 'midnightblue', 'blue', 'purple', 'fuchsia', 'saddlebrown','olive', 'slategrey', 'darkred', 'darkslateblue', 'indigo', 'lime', 'teal', 'darkolivegreen', 'mediumvioletred']
gens = [i for i in range(51)]
hits=0
def div(left, right):
    try:
        return left/right
    except ZeroDivisionError:
        return 1 

def logarithm(x):
    try:
        if x > 0:
            return math.log(x)
        else:
            return 0.0
    except Exception as e:
        return 0.0


def protected_exp(x):
    try:
        if x < 700:
            return math.exp(x)
        else:
            return math.exp(700) 
    except Exception as e:
        return 0.0 
    
def protected_sqrt(x):
    try:
        if x > 0:
            return x**0.5
        elif x==0:
            return 0.0
        else:
            return 0.0
    except Exception as e:
        return 0.0
    
def multiply(a, b):
    try:
        a = float(a)
        b = float(b)
    except ValueError:
        return "Error: both inputs must be numeric values."
    try:
        result = a * b
    except OverflowError:
        return "Error: the result is too large to handle."
    return result

def fitness_eval(individual, points, tb):
    evalu = tb.compile(expr=individual)
    sq_error = 0
    for i in range(len(points)):
        x = points[i]
        error = evalu(x) - (x**0.5)
        error = abs(error)
        sq_error = sq_error + error
    if (sq_error)<0.01:
        global hits
        hits = hits+1
    return sq_error,


def primitive_set():
    pset = gp.PrimitiveSet("main", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(multiply, 2)
    pset.addPrimitive(div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(logarithm, 1)
    pset.addPrimitive(protected_exp, 1)
    pset.renameArguments(ARG0='x')
    pset.addTerminal(1)
    return pset

def initialization(pset_): 
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)
    tb = base.Toolbox()
    tb.register('expr', gp.genHalfAndHalf , pset = pset_, min_=1, max_=6)  #plain definition/no execution
    tb.register('individual', tools.initIterate, creator.Individual, tb.expr) #singular object
    tb.register('population', tools.initRepeat, list, tb.individual) # collection of singular objects
    tb.register('compile', gp.compile, pset=pset_)
    tb.register('evaluate', fitness_eval, points=[(random.randint(0, 40) / 10) for x in range(20)], tb = tb) #calculate fitness score
    tb.register('select', tools.selTournament, tournsize=3) #select the parents via tournament selection
    tb.register('mate', gp.cxOnePoint) #crossover of the parents: twopoint
    tb.register('mut_expr', gp.genFull, min_=0, max_=5) #registering the type of individual to be made after crossover
    tb.register('mutate', gp.mutUniform, expr=tb.mut_expr, pset=pset_) #mutation
    tb.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=16))
    tb.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=16))
    return tb


##defining the easimple algo
def easimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
    
    # return population, log

def trial_runs(seed_number, tb, arr1, arr2, popn, colors, gens):
    for i in range(len(seed_number)):
        random.seed(seed_number[i])
        hof = tools.HallOfFame(1)
        pop = tb.population(n=popn)
        stats_fit= tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register('avg', np.mean)
        mstats.register('std', np.std)
        mstats.register('min', np.min)
        mstats.register('max', np.max)
        pop, log = easimple(pop, tb,  0.9, 0.05, 50, stats=mstats, halloffame=hof, verbose=True)
        for j in range(51):
            arr1[i].append(log.chapters['fitness'][j]['min'])
        for j in range(51):
            arr2[i].append(log.chapters['size'][j]['max'])
        # mp.plot(gens, arr1[i], label='Min Fitness')
        # mp.show()
        # mp.plot(gens, arr2[i], label='Max size')
        # mp.show()
        prim_set = primitive_set()
        toolbx = initialization(prim_set)
    return arr1, arr2

def run_main():
    prim_set = primitive_set()
    toolbx = initialization(prim_set)

    seeds=[318, 472, 678, 123, 458, 900, 103, 200, 300, 859, 222, 904, 581, 154, 189, 349, 109, 13, 42, 100]

    f_fitness=[]
    for i in range(len(seeds)):
        f_fitness.append([])

    f_tree=[]
    for i in range(len(seeds)):
        f_tree.append([])
    min_f, tree_f = trial_runs(seeds, toolbx, f_fitness, f_tree, 500, colors, gens)

    return min_f, tree_f,seeds

def tabulation(min_f, tree_f, file_name, seeds):
    f_minfit=[]
    f_maxfit=[]
    f_avgfit=[]
    f_stdfit=[]
    f_avgsize=[]
    f_maxsize=[]
    mp.boxplot(min_f, positions=range(len(min_f)))
    mp.show()
    for i in min_f:
        f_minfit.append(min(i))
        f_maxfit.append(max(i))
        f_avgfit.append(mean(i))
        f_stdfit.append(np.std(i))
    for i in tree_f:
        f_avgsize.append(mean(i))
        f_maxsize.append(max(i))

    for i in range(len(min_f)):
        mp.plot(gens, min_f[i], label=str(seeds[i]), color=colors[i])

    mp.legend()
    mp.show()
    
# print(f6_minfit)
# print(f6_maxfit)
# print(f6_avgfit)
# print(f6_stdfit)
# print(f6_sizemin)
# print(f6_avgsize)

    df_f = pd.DataFrame()
    df_f['MinFit'] = f_minfit
    df_f['MaxFit'] = f_maxfit
    df_f['AvgFit'] = f_avgfit
    df_f['StdFit'] = f_stdfit
    df_f['Average Tree Size'] = f_avgsize
    df_f['Max Tree Size'] = f_maxsize

    # df_f.to_excel(file_name)



arr_1, arr_2, arr_3 = run_main()
tabulation(arr_1, arr_2, 'Function_22.xlsx', arr_3)