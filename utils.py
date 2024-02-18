import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

#enumerate all possible board locations
board_locations = set([(i,j) for i in range(8) for j in range(8)])

# function to plot the board with queens
def plot_board(individual):

    chessboard = np.zeros((8, 8))
    chessboard[1::2, 0::2] = 1
    chessboard[0::2, 1::2] = 1
    plt.imshow(chessboard, cmap='binary')
    for i, j in individual:
        plt.text(j, i, '♕', fontsize=30, ha='center', va='center',
                 color='black' if (i - j) % 2 == 0 else 'white')
    plt.show()


def evaluate(individual):

    size = len(individual)
    # check for queens in the same row
    board = np.zeros((8, 8), dtype=int)

    # fill board with queens using indices
    for i, j in individual:
        board[i][j] = 1

    individual = board

    row_clashes = sum(np.clip(np.sum(individual, axis=1), 1, None) - 1)
    # check for queens in the same column
    column_clashes = sum(np.clip(np.sum(individual, axis=0), 1, None) - 1)

    row_indices, column_indices = np.where(individual == 1)

    left_diagonals = [0] * (2 * size - 1)
    right_diagonals = [0] * (2 * size - 1)

    for qr, qc in zip(row_indices, column_indices):
        left_diagonals[qr + qc] += 1
        right_diagonals[size - 1 - qr + qc] += 1

    left_diagonals, right_diagonals = np.array(
        left_diagonals), np.array(right_diagonals)
    # clip diagonals
    left_diagonals = np.clip(left_diagonals, 1, None) - 1
    right_diagonals = np.clip(right_diagonals, 1, None) - 1
    diagonal_clashes = (np.sum(left_diagonals)) + (np.sum(right_diagonals))
    # print(row_clashes, column_clashes, diagonal_clashes)

    clashes = row_clashes + column_clashes + diagonal_clashes

    return clashes,


def crossover(ind_1, ind_2):

    indices_to_cross = random.sample(range(8), 4)
    for i in indices_to_cross:
        # cross index if it isn't present in the individual, else continue
        if ind_1[i] not in ind_2 and ind_2[i] not in ind_1:
            ind_1[i], ind_2[i] = ind_2[i], ind_1[i]
        else:
            continue

    return ind_1, ind_2


def mutate(individual):

    # randomly select one indice
    indice_to_mutate = random.choice(range(8))
    x, y = individual[indice_to_mutate]
    if (y, x) not in individual:
        individual[indice_to_mutate] = (y, x)

    return individual,


def eaSimple(population, toolbox, cxpb, mutpb, ngen, max, stats=None, halloffame=None, ):
    # Create a logbook to store statistics about the evolution process
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the fitness of the initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the initial population if provided
    if halloffame is not None:
        halloffame.update(population)

    # Record the statistics of the initial population
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)
    done = False

    # Begin the generational process
    for gen in range(1, ngen + 1):
        if done:
            return halloffame[0], logbook
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals to create the offspring
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                              offspring[i])
                # Invalidate the fitness values of the offspring
                del offspring[i -
                              1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                # Invalidate the fitness value of the offspring
                del offspring[i].fitness.values

        # Evaluate the fitness of the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            # Check if the solution has been found
            if fit[0] == 0:
                print("Solved")

                done = True

        # Update the hall of fame with the best individuals found so far
        if halloffame is not None:
            halloffame.update(offspring)
            # plot_board(halloffame[0])

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)
    
    print("Done")
    return halloffame[0], logbook


def solve(
    population=1000,
):

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("positions", random.sample, sorted(board_locations), 8)
    toolbox.register("individual", tools.initIterate,
                    creator.Individual, toolbox.positions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register('select', tools.selTournament, tournsize=3)

    seed = 99
    random.seed(seed)

    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)

    solution, logbook = eaSimple(pop, toolbox, cxpb=1, mutpb=1, ngen=100, max = 8,
    stats=stats ,halloffame=hof)

    # plot_board(solution)
