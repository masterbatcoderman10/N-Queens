# Solving 8-Queens problem using Evolutionary Algorithms

## Instructions

To run the program and observe the output, the `solving_n_queens.ipynb` file can be opened in Google Colab. When running the file in colab the `utils.py` file also need to be loaded in the colab envinronment.

The `utils.py` file contains the functions that allow the algorithm to be run.

## Problem Statement

The 8-Queens problem is a classic problem in the field of Artificial Intelligence. The problem is to place 8 queens on an 8x8 chessboard such that no two queens are attacking each other. A queen can attack another queen if it is in the same row, column, or diagonal as the other queen.

One interesting way to solve this problem is through the use of evolutionary algorithms. This is how I solve this problem in this project using the `deap` python library.

## Evolutionary Algorithm

### Fitness Function

The fitness function calculated the number of clashes on the board. This value was then minimized.

### The Algorithm

The algorithm proceeds as follows:

- Get initial population fitnesses.
- Updated the records (hall of fame) with individual fitness statistics.
- Select offspring
    - Or rather the parents for the next generation. These are selected using tournament selection where a specified number of winners are selected.
    - After the offspring individuals are selected, they are cloned. 
        - This is for technical reasons.
- Cross over the offspring with probability
- Delete the fitness values for the crossed over offspring
- Mutate the offspring with probability
- Calculate fitness
- Check if the condition is met
    - In this case, this is checking whether there are no clashes on any board within the population.
    - If the condition is met the evolution stops.
    - Otherwise this process is run for a specified number of times.
