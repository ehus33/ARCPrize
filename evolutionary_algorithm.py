from deap import base, creator, tools, algorithms
import random
import numpy as np
from transformation import apply_transformations

from functools import partial

def setup_evolutionary_algorithm(model, input_data):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", partial(hybrid_fitness_function, model=model, input_data=input_data))

    return toolbox



def hybrid_fitness_function(individual, model, input_data):
    transformed_grid = apply_transformations(input_data, individual)
    prediction = model.predict(np.array([transformed_grid]))
    return (prediction[0][0],) 

def evolve_population(test_data, model):
    submission = {}
    for task_id, grids in test_data.items():
        predictions = []
        for grid in grids:
            # Set up the toolbox with the current grid as input_data
            toolbox = setup_evolutionary_algorithm(model, grid)
            population = toolbox.population(n=100)
            for gen in range(40):
                offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
                fits = list(map(toolbox.evaluate, offspring))
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                population = toolbox.select(offspring, k=len(population))
            
            best_individual = tools.selBest(population, k=1)[0]
            output_grid = apply_transformations(grid, best_individual)
            predictions.append(output_grid.tolist())
        
        while len(predictions) < 2:
            predictions.append(predictions[0])
        
        submission[task_id] = [{"attempt_1": predictions[0], "attempt_2": predictions[1]}]
    
    return submission
