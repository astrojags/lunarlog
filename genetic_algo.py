import numpy as np

def initialize_population(pop_size, chromosome_length):
    """Generate an initial population of random solutions."""
    return np.random.randint(2, size=(pop_size, chromosome_length))

def fitness_function(chromosome):
    """Evaluate the fitness of a chromosome (minimize cost in this case)."""
    return sum(chromosome)  # Differnet fitness functions for each constraint or one fitness function with multiple constraints??

def select_parents(population, fitness_scores):
    """Select parents using roulette wheel selection."""
    probabilities = fitness_scores / fitness_scores.sum()
    indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[indices]

def crossover(parent1, parent2):
    """Perform single-point crossover."""
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

def mutate(chromosome, mutation_rate):
    """Mutate a chromosome with a given mutation rate."""
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def genetic_algorithm(pop_size, chromosome_length, generations, mutation_rate):
    """Main genetic algorithm loop."""
    population = initialize_population(pop_size, chromosome_length)

    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness_function(ind) for ind in population])
        
        # Select parents
        parents = select_parents(population, fitness_scores)
        
        # Perform crossover and mutation
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, mutation_rate))
            next_generation.append(mutate(child2, mutation_rate))
        
        population = np.array(next_generation)

        # Print progress
        best_fitness = fitness_scores.max()
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # Return the best solution
    fitness_scores = np.array([fitness_function(ind) for ind in population])
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual, fitness_scores.max()
