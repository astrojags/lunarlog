from genetic_algo import genetic_algorithm
from fitness_function import payload_cost

# Problem parameters
POP_SIZE = 20
CHROMOSOME_LENGTH = 10
GENERATIONS = 50
MUTATION_RATE = 0.01

# Example cost data
base_costs = [10, 20, 15, 30, 25, 40, 35, 10, 15, 50]  # Cost per unit payload
MAX_PAYLOAD = 50  # Maximum payload capacity

# Fitness function
def fitness_function(chromosome):
    return -payload_cost(chromosome, base_costs, MAX_PAYLOAD)

# Run the genetic algorithm
best_solution, best_fitness = genetic_algorithm(
    pop_size=POP_SIZE,
    chromosome_length=CHROMOSOME_LENGTH,
    generations=GENERATIONS,
    mutation_rate=MUTATION_RATE
)

print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
