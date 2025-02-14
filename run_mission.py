from population import MissionChromosome, generate_initial_population
from mission_data_1 import (
    example_payloads,
    example_vehicles,
    example_valid_routes,
    example_stacking_rules,
    campaign_parameters
)

def main():
    # Generate initial population using your data
    initial_population = generate_initial_population(
        size=100,
        payload_map=example_payloads,
        vehicle_map=example_vehicles,
        valid_routes=example_valid_routes,
        max_campaign_duration=campaign_parameters["max_campaign_duration"]
    )
    
    # Now you have your initial population to work with
    print(f"Generated {len(initial_population)} valid mission plans")

if __name__ == "__main__":
    main()