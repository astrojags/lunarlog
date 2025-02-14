def payload_cost(chromosome, base_costs, max_payload):
    """
    Compute the cost of delivering supplies based on the payload allocation.
    - chromosome: binary representation of payload allocation
    - base_costs: cost per unit payload for each lunar base
    - max_payload: maximum payload capacity
    """
    total_payload = sum(chromosome)
    if total_payload > max_payload:
        return float('inf')  # Penalize exceeding payload capacity
    return sum(base_costs[i] * chromosome[i] for i in range(len(chromosome)))
