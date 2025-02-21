lass MissionFitnessEvaluator:
    def __init__(self,
                 launch_timing_gene: LaunchTimingGene,
                 payload_assignment_gene: PayloadAssignmentGene,
                 routing_flow_gene: RoutingFlowGene,
                 vehicle_utilization_gene: VehicleUtilizationGene,
                 propellant_resource_gene: PropellantResourceGene,
                 payload_constraints_gene: PayloadConstraintsGene,
                 launch_frequency_gene: LaunchFrequencyGene):
        
        self.launch_timing_gene = launch_timing_gene
        self.payload_assignment_gene = payload_assignment_gene
        self.routing_flow_gene = routing_flow_gene
        self.vehicle_utilization_gene = vehicle_utilization_gene
        self.propellant_resource_gene = propellant_resource_gene
        self.payload_constraints_gene = payload_constraints_gene
        self.launch_frequency_gene = launch_frequency_gene

        # Penalty weights for different constraint violations
        self.penalty_weights = {
            'timing_violation': 10000,
            'payload_capacity': 8000,
            'routing_infeasible': 7000,
            'vehicle_overuse': 6000,
            'propellant_shortage': 5000,
            'constraint_violation': 9000,
            'frequency_violation': 4000
        }

    def evaluate_fitness(self) -> float:
        """
        Calculate overall fitness score for the mission plan.
        Lower score is better. Returns infinity for infeasible solutions.
        """
        total_score = 0
        penalties = 0

        # 1. Basic Mission Cost Calculations
        base_cost = self._calculate_base_mission_cost()
        total_score += base_cost

        # 2. Evaluate each gene and accumulate penalties
        timing_penalty = self._evaluate_launch_timing()
        assignment_penalty = self._evaluate_payload_assignment()
        routing_penalty = self._evaluate_routing()
        utilization_penalty = self._evaluate_vehicle_utilization()
        propellant_penalty = self._evaluate_propellant_resources()
        constraints_penalty = self._evaluate_payload_constraints()
        frequency_penalty = self._evaluate_launch_frequency()

        penalties = (timing_penalty + assignment_penalty + routing_penalty +
                    utilization_penalty + propellant_penalty + 
                    constraints_penalty + frequency_penalty)

        # If solution is completely infeasible, return infinity
        if penalties > 0:
            return float('inf')

        # 3. Optimization Objectives
        total_score += self._calculate_time_efficiency_score()
        total_score += self._calculate_resource_efficiency_score()
        total_score += self._calculate_risk_score()

        return total_score

    def _calculate_base_mission_cost(self) -> float:
        """Calculate fundamental mission costs"""
        cost = 0
        
        # Vehicle usage costs
        for vehicle_id, timeline in self.vehicle_utilization_gene.utilization_timeline.items():
            vehicle = self.vehicle_map[vehicle_id]
            cost += sum(timeline) * vehicle.cost_per_launch

        # Propellant costs
        total_propellant = self.propellant_resource_gene.get_total_propellant_mass()
        cost += total_propellant * PROPELLANT_COST_PER_KG

        return cost

    def _evaluate_launch_timing(self) -> float:
        """Evaluate launch timing feasibility and efficiency"""
        penalty = 0
        
        # Check if any launches are outside their windows
        for payload_id, launch_time in self.launch_timing_gene.launch_times.items():
            payload = self.payload_map[payload_id]
            if not payload.required_launch_window.contains(launch_time):
                penalty += self.penalty_weights['timing_violation']

        return penalty

    def _evaluate_payload_assignment(self) -> float:
        """Evaluate payload assignment feasibility"""
        penalty = 0
        
        for payload_id, vehicle_id in self.payload_assignment_gene.assignments.items():
            payload = self.payload_map[payload_id]
            vehicle = self.vehicle_map[vehicle_id]
            
            # Check mass and volume constraints
            if payload.mass > vehicle.payload_capacity:
                penalty += self.penalty_weights['payload_capacity']
            if payload.volume > vehicle.volume_capacity:
                penalty += self.penalty_weights['payload_capacity']

        return penalty

    def _evaluate_routing(self) -> float:
        """Evaluate routing feasibility and efficiency"""
        penalty = 0
        
        for payload_id, route in self.routing_flow_gene.routing_plan.items():
            # Check if route exceeds vehicle capabilities
            vehicle_id = self.payload_assignment_gene.assignments[payload_id]
            vehicle = self.vehicle_map[vehicle_id]
            
            if route.total_delta_v > vehicle.max_delta_v:
                penalty += self.penalty_weights['routing_infeasible']

        return penalty

    def _evaluate_vehicle_utilization(self) -> float:
        """Evaluate vehicle utilization constraints"""
        penalty = 0
        
        for vehicle_id, timeline in self.vehicle_utilization_gene.utilization_timeline.items():
            vehicle = self.vehicle_map[vehicle_id]
            
            # Check if vehicle usage exceeds availability
            if max(timeline) > vehicle.max_simultaneous_operations:
                penalty += self.penalty_weights['vehicle_overuse']

        return penalty

    def _evaluate_propellant_resources(self) -> float:
        """Evaluate propellant resource constraints"""
        penalty = 0
        
        for (payload_id, arc), requirement in self.propellant_resource_gene.resource_allocations.items():
            vehicle_id = self.payload_assignment_gene.assignments[payload_id]
            vehicle = self.vehicle_map[vehicle_id]
            
            # Check propellant capacity constraints
            if requirement.propellant_mass > vehicle.propellant_capacity:
                penalty += self.penalty_weights['propellant_shortage']

        return penalty

    def _calculate_time_efficiency_score(self) -> float:
        """Calculate score based on mission timeline efficiency"""
        total_time = max(self.launch_timing_gene.launch_times.values())
        return total_time * TIME_PENALTY_FACTOR

    def _calculate_resource_efficiency_score(self) -> float:
        """Calculate score based on resource utilization efficiency"""
        # Consider propellant usage efficiency
        propellant_efficiency = (
            self.propellant_resource_gene.get_total_propellant_mass() / 
            self._calculate_theoretical_minimum_propellant()
        )
        
        # Consider vehicle usage efficiency
        vehicle_efficiency = self._calculate_vehicle_utilization_efficiency()
        
        return (propellant_efficiency + vehicle_efficiency) * EFFICIENCY_WEIGHT

    def _calculate_risk_score(self) -> float:
        """Calculate score based on mission risk factors"""
        risk_score = 0
        
        # Consider launch window margins
        risk_score += self._evaluate_launch_window_margins()
        
        # Consider propellant margins
        risk_score += self._evaluate_propellant_margins()
        
        # Consider vehicle redundancy
        risk_score += self._evaluate_vehicle_redundancy()
        
        return risk_score * RISK_WEIGHT
