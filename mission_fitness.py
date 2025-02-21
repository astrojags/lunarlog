from typing import Dict, List, Tuple
from population import MissionChromosome, NodeType, Route
import math

class MissionFitnessCalculator:
    """
    Calculates fitness scores for mission chromosomes based on multiple criteria
    including cost, reliability, timeline efficiency, and constraint satisfaction.
    """
    
    def __init__(self, 
                 weight_cost: float = 0.3,
                 weight_reliability: float = 0.2,
                 weight_timeline: float = 0.2,
                 weight_resource: float = 0.15,
                 weight_constraints: float = 0.15):
        """
        Initialize with weights for different fitness components.
        Weights should sum to 1.0
        """
        self.weight_cost = weight_cost
        self.weight_reliability = weight_reliability
        self.weight_timeline = weight_timeline
        self.weight_resource = weight_resource
        self.weight_constraints = weight_constraints
        
        # Baseline costs for normalization
        self.base_launch_cost = 1e7  # $10M base launch cost
        self.cost_per_kg = 1e4       # $10K per kg
        self.setup_cost = 1e6        # $1M setup cost per launch
        
    def calculate_fitness(self, chromosome: MissionChromosome) -> float:
        """
        Calculate overall fitness score for a mission plan.
        Returns value between 0 and 1, higher is better.
        """
        # Calculate component scores
        cost_score = self._calculate_cost_score(chromosome)
        reliability_score = self._calculate_reliability_score(chromosome)
        timeline_score = self._calculate_timeline_score(chromosome)
        resource_score = self._calculate_resource_score(chromosome)
        constraint_score = self._calculate_constraint_score(chromosome)
        
        # Combine weighted scores
        total_fitness = (
            self.weight_cost * cost_score +
            self.weight_reliability * reliability_score +
            self.weight_timeline * timeline_score +
            self.weight_resource * resource_score +
            self.weight_constraints * constraint_score
        )
        
        return total_fitness
    
    def _calculate_cost_score(self, chromosome: MissionChromosome) -> float:
        """
        Calculate cost efficiency score based on:
        - Launch vehicle costs
        - Propellant costs
        - Setup and operations costs
        """
        total_cost = 0.0
        
        # Calculate launch vehicle costs
        for payload_id, vehicle_id in chromosome.payload_assignment.assignments.items():
            vehicle = chromosome.payload_assignment.vehicle_map[vehicle_id]
            payload = chromosome.payload_assignment.payload_map[payload_id]
            
            # Base launch cost
            launch_cost = self.base_launch_cost
            
            # Add mass-based cost
            launch_cost += payload.mass * self.cost_per_kg
            
            # Add setup cost
            launch_cost += self.setup_cost
            
            # Add additional costs for vehicle stacking
            if payload_id in chromosome.vehicle_stacking.vehicle_stacks:
                stack = chromosome.vehicle_stacking.vehicle_stacks[payload_id]
                launch_cost += len(stack) * self.setup_cost * 0.5  # 50% setup cost for additional vehicles
            
            total_cost += launch_cost
        
        # Add propellant costs
        total_propellant = chromosome.propellant_resource.get_total_propellant_mass()
        total_cost += total_propellant * self.cost_per_kg * 0.1  # Propellant cost factor
        
        # Normalize cost score (inverse relationship - lower cost is better)
        max_expected_cost = len(chromosome.payload_assignment.assignments) * (
            self.base_launch_cost + 
            max(p.mass for p in chromosome.payload_assignment.payload_map.values()) * self.cost_per_kg +
            self.setup_cost * 2  # Assume maximum two vehicles per stack
        )
        
        cost_score = 1.0 - (total_cost / max_expected_cost)
        return max(0.0, min(1.0, cost_score))
    
    def _calculate_reliability_score(self, chromosome: MissionChromosome) -> float:
        """
        Calculate reliability score based on:
        - Vehicle reliability factors
        - Route complexity
        - Resource margins
        """
        reliability_scores = []
        
        for payload_id in chromosome.payload_assignment.assignments:
            # Get mission components
            vehicle_id = chromosome.payload_assignment.assignments[payload_id]
            vehicle = chromosome.payload_assignment.vehicle_map[vehicle_id]
            route = chromosome.routing_flow.routing_plan[payload_id]
            
            # Base reliability on vehicle
            vehicle_reliability = self._get_vehicle_reliability(vehicle)
            
            # Adjust for route complexity
            route_complexity = len(route.arcs)
            route_reliability = math.exp(-0.1 * route_complexity)  # Decreases with more complex routes
            
            # Check resource margins
            resource_margin = self._get_resource_margin(chromosome, payload_id, route)
            margin_reliability = min(1.0, resource_margin / 0.2)  # 20% margin gives full score
            
            # Combine factors
            mission_reliability = (
                0.4 * vehicle_reliability +
                0.3 * route_reliability +
                0.3 * margin_reliability
            )
            
            reliability_scores.append(mission_reliability)
        
        # Overall reliability is the product of individual mission reliabilities
        overall_reliability = sum(reliability_scores) / len(reliability_scores)
        return max(0.0, min(1.0, overall_reliability))
    
    def _get_vehicle_reliability(self, vehicle) -> float:
        """Estimate vehicle reliability based on characteristics"""
        # This could be expanded with actual vehicle reliability data
        if "HEAVY" in vehicle.id:
            return 0.95  # Example: Heavy vehicles slightly less reliable
        elif "MEDIUM" in vehicle.id:
            return 0.98
        else:
            return 0.96
    
    def _get_resource_margin(self, 
                           chromosome: MissionChromosome, 
                           payload_id: str, 
                           route: Route) -> float:
        """Calculate average resource margin across mission arcs"""
        margins = []
        for arc in route.arcs:
            requirement = chromosome.propellant_resource.get_arc_requirement(payload_id, arc)
            if requirement:
                margins.append(requirement.margin)
        return sum(margins) / len(margins) if margins else 0.0
    
    def _calculate_timeline_score(self, chromosome: MissionChromosome) -> float:
        """
        Calculate timeline efficiency score based on:
        - Launch window utilization
        - Sequence optimization
        - Pad turnover efficiency
        """
        # Track earliest and latest mission times
        launch_times = chromosome.launch_timing_gene.launch_times
        earliest_launch = min(launch_times.values())
        latest_launch = max(launch_times.values())
        mission_duration = latest_launch - earliest_launch + 1
        
        # Calculate launch window utilization
        window_utilization = len(launch_times) / mission_duration
        
        # Calculate sequence efficiency
        sequence_score = self._calculate_sequence_efficiency(chromosome)
        
        # Calculate pad turnover efficiency
        turnover_score = self._calculate_pad_turnover_efficiency(chromosome)
        
        # Combine scores
        timeline_score = (
            0.4 * window_utilization +
            0.3 * sequence_score +
            0.3 * turnover_score
        )
        
        return max(0.0, min(1.0, timeline_score))
    
    def _calculate_sequence_efficiency(self, chromosome: MissionChromosome) -> float:
        """Calculate how well the mission sequence is optimized"""
        total_gaps = 0
        last_time = None
        
        # Sort launches by time
        sorted_launches = sorted(
            chromosome.launch_timing_gene.launch_times.items(),
            key=lambda x: x[1]
        )
        
        for payload_id, time in sorted_launches:
            if last_time is not None:
                gap = time - last_time
                total_gaps += gap
            last_time = time
        
        # Calculate average gap
        avg_gap = total_gaps / (len(sorted_launches) - 1) if len(sorted_launches) > 1 else 0
        
        # Score decreases with larger average gaps
        return math.exp(-0.2 * avg_gap)
    
    def _calculate_pad_turnover_efficiency(self, chromosome: MissionChromosome) -> float:
        """Calculate efficiency of launch pad usage"""
        # Get pad turnover times from launch frequency gene
        violations = len(chromosome.launch_frequency._find_schedule_violations())
        total_launches = len(chromosome.launch_timing_gene.launch_times)
        
        # Score based on violation ratio
        if total_launches == 0:
            return 0.0
        return 1.0 - (violations / total_launches)
    
    def _calculate_resource_score(self, chromosome: MissionChromosome) -> float:
        """
        Calculate resource utilization efficiency score based on:
        - Propellant usage optimization
        - Vehicle capacity utilization
        - Resource margins
        """
        # Calculate propellant efficiency
        total_propellant = chromosome.propellant_resource.get_total_propellant_mass()
        total_payload_mass = sum(
            payload.mass 
            for payload in chromosome.payload_assignment.payload_map.values()
        )
        propellant_ratio = total_payload_mass / total_propellant if total_propellant > 0 else 0
        
        # Calculate vehicle utilization
        utilization_scores = []
        for payload_id, vehicle_id in chromosome.payload_assignment.assignments.items():
            vehicle = chromosome.payload_assignment.vehicle_map[vehicle_id]
            payload = chromosome.payload_assignment.payload_map[payload_id]
            
            mass_utilization = payload.mass / vehicle.payload_capacity
            volume_utilization = payload.volume / vehicle.volume_capacity
            utilization_scores.append((mass_utilization + volume_utilization) / 2)
        
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0
        
        # Combine scores
        resource_score = (
            0.5 * propellant_ratio +
            0.5 * avg_utilization
        )
        
        return max(0.0, min(1.0, resource_score))
    
    def _calculate_constraint_score(self, chromosome: MissionChromosome) -> float:
        """
        Calculate constraint satisfaction score based on:
        - Payload timing constraints
        - Vehicle utilization constraints
        - Launch frequency constraints
        """
        scores = []
        
        # Check payload constraints
        if chromosome.payload_constraints.validate_launch_times():
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Check vehicle utilization
        if chromosome.vehicle_utilization.is_feasible():
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Check launch frequency
        violations = chromosome.launch_frequency._find_schedule_violations()
        frequency_score = 1.0 - (len(violations) / len(chromosome.launch_timing_gene.launch_times))
        scores.append(frequency_score)
        
        # Overall constraint score is average of all constraint checks
        return sum(scores) / len(scores)