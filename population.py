from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import random
from datetime import datetime, timedelta

@dataclass
class TimeWindow:
    """Represents a launch window with start and end times"""
    start: int  # Time step (e.g. month number) when window opens
    end: int    # Time step when window closes
    
    def contains(self, time: int) -> bool:
        return self.start <= time <= self.end

    def random_time(self) -> int:
        return random.randint(self.start, self.end)

@dataclass
class Payload:
    """Represents a payload with its requirements"""
    id: str
    mass: float  # kg
    volume: float  # m³
    required_launch_window: TimeWindow
    precursor_payloads: List[str] = None  # IDs of payloads that must launch before
    co_payloads: List[str] = None  # IDs of payloads that must launch together
    
    def __post_init__(self):
        self.precursor_payloads = self.precursor_payloads or []
        self.co_payloads = self.co_payloads or []

class LaunchTimingGene:
    """
    Represents the launch timing for payloads in the mission.
    Handles dependencies between payloads and launch window constraints.
    """
    def __init__(self, 
                 payload_map: Dict[str, Payload],
                 initial_times: Dict[str, int] = None):
        self.payload_map = payload_map
        self.launch_times = {}  # Maps payload ID to launch time
        
        if initial_times:
            self.launch_times = self._validate_initial_times(initial_times)
        else:
            self._initialize_random_times()

    def _validate_initial_times(self, initial_times: Dict[str, int]) -> Dict[str, int]:
        """Validates provided launch times against constraints"""
        validated_times = {}
        for payload_id, time in initial_times.items():
            payload = self.payload_map[payload_id]
            
            # Check launch window constraint
            if not payload.required_launch_window.contains(time):
                raise ValueError(f"Launch time {time} outside window for payload {payload_id}")
            
            # Check precursor constraints
            for precursor_id in payload.precursor_payloads:
                if precursor_id in initial_times:
                    if initial_times[precursor_id] >= time:
                        raise ValueError(
                            f"Precursor {precursor_id} must launch before {payload_id}")
            
            # Check co-payload constraints
            for co_payload_id in payload.co_payloads:
                if co_payload_id in initial_times:
                    if initial_times[co_payload_id] != time:
                        raise ValueError(
                            f"Co-payload {co_payload_id} must launch with {payload_id}")
            
            validated_times[payload_id] = time
        return validated_times

    def _initialize_random_times(self):
        """Initialize random but valid launch times for all payloads"""
        # First, create a dependency graph to determine initialization order
        dependency_graph = self._create_dependency_graph()
        
        # Initialize in order of dependencies
        for payload_id in self._topological_sort(dependency_graph):
            payload = self.payload_map[payload_id]
            
            # Calculate earliest possible launch time considering precursors
            earliest_time = payload.required_launch_window.start
            for precursor_id in payload.precursor_payloads:
                if precursor_id in self.launch_times:
                    earliest_time = max(
                        earliest_time, 
                        self.launch_times[precursor_id] + 1
                    )
            
            # If there are co-payloads already scheduled, use their time
            for co_payload_id in payload.co_payloads:
                if co_payload_id in self.launch_times:
                    self.launch_times[payload_id] = self.launch_times[co_payload_id]
                    return
            
            # Otherwise, choose random time within constraints
            latest_time = min(
                payload.required_launch_window.end,
                self._get_latest_possible_time(payload_id)
            )
            
            if latest_time < earliest_time:
                raise ValueError(f"No valid launch time window for payload {payload_id}")
            
            self.launch_times[payload_id] = random.randint(earliest_time, latest_time)

    def _create_dependency_graph(self) -> Dict[str, List[str]]:
        """Creates a graph of payload dependencies"""
        graph = {pid: [] for pid in self.payload_map.keys()}
        for pid, payload in self.payload_map.items():
            for precursor in payload.precursor_payloads:
                graph[precursor].append(pid)
        return graph

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Returns payloads in dependency-respecting order"""
        visited = set()
        temp = set()
        order = []
        
        def visit(pid: str):
            if pid in temp:
                raise ValueError("Circular dependency detected")
            if pid not in visited:
                temp.add(pid)
                for dependent in graph[pid]:
                    visit(dependent)
                temp.remove(pid)
                visited.add(pid)
                order.insert(0, pid)
                
        for pid in graph:
            if pid not in visited:
                visit(pid)
        return order

    def _get_latest_possible_time(self, payload_id: str) -> int:
        """Calculate latest possible launch time considering dependent payloads"""
        payload = self.payload_map[payload_id]
        latest = payload.required_launch_window.end
        
        # Look through all payloads for ones that depend on this one
        for pid, p in self.payload_map.items():
            if payload_id in p.precursor_payloads:
                latest = min(latest, p.required_launch_window.start - 1)
        
        return latest

    def mutate(self, mutation_rate: float = 0.1):
        """Randomly modify launch times while maintaining constraints"""
        for payload_id in self.launch_times:
            if random.random() < mutation_rate:
                # Store old time in case new one is invalid
                old_time = self.launch_times[payload_id]
                try:
                    # Try to set a new random time
                    new_time = self._get_valid_random_time(payload_id)
                    self.launch_times[payload_id] = new_time
                except ValueError:
                    # Restore old time if new one was invalid
                    self.launch_times[payload_id] = old_time

    def crossover(self, other: 'LaunchTimingGene') -> Tuple['LaunchTimingGene', 'LaunchTimingGene']:
        """Perform crossover with another LaunchTimingGene"""
        # Create two new offspring
        child1 = LaunchTimingGene(self.payload_map)
        child2 = LaunchTimingGene(self.payload_map)
        
        # For each payload, randomly choose parent's launch time
        for payload_id in self.payload_map:
            if random.random() < 0.5:
                child1.launch_times[payload_id] = self.launch_times[payload_id]
                child2.launch_times[payload_id] = other.launch_times[payload_id]
            else:
                child1.launch_times[payload_id] = other.launch_times[payload_id]
                child2.launch_times[payload_id] = self.launch_times[payload_id]
        
        # Validate and repair if necessary
        child1._repair()
        child2._repair()
        
        return child1, child2

    def _repair(self):
        """Attempt to repair invalid launch times while maintaining as much information as possible"""
        for payload_id in self.payload_map:
            payload = self.payload_map[payload_id]
            
            # Check and repair precursor constraints
            for precursor_id in payload.precursor_payloads:
                if self.launch_times[precursor_id] >= self.launch_times[payload_id]:
                    self.launch_times[payload_id] = self.launch_times[precursor_id] + 1
            
            # Check and repair co-payload constraints
            for co_payload_id in payload.co_payloads:
                self.launch_times[co_payload_id] = self.launch_times[payload_id]

@dataclass
class Vehicle:
    """Represents a launch vehicle with its capabilities"""
    id: str
    payload_capacity: float  # Maximum mass in kg
    volume_capacity: float   # Maximum volume in m³
    specific_impulse: float  # Isp in seconds
    availability_window: TimeWindow
    launch_frequency: int    # Minimum months between launches
    operational_domains: List[str]  # e.g., ["Earth", "LEO", "LLO"]
    
    def can_accommodate(self, payload: Payload) -> bool:
        """Check if vehicle can physically carry the payload"""
        return (payload.mass <= self.payload_capacity and 
                payload.volume <= self.volume_capacity)

class PayloadAssignmentGene:
    """
    Manages the assignment of payloads to vehicles while respecting 
    physical constraints and vehicle availability.
    """
    def __init__(self, 
                 payload_map: Dict[str, Payload],
                 vehicle_map: Dict[str, Vehicle],
                 launch_timing_gene: LaunchTimingGene,
                 initial_assignments: Dict[str, str] = None):
        self.payload_map = payload_map
        self.vehicle_map = vehicle_map
        self.launch_timing_gene = launch_timing_gene
        self.assignments = {}  # Maps payload_id to vehicle_id
        
        # Track vehicle usage times for launch frequency constraints
        self.vehicle_usage_times = {v_id: [] for v_id in vehicle_map}
        
        if initial_assignments:
            self.assignments = self._validate_assignments(initial_assignments)
        else:
            self._initialize_random_assignments()

    def _validate_assignments(self, assignments: Dict[str, str]) -> Dict[str, str]:
        """Validate that assignments meet all constraints"""
        validated = {}
        for payload_id, vehicle_id in assignments.items():
            payload = self.payload_map[payload_id]
            vehicle = self.vehicle_map[vehicle_id]
            
            # Check physical constraints
            if not vehicle.can_accommodate(payload):
                raise ValueError(
                    f"Vehicle {vehicle_id} cannot accommodate payload {payload_id}")
            
            # Check vehicle availability at launch time
            launch_time = self.launch_timing_gene.launch_times[payload_id]
            if not vehicle.availability_window.contains(launch_time):
                raise ValueError(
                    f"Vehicle {vehicle_id} not available at time {launch_time}")
            
            # Check launch frequency constraints
            if not self._respects_launch_frequency(vehicle_id, launch_time):
                raise ValueError(
                    f"Launch frequency violation for vehicle {vehicle_id}")
            
            validated[payload_id] = vehicle_id
            self.vehicle_usage_times[vehicle_id].append(launch_time)
        
        return validated

    def _initialize_random_assignments(self):
        """Create initial random but valid assignments"""
        for payload_id, payload in self.payload_map.items():
            launch_time = self.launch_timing_gene.launch_times[payload_id]
            
            # Find all suitable vehicles
            suitable_vehicles = [
                v_id for v_id, vehicle in self.vehicle_map.items()
                if (vehicle.can_accommodate(payload) and
                    vehicle.availability_window.contains(launch_time) and
                    self._respects_launch_frequency(v_id, launch_time))
            ]
            
            if not suitable_vehicles:
                raise ValueError(
                    f"No suitable vehicle found for payload {payload_id}")
            
            # Randomly select from suitable vehicles
            chosen_vehicle = random.choice(suitable_vehicles)
            self.assignments[payload_id] = chosen_vehicle
            self.vehicle_usage_times[chosen_vehicle].append(launch_time)

    def _respects_launch_frequency(self, vehicle_id: str, launch_time: int) -> bool:
        """Check if adding a launch respects vehicle's minimum launch frequency"""
        vehicle = self.vehicle_map[vehicle_id]
        previous_launches = sorted(self.vehicle_usage_times[vehicle_id])
        
        return all(
            abs(launch_time - prev_time) >= vehicle.launch_frequency
            for prev_time in previous_launches
        )

    def mutate(self, mutation_rate: float = 0.1):
        """Randomly modify vehicle assignments while maintaining feasibility"""
        for payload_id in self.assignments:
            if random.random() < mutation_rate:
                launch_time = self.launch_timing_gene.launch_times[payload_id]
                current_vehicle = self.assignments[payload_id]
                
                # Remove current usage time
                self.vehicle_usage_times[current_vehicle].remove(launch_time)
                
                # Find alternative vehicles
                suitable_vehicles = [
                    v_id for v_id, vehicle in self.vehicle_map.items()
                    if (vehicle.can_accommodate(self.payload_map[payload_id]) and
                        vehicle.availability_window.contains(launch_time) and
                        self._respects_launch_frequency(v_id, launch_time))
                ]
                
                if suitable_vehicles:
                    new_vehicle = random.choice(suitable_vehicles)
                    self.assignments[payload_id] = new_vehicle
                    self.vehicle_usage_times[new_vehicle].append(launch_time)
                else:
                    # Revert if no alternative found
                    self.assignments[payload_id] = current_vehicle
                    self.vehicle_usage_times[current_vehicle].append(launch_time)

    def crossover(self, other: 'PayloadAssignmentGene') -> Tuple['PayloadAssignmentGene', 'PayloadAssignmentGene']:
        """Perform crossover while maintaining feasibility"""
        child1 = PayloadAssignmentGene(self.payload_map, self.vehicle_map, self.launch_timing_gene)
        child2 = PayloadAssignmentGene(self.payload_map, self.vehicle_map, self.launch_timing_gene)
        
        # Try to inherit assignments from parents
        for payload_id in self.payload_map:
            if random.random() < 0.5:
                child1.try_assign(payload_id, self.assignments[payload_id])
                child2.try_assign(payload_id, other.assignments[payload_id])
            else:
                child1.try_assign(payload_id, other.assignments[payload_id])
                child2.try_assign(payload_id, self.assignments[payload_id])
        
        # Repair any invalid assignments
        child1._repair()
        child2._repair()
        
        return child1, child2

    def try_assign(self, payload_id: str, vehicle_id: str) -> bool:
        """Try to assign a payload to a vehicle, return success"""
        launch_time = self.launch_timing_gene.launch_times[payload_id]
        if (self.vehicle_map[vehicle_id].can_accommodate(self.payload_map[payload_id]) and
            self.vehicle_map[vehicle_id].availability_window.contains(launch_time) and
            self._respects_launch_frequency(vehicle_id, launch_time)):
            self.assignments[payload_id] = vehicle_id
            self.vehicle_usage_times[vehicle_id].append(launch_time)
            return True
        return False

    def _repair(self):
        """Fix any invalid assignments"""
        for payload_id in self.payload_map:
            if (payload_id not in self.assignments or
                not self.try_assign(payload_id, self.assignments[payload_id])):
                # Find any valid vehicle
                self._initialize_random_assignments()
                break

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

class NodeType(Enum):
    """Defines all possible nodes in the space logistics network"""
    EARTH = "Earth"
    LEO = "Low_Earth_Orbit"
    GEO = "Geostationary_Orbit"
    LLO = "Low_Lunar_Orbit"
    LUNAR_SURFACE = "Lunar_Surface"
    LUNAR_SOUTH_POLE = "Lunar_South_Pole"
    LUNAR_RESOURCE_1 = "Lunar_Resource_1"
    LUNAR_RESOURCE_2 = "Lunar_Resource_2"
    EML1 = "Earth_Moon_L1"
    EML2 = "Earth_Moon_L2"
    LMT = "Lunar_Mars_Transfer"

@dataclass
class RouteArc:
    """Represents a single route segment between two nodes"""
    origin: NodeType
    destination: NodeType
    delta_v: float  # Delta-V required for transfer (m/s)
    time_of_flight: float  # Duration of transfer (days)
    allowed_vehicles: Set[str]  # Vehicle IDs that can traverse this arc
    
    def is_valid_for_vehicle(self, vehicle: Vehicle) -> bool:
        """Check if a vehicle can traverse this arc"""
        return (vehicle.id in self.allowed_vehicles and
                vehicle.operational_domains.issuperset({self.origin, self.destination}))

@dataclass
class Route:
    """Represents a complete route with multiple arcs"""
    arcs: List[RouteArc]
    
    @property
    def total_delta_v(self) -> float:
        return sum(arc.delta_v for arc in self.arcs)
    
    @property
    def total_time(self) -> float:
        return sum(arc.time_of_flight for arc in self.arcs)


class RoutingFlowGene:
    """
    Manages routing and flow of payloads through the space logistics network,
    considering vehicle capabilities and mission constraints.
    """
    def __init__(self,
                 payload_map: Dict[str, Payload],
                 vehicle_map: Dict[str, Vehicle],
                 launch_timing_gene: LaunchTimingGene,
                 payload_assignment_gene: PayloadAssignmentGene,
                 valid_routes: Dict[Tuple[NodeType, NodeType], List[Route]]):
        
        self.payload_map = payload_map
        self.vehicle_map = vehicle_map
        self.launch_timing_gene = launch_timing_gene
        self.payload_assignment_gene = payload_assignment_gene
        self.valid_routes = valid_routes
        self.routing_plan = {}  # Maps payload_id to its Route
        
        self._initialize_routes()

    def _initialize_routes(self):
        """Initialize feasible routes for all payloads"""
        for payload_id, payload in self.payload_map.items():
            vehicle = self.vehicle_map[
                self.payload_assignment_gene.assignments[payload_id]]
            launch_time = self.launch_timing_gene.launch_times[payload_id]
            
            # Find valid routes for this payload-vehicle combination
            origin = self._get_payload_origin(payload)
            destination = self._get_payload_destination(payload)
            
            valid_routes = self._find_valid_routes(
                origin, destination, vehicle, launch_time)
            
            if not valid_routes:
                raise ValueError(
                    f"No valid route found for payload {payload_id}")
            
            # Select route that minimizes delta-v
            self.routing_plan[payload_id] = min(
                valid_routes, key=lambda r: r.total_delta_v)

    def _find_valid_routes(self,
                          origin: NodeType,
                          destination: NodeType,
                          vehicle: Vehicle,
                          launch_time: int) -> List[Route]:
        """Find all valid routes between origin and destination"""
        key = (origin, destination)
        if key not in self.valid_routes:
            return []
        
        return [
            route for route in self.valid_routes[key]
            if self._is_route_valid(route, vehicle, launch_time)
        ]

    def _is_route_valid(self,
                       route: Route,
                       vehicle: Vehicle,
                       launch_time: int) -> bool:
        """Check if a route is valid for a vehicle at a given time"""
        # Check vehicle capabilities for each arc
        for arc in route.arcs:
            if not arc.is_valid_for_vehicle(vehicle):
                return False
        
        # Check propellant requirements
        if not self._has_sufficient_propellant(route, vehicle):
            return False
            
        # Check time constraints
        if not self._meets_time_constraints(route, launch_time):
            return False
            
        return True

    def _has_sufficient_propellant(self,
                                 route: Route,
                                 vehicle: Vehicle) -> bool:
        """Check if vehicle has enough propellant for the route"""
        required_propellant = self._calculate_propellant_requirement(
            route.total_delta_v, vehicle)
        return required_propellant <= vehicle.propellant_capacity

    def _calculate_propellant_requirement(self,
                                        delta_v: float,
                                        vehicle: Vehicle) -> float:
        """Calculate propellant needed using rocket equation"""
        mass_ratio = math.exp(delta_v / (vehicle.specific_impulse * 9.81))
        return vehicle.dry_mass * (mass_ratio - 1)

    def mutate(self, mutation_rate: float = 0.1):
        """Modify routes while maintaining feasibility"""
        for payload_id in self.routing_plan:
            if random.random() < mutation_rate:
                vehicle = self.vehicle_map[
                    self.payload_assignment_gene.assignments[payload_id]]
                launch_time = self.launch_timing_gene.launch_times[payload_id]
                
                origin = self._get_payload_origin(self.payload_map[payload_id])
                destination = self._get_payload_destination(
                    self.payload_map[payload_id])
                
                valid_routes = self._find_valid_routes(
                    origin, destination, vehicle, launch_time)
                
                if valid_routes:
                    # Choose random valid route
                    self.routing_plan[payload_id] = random.choice(valid_routes)

    def crossover(self, 
                 other: 'RoutingFlowGene'
                ) -> Tuple['RoutingFlowGene', 'RoutingFlowGene']:
        """Perform crossover while maintaining feasibility"""
        child1 = RoutingFlowGene(
            self.payload_map, self.vehicle_map,
            self.launch_timing_gene, self.payload_assignment_gene,
            self.valid_routes)
        child2 = RoutingFlowGene(
            self.payload_map, self.vehicle_map,
            self.launch_timing_gene, self.payload_assignment_gene,
            self.valid_routes)
        
        # Inherit routes from parents
        for payload_id in self.payload_map:
            if random.random() < 0.5:
                child1.try_assign_route(payload_id, self.routing_plan[payload_id])
                child2.try_assign_route(payload_id, other.routing_plan[payload_id])
            else:
                child1.try_assign_route(payload_id, other.routing_plan[payload_id])
                child2.try_assign_route(payload_id, self.routing_plan[payload_id])
        
        return child1, child2

    def try_assign_route(self, payload_id: str, route: Route) -> bool:
        """Try to assign a route to a payload"""
        vehicle = self.vehicle_map[
            self.payload_assignment_gene.assignments[payload_id]]
        launch_time = self.launch_timing_gene.launch_times[payload_id]
        
        if self._is_route_valid(route, vehicle, launch_time):
            self.routing_plan[payload_id] = route
            return True
        return False


class VehicleUtilizationGene:
    """
    Manages the utilization of vehicles across the mission timeline.
    Tracks when vehicles are active/inactive and ensures we don't exceed
    vehicle availability constraints.
    """
    def __init__(self,
                 vehicle_map: Dict[str, Vehicle],
                 launch_timing_gene: LaunchTimingGene,
                 payload_assignment_gene: PayloadAssignmentGene,
                 routing_gene: RoutingFlowGene,
                 max_campaign_duration: int):
        
        self.vehicle_map = vehicle_map
        self.launch_timing_gene = launch_timing_gene
        self.payload_assignment_gene = payload_assignment_gene
        self.routing_gene = routing_gene
        self.max_campaign_duration = max_campaign_duration
        
        # Create a timeline for each vehicle
        # This tracks how many instances of each vehicle are active at each timestep
        self.utilization_timeline = {
            vehicle_id: [0] * max_campaign_duration 
            for vehicle_id in vehicle_map
        }
        
        # Track the maximum number of each vehicle type available simultaneously
        self.max_vehicle_instances = {
            vehicle_id: self._calculate_max_instances(vehicle)
            for vehicle_id, vehicle in vehicle_map.items()
        }
        
        self._initialize_utilization()

    def _calculate_max_instances(self, vehicle: Vehicle) -> int:
        """
        Calculate maximum number of instances of a vehicle that can be active
        simultaneously based on manufacturing and operational constraints.
        """
        # This could be expanded based on actual vehicle production capabilities
        if vehicle.id.startswith("Heavy"):
            return 2  # Example: Heavy vehicles are limited
        elif vehicle.id.startswith("Reusable"):
            return 3  # Example: Reusable vehicles have more availability
        else:
            return 4  # Default maximum instances

    def _initialize_utilization(self):
        """
        Initialize the utilization timeline based on payload assignments
        and routing plans.
        """
        # Clear existing utilization
        for timeline in self.utilization_timeline.values():
            timeline.clear()
            timeline.extend([0] * self.max_campaign_duration)
        
        # For each payload, mark the vehicle as utilized during its mission
        for payload_id, payload in self.payload_map.items():
            vehicle_id = self.payload_assignment_gene.assignments[payload_id]
            route = self.routing_gene.routing_plan[payload_id]
            launch_time = self.launch_timing_gene.launch_times[payload_id]
            
            # Calculate mission duration from route
            mission_duration = self._calculate_mission_duration(route)
            
            # Mark vehicle as active throughout its mission
            for t in range(launch_time, min(launch_time + mission_duration,
                                          self.max_campaign_duration)):
                self.utilization_timeline[vehicle_id][t] += 1
                
                # Verify we haven't exceeded maximum instances
                if (self.utilization_timeline[vehicle_id][t] > 
                        self.max_vehicle_instances[vehicle_id]):
                    raise ValueError(
                        f"Vehicle {vehicle_id} capacity exceeded at time {t}")

    def _calculate_mission_duration(self, route: Route) -> int:
        """
        Calculate total mission duration including route time and 
        operational margins.
        """
        base_duration = math.ceil(route.total_time)  # Route time in days
        operational_margin = 2  # Additional days for operations
        return base_duration + operational_margin

    def is_feasible(self) -> bool:
        """
        Check if the current utilization plan is feasible regarding
        vehicle availability constraints.
        """
        for vehicle_id, timeline in self.utilization_timeline.items():
            # Check maximum instances constraint
            if max(timeline) > self.max_vehicle_instances[vehicle_id]:
                return False
            
            # Check launch frequency constraints
            vehicle = self.vehicle_map[vehicle_id]
            active_periods = self._get_active_periods(timeline)
            for start1, end1 in active_periods:
                for start2, end2 in active_periods:
                    if start1 < start2 and start2 - end1 < vehicle.launch_frequency:
                        return False
        
        return True

    def _get_active_periods(self, timeline: List[int]) -> List[Tuple[int, int]]:
        """
        Extract periods where a vehicle is active from its timeline.
        Returns list of (start, end) tuples.
        """
        active_periods = []
        start = None
        
        for t, active in enumerate(timeline):
            if active and start is None:
                start = t
            elif not active and start is not None:
                active_periods.append((start, t-1))
                start = None
                
        if start is not None:
            active_periods.append((start, len(timeline)-1))
            
        return active_periods

    def mutate(self, mutation_rate: float = 0.1):
        """
        Attempt to modify vehicle utilization while maintaining feasibility.
        """
        # Store current state in case we need to revert
        original_timeline = copy.deepcopy(self.utilization_timeline)
        
        try:
            # Attempt mutations
            modified = False
            for vehicle_id, timeline in self.utilization_timeline.items():
                if random.random() < mutation_rate:
                    active_periods = self._get_active_periods(timeline)
                    if active_periods:
                        # Randomly select a period to modify
                        period = random.choice(active_periods)
                        
                        # Try to shift the period slightly
                        shift = random.randint(-2, 2)
                        self._shift_active_period(vehicle_id, period, shift)
                        modified = True
            
            # If modifications made, verify feasibility
            if modified and not self.is_feasible():
                self.utilization_timeline = original_timeline
                
        except ValueError:
            # Revert if any error occurs
            self.utilization_timeline = original_timeline

    def _shift_active_period(self, 
                           vehicle_id: str, 
                           period: Tuple[int, int], 
                           shift: int):
        """
        Attempt to shift an active period by a given amount of time steps.
        """
        start, end = period
        new_start = max(0, start + shift)
        new_end = min(self.max_campaign_duration - 1, end + shift)
        
        # Clear old period
        for t in range(start, end + 1):
            self.utilization_timeline[vehicle_id][t] -= 1
            
        # Set new period
        for t in range(new_start, new_end + 1):
            self.utilization_timeline[vehicle_id][t] += 1

    def crossover(self, 
                 other: 'VehicleUtilizationGene'
                 ) -> Tuple['VehicleUtilizationGene', 'VehicleUtilizationGene']:
        """
        Perform crossover operation while maintaining feasibility.
        """
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        # For each vehicle, randomly swap utilization patterns
        for vehicle_id in self.vehicle_map:
            if random.random() < 0.5:
                child1.utilization_timeline[vehicle_id], \
                child2.utilization_timeline[vehicle_id] = \
                    child2.utilization_timeline[vehicle_id], \
                    child1.utilization_timeline[vehicle_id]
        
        # Verify and repair if necessary
        if not child1.is_feasible():
            child1._repair()
        if not child2.is_feasible():
            child2._repair()
        
        return child1, child2

    def _repair(self):
        """
        Attempt to repair an infeasible utilization plan by adjusting
        active periods.
        """
        for vehicle_id, timeline in self.utilization_timeline.items():
            max_instances = self.max_vehicle_instances[vehicle_id]
            
            # Find and fix overutilization
            for t in range(len(timeline)):
                while timeline[t] > max_instances:
                    # Find an active mission to delay
                    for payload_id, payload in self.payload_map.items():
                        if (self.payload_assignment_gene.assignments[payload_id] 
                                == vehicle_id):
                            launch_time = self.launch_timing_gene.launch_times[
                                payload_id]
                            if launch_time == t:
                                # Try to delay this mission
                                new_time = self._find_next_available_time(
                                    vehicle_id, t)
                                if new_time:
                                    self.launch_timing_gene.launch_times[
                                        payload_id] = new_time
                                    timeline[t] -= 1
                                    timeline[new_time] += 1
                                    break

    def _find_next_available_time(self, 
                                vehicle_id: str, 
                                after_time: int) -> Optional[int]:
        """
        Find the next time step where a vehicle can be utilized.
        """
        timeline = self.utilization_timeline[vehicle_id]
        max_instances = self.max_vehicle_instances[vehicle_id]
        
        for t in range(after_time + 1, self.max_campaign_duration):
            if timeline[t] < max_instances:
                return t
        
        return None


from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
import random

@dataclass
class StackingRule:
    """
    Defines rules for how vehicles can be stacked together.
    Think of this like a instruction manual for assembling vehicles.
    """
    upper_vehicle: str  # Vehicle ID that goes on top
    lower_vehicle: str  # Vehicle ID that goes below
    max_payload_fraction: float  # How much of the lower vehicle's payload capacity can be used
    interface_mass: float  # Mass of connecting hardware between vehicles
    
    def is_valid_stack(self, upper: Vehicle, lower: Vehicle) -> bool:
        """Check if two vehicles can be physically connected"""
        # Calculate total mass of upper vehicle
        upper_mass = upper.dry_mass + upper.propellant_capacity
        
        # Check if lower vehicle can support this mass
        max_allowed_mass = lower.payload_capacity * self.max_payload_fraction
        return upper_mass + self.interface_mass <= max_allowed_mass

class VehicleStackingGene:
    """
    Manages how vehicles are combined into stacks for missions.
    This is like a blueprint for assembling our space vehicles.
    """
    def __init__(self,
                 vehicle_map: Dict[str, Vehicle],
                 payload_assignment_gene: PayloadAssignmentGene,
                 stacking_rules: Dict[Tuple[str, str], StackingRule]):
        
        self.vehicle_map = vehicle_map
        self.payload_assignment_gene = payload_assignment_gene
        self.stacking_rules = stacking_rules
        
        # Dictionary mapping payload IDs to their vehicle stacks
        # Each stack is a list of vehicle IDs from bottom to top
        self.vehicle_stacks: Dict[str, List[str]] = {}
        
        # Track which vehicles are already part of stacks
        self.assigned_vehicles: Set[str] = set()
        
        self._initialize_stacks()

    def _initialize_stacks(self):
        """
        Create initial vehicle stacks that satisfy mission requirements.
        Like solving a puzzle to find which vehicles work well together.
        """
        self.vehicle_stacks.clear()
        self.assigned_vehicles.clear()
        
        # Process each payload to create appropriate stacks
        for payload_id, payload in self.payload_map.items():
            primary_vehicle_id = self.payload_assignment_gene.assignments[payload_id]
            
            # Try to create a stack for this payload
            stack = self._build_feasible_stack(primary_vehicle_id, payload)
            if stack:
                self.vehicle_stacks[payload_id] = stack
                self.assigned_vehicles.update(stack)
            else:
                # If no stack possible, use single vehicle
                self.vehicle_stacks[payload_id] = [primary_vehicle_id]
                self.assigned_vehicles.add(primary_vehicle_id)

    def _build_feasible_stack(self, 
                            base_vehicle_id: str, 
                            payload: Payload) -> Optional[List[str]]:
        """
        Attempt to build a feasible stack starting with a base vehicle.
        Like building a tower while making sure each piece fits properly.
        """
        stack = [base_vehicle_id]
        current_vehicle = self.vehicle_map[base_vehicle_id]
        remaining_payload = payload.mass
        
        # Try adding vehicles to the stack until payload requirements are met
        while remaining_payload > current_vehicle.payload_capacity:
            # Find a compatible vehicle to add to the stack
            next_vehicle_id = self._find_compatible_vehicle(
                current_vehicle, remaining_payload)
            
            if not next_vehicle_id:
                return None  # No feasible stack possible
                
            stack.append(next_vehicle_id)
            current_vehicle = self.vehicle_map[next_vehicle_id]
            remaining_payload = (remaining_payload - 
                               current_vehicle.payload_capacity)
            
        return stack if len(stack) > 1 else None

    def _find_compatible_vehicle(self, 
                               lower_vehicle: Vehicle,
                               required_capacity: float) -> Optional[str]:
        """
        Find a vehicle that can be stacked on top of another.
        Like finding the right piece to add to our tower.
        """
        for vehicle_id, vehicle in self.vehicle_map.items():
            if vehicle_id not in self.assigned_vehicles:
                stack_rule = self.stacking_rules.get(
                    (vehicle_id, lower_vehicle.id))
                
                if stack_rule and stack_rule.is_valid_stack(
                    vehicle, lower_vehicle):
                    if vehicle.payload_capacity >= required_capacity:
                        return vehicle_id
        return None

    def mutate(self, mutation_rate: float = 0.1):
        """
        Randomly modify vehicle stacks while maintaining feasibility.
        Like carefully rearranging our tower pieces.
        """
        for payload_id in self.vehicle_stacks:
            if random.random() < mutation_rate:
                # Store original stack in case new one isn't feasible
                original_stack = self.vehicle_stacks[payload_id].copy()
                
                try:
                    # Try to build a new stack
                    primary_vehicle_id = self.payload_assignment_gene.assignments[
                        payload_id]
                    new_stack = self._build_feasible_stack(
                        primary_vehicle_id, 
                        self.payload_map[payload_id]
                    )
                    
                    if new_stack:
                        # Remove vehicles from old stack
                        self.assigned_vehicles.difference_update(
                            set(original_stack))
                        # Add new stack
                        self.vehicle_stacks[payload_id] = new_stack
                        self.assigned_vehicles.update(new_stack)
                        
                except Exception:
                    # Restore original stack if anything goes wrong
                    self.vehicle_stacks[payload_id] = original_stack

    def crossover(self, 
                 other: 'VehicleStackingGene'
                 ) -> Tuple['VehicleStackingGene', 'VehicleStackingGene']:
        """
        Combine stacking strategies from two parents.
        Like mixing and matching successful tower configurations.
        """
        child1 = VehicleStackingGene(
            self.vehicle_map, 
            self.payload_assignment_gene,
            self.stacking_rules
        )
        child2 = VehicleStackingGene(
            self.vehicle_map,
            self.payload_assignment_gene,
            self.stacking_rules
        )
        
        # For each payload, randomly choose parent's stack
        for payload_id in self.vehicle_stacks:
            if random.random() < 0.5:
                child1.vehicle_stacks[payload_id] = self.vehicle_stacks[
                    payload_id].copy()
                child2.vehicle_stacks[payload_id] = other.vehicle_stacks[
                    payload_id].copy()
            else:
                child1.vehicle_stacks[payload_id] = other.vehicle_stacks[
                    payload_id].copy()
                child2.vehicle_stacks[payload_id] = self.vehicle_stacks[
                    payload_id].copy()
        
        # Update assigned vehicles for both children
        child1._update_assigned_vehicles()
        child2._update_assigned_vehicles()
        
        return child1, child2

    def _update_assigned_vehicles(self):
        """
        Update the set of assigned vehicles based on current stacks.
        Like taking inventory of which pieces we're using.
        """
        self.assigned_vehicles.clear()
        for stack in self.vehicle_stacks.values():
            self.assigned_vehicles.update(stack)

    def is_feasible(self) -> bool:
        """
        Check if all vehicle stacks are valid.
        Like verifying our towers are stable and properly built.
        """
        for payload_id, stack in self.vehicle_stacks.items():
            payload = self.payload_map[payload_id]
            
            # Check if stack can handle payload
            total_capacity = 0
            for i in range(len(stack)-1):
                lower_vehicle = self.vehicle_map[stack[i]]
                upper_vehicle = self.vehicle_map[stack[i+1]]
                
                # Check stacking rule exists and is valid
                rule = self.stacking_rules.get((upper_vehicle.id, lower_vehicle.id))
                if not rule or not rule.is_valid_stack(upper_vehicle, lower_vehicle):
                    return False
                
                total_capacity += upper_vehicle.payload_capacity
            
            # Check if stack can accommodate payload
            if total_capacity < payload.mass:
                return False
                
        return True
    
    
@dataclass
class PropellantType:
    """Defines different types of propellant and their characteristics"""
    name: str
    density: float  # kg/m³
    specific_impulse: float  # seconds
    boiloff_rate: float  # % per day, 0 for storable propellants
    oxidizer_ratio: float  # Mass ratio of oxidizer to fuel

@dataclass
class ResourceRequirement:
    """Tracks resource requirements for a mission segment"""
    propellant_mass: float  # Total propellant mass needed
    oxidizer_mass: float  # Mass of oxidizer
    fuel_mass: float  # Mass of fuel
    margin: float  # Safety margin (percentage)
    boiloff_allowance: float  # Additional propellant for boiloff

class PropellantResourceGene:
    """
    Manages propellant and resource allocation across mission arcs.
    Integrates with other genes to ensure feasible resource utilization.
    """
    def __init__(self,
                 vehicle_map: Dict[str, Vehicle],
                 launch_timing_gene: LaunchTimingGene,
                 payload_assignment_gene: PayloadAssignmentGene,
                 routing_gene: RoutingFlowGene,
                 safety_margin: float = 0.1):  # 10% safety margin
        
        self.vehicle_map = vehicle_map
        self.launch_timing_gene = launch_timing_gene
        self.payload_assignment_gene = payload_assignment_gene
        self.routing_gene = routing_gene
        self.safety_margin = safety_margin
        
        # Initialize propellant types
        self.propellant_types = {
            "Storable": PropellantType(
                name="Storable",
                density=1000,
                specific_impulse=320,
                boiloff_rate=0,
                oxidizer_ratio=1.65
            ),
            "Cryogenic": PropellantType(
                name="Cryogenic",
                density=1140,
                specific_impulse=450,
                boiloff_rate=0.0016,  # 0.16% per day
                oxidizer_ratio=5.5
            )
        }
        
        # Maps (payload_id, arc) to resource requirements
        self.resource_allocations: Dict[Tuple[str, RouteArc], ResourceRequirement] = {}
        
        self._initialize_allocations()

    def _initialize_allocations(self):
        """Calculate initial resource requirements for all mission arcs"""
        for payload_id, payload in self.payload_map.items():
            vehicle = self.vehicle_map[
                self.payload_assignment_gene.assignments[payload_id]]
            route = self.routing_gene.routing_plan[payload_id]
            
            # Calculate requirements for each arc in the route
            for arc in route.arcs:
                requirement = self._calculate_resource_requirement(
                    payload, vehicle, arc)
                self.resource_allocations[(payload_id, arc)] = requirement

    def _calculate_resource_requirement(self,
                                     payload: Payload,
                                     vehicle: Vehicle,
                                     arc: RouteArc) -> ResourceRequirement:
        """
        Calculate propellant and resource requirements for a specific arc.
        Uses rocket equation and accounts for boiloff and margins.
        """
        # Get appropriate propellant type based on vehicle Isp
        propellant_type = self._get_propellant_type(vehicle)
        
        # Calculate basic propellant requirement using rocket equation
        total_mass = vehicle.dry_mass + payload.mass
        delta_v = arc.delta_v
        
        propellant_mass = total_mass * (
            math.exp(delta_v / (propellant_type.specific_impulse * 9.81)) - 1)
        
        # Add safety margin
        propellant_mass *= (1 + self.safety_margin)
        
        # Calculate oxidizer and fuel masses
        oxidizer_mass = (propellant_mass * propellant_type.oxidizer_ratio / 
                        (1 + propellant_type.oxidizer_ratio))
        fuel_mass = propellant_mass - oxidizer_mass
        
        # Calculate boiloff allowance
        time_of_flight = arc.time_of_flight
        boiloff_allowance = (oxidizer_mass * 
                           propellant_type.boiloff_rate * 
                           time_of_flight)
        
        return ResourceRequirement(
            propellant_mass=propellant_mass,
            oxidizer_mass=oxidizer_mass,
            fuel_mass=fuel_mass,
            margin=self.safety_margin,
            boiloff_allowance=boiloff_allowance
        )

    def _get_propellant_type(self, vehicle: Vehicle) -> PropellantType:
        """Determine propellant type based on vehicle characteristics"""
        if vehicle.specific_impulse >= 400:
            return self.propellant_types["Cryogenic"]
        return self.propellant_types["Storable"]

    def is_feasible(self) -> bool:
        """Check if current resource allocation is feasible"""
        for (payload_id, arc), requirement in self.resource_allocations.items():
            vehicle = self.vehicle_map[
                self.payload_assignment_gene.assignments[payload_id]]
            
            # Check against vehicle propellant capacity
            total_required = (requirement.propellant_mass + 
                            requirement.boiloff_allowance)
            if total_required > vehicle.propellant_capacity:
                return False
            
            # Check if propellant type matches vehicle capabilities
            if (requirement.oxidizer_mass > 0 and 
                    not vehicle.can_use_cryogenic_propellant):
                return False
                
        return True

    def mutate(self, mutation_rate: float = 0.1):
        """
        Adjust resource allocations within feasible bounds.
        Primarily affects safety margins and boiloff allowances.
        """
        if random.random() < mutation_rate:
            # Store original state
            original_allocations = copy.deepcopy(self.resource_allocations)
            
            try:
                # Randomly adjust safety margins for some allocations
                for key, requirement in self.resource_allocations.items():
                    if random.random() < mutation_rate:
                        new_margin = max(0.05, 
                                       min(0.20, 
                                           requirement.margin + 
                                           random.uniform(-0.02, 0.02)))
                        
                        # Recalculate with new margin
                        payload_id, arc = key
                        payload = self.payload_map[payload_id]
                        vehicle = self.vehicle_map[
                            self.payload_assignment_gene.assignments[payload_id]]
                        
                        self.resource_allocations[key] = \
                            self._calculate_resource_requirement(
                                payload, vehicle, arc)
                
                # Verify feasibility
                if not self.is_feasible():
                    self.resource_allocations = original_allocations
                    
            except Exception:
                self.resource_allocations = original_allocations

    def crossover(self, 
                 other: 'PropellantResourceGene'
                 ) -> Tuple['PropellantResourceGene', 'PropellantResourceGene']:
        """
        Perform crossover by exchanging resource allocation strategies
        """
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        # Exchange safety margins between parents
        if random.random() < 0.5:
            child1.safety_margin, child2.safety_margin = \
                child2.safety_margin, child1.safety_margin
        
        # Recalculate allocations with new margins
        child1._initialize_allocations()
        child2._initialize_allocations()
        
        return child1, child2

    def get_total_propellant_mass(self) -> float:
        """Calculate total propellant mass required for the mission"""
        return sum(req.propellant_mass + req.boiloff_allowance 
                  for req in self.resource_allocations.values())

    def get_arc_requirement(self, 
                          payload_id: str, 
                          arc: RouteArc) -> Optional[ResourceRequirement]:
        """Get resource requirements for a specific arc"""
        return self.resource_allocations.get((payload_id, arc))

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import random

class ConstraintType(Enum):
    """
    Defines different types of relationships between payloads.
    """
    CO_PAYLOAD = "co_payload"         # Must launch together
    PRECURSOR = "precursor"           # Must launch before or with
    STRICT_PREDECESSOR = "strict_predecessor"  # Must launch strictly before
    SUCCESSOR = "successor"           # Must launch after or with
    STRICT_SUCCESSOR = "strict_successor"      # Must launch strictly after

@dataclass
class PayloadConstraint:
    """
    Represents a specific constraint between payloads.
    Includes validation logic and timing requirements.
    """
    constraint_type: ConstraintType
    related_payload_ids: List[str]
    minimum_time_gap: Optional[int] = None  # For strict constraints
    maximum_time_gap: Optional[int] = None  # Optional upper bound

    def validate_timing(self, 
                       primary_time: int, 
                       related_times: List[int]) -> bool:
        """
        Validates if the timing between payloads satisfies this constraint.
        """
        if self.constraint_type == ConstraintType.CO_PAYLOAD:
            # All payloads must launch at the same time
            return all(t == primary_time for t in related_times)
            
        elif self.constraint_type == ConstraintType.PRECURSOR:
            # All related payloads must launch before or with primary
            return all(t <= primary_time for t in related_times)
            
        elif self.constraint_type == ConstraintType.STRICT_PREDECESSOR:
            # Must launch strictly before with minimum gap
            return all(t + (self.minimum_time_gap or 1) <= primary_time 
                      for t in related_times)
            
        elif self.constraint_type == ConstraintType.SUCCESSOR:
            # All related payloads must launch after or with primary
            return all(t >= primary_time for t in related_times)
            
        elif self.constraint_type == ConstraintType.STRICT_SUCCESSOR:
            # Must launch strictly after with minimum gap
            return all(t >= primary_time + (self.minimum_time_gap or 1) 
                      for t in related_times)
            
        return False

class PayloadConstraintsGene:
    """
    Manages constraints between payloads and ensures their relationships
    are maintained throughout the mission planning process.
    """
    def __init__(self,
                 payload_map: Dict[str, Payload],
                 launch_timing_gene: LaunchTimingGene):
        
        self.payload_map = payload_map
        self.launch_timing_gene = launch_timing_gene
        
        # Dictionary mapping payload IDs to their constraints
        self.constraints: Dict[str, List[PayloadConstraint]] = {}
        
        # Initialize constraint relationships
        self._initialize_constraints()
        
        # Build dependency graph for validation
        self.dependency_graph = self._build_dependency_graph()

    def _initialize_constraints(self):
        """
        Set up initial constraints based on payload requirements.
        This creates a complete constraint network between payloads.
        """
        for payload_id, payload in self.payload_map.items():
            self.constraints[payload_id] = []
            
            # Add co-payload constraints
            if payload.co_payloads:
                self.constraints[payload_id].append(
                    PayloadConstraint(
                        constraint_type=ConstraintType.CO_PAYLOAD,
                        related_payload_ids=payload.co_payloads
                    )
                )
            
            # Add precursor constraints
            if payload.precursor_payloads:
                self.constraints[payload_id].append(
                    PayloadConstraint(
                        constraint_type=ConstraintType.PRECURSOR,
                        related_payload_ids=payload.precursor_payloads
                    )
                )
            
            # Add any additional sequential requirements
            # These might come from mission-specific rules
            self._add_mission_specific_constraints(payload_id)

    def _add_mission_specific_constraints(self, payload_id: str):
        """
        Add constraints specific to mission requirements.
        This can be customized based on mission needs.
        """
        payload = self.payload_map[payload_id]
        
        # Example: If this is a crew payload, ensure life support 
        # is launched first
        if payload.payload_type == "CREW":
            self.constraints[payload_id].append(
                PayloadConstraint(
                    constraint_type=ConstraintType.STRICT_PREDECESSOR,
                    related_payload_ids=["LIFE_SUPPORT"],
                    minimum_time_gap=2  # Ensure life support is ready
                )
            )

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Creates a graph of payload dependencies to detect cycles
        and validate sequencing.
        """
        graph = {pid: set() for pid in self.payload_map}
        
        for payload_id, constraints in self.constraints.items():
            for constraint in constraints:
                if constraint.constraint_type in [
                    ConstraintType.PRECURSOR,
                    ConstraintType.STRICT_PREDECESSOR
                ]:
                    # Add edges for dependencies
                    for related_id in constraint.related_payload_ids:
                        graph[payload_id].add(related_id)
        
        return graph

    def validate_launch_times(self) -> bool:
        """
        Check if current launch times satisfy all constraints.
        """
        # First check for cyclic dependencies
        if self._has_cycle():
            return False
            
        # Then validate all constraints
        for payload_id, constraints in self.constraints.items():
            primary_time = self.launch_timing_gene.launch_times[payload_id]
            
            for constraint in constraints:
                related_times = [
                    self.launch_timing_gene.launch_times[rid]
                    for rid in constraint.related_payload_ids
                ]
                
                if not constraint.validate_timing(primary_time, related_times):
                    return False
        
        return True

    def _has_cycle(self) -> bool:
        """
        Detect cyclic dependencies using depth-first search.
        """
        visited = set()
        recursion_stack = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            recursion_stack.add(node)
            
            for neighbor in self.dependency_graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(node)
            return False
        
        for node in self.dependency_graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def repair_launch_times(self):
        """
        Attempt to fix launch times that violate constraints.
        Uses a topological sort to assign valid times.
        """
        if not self._has_cycle():  # Only proceed if no cycles
            ordered_payloads = self._topological_sort()
            
            # Assign times in dependency order
            for payload_id in ordered_payloads:
                constraints = self.constraints.get(payload_id, [])
                
                # Find valid launch time
                valid_time = self._find_valid_launch_time(
                    payload_id, constraints)
                
                if valid_time is not None:
                    self.launch_timing_gene.launch_times[payload_id] = valid_time

    def _topological_sort(self) -> List[str]:
        """
        Sort payloads based on their dependencies.
        """
        visited = set()
        ordered = []
        
        def visit(node: str):
            if node not in visited:
                visited.add(node)
                for neighbor in self.dependency_graph[node]:
                    visit(neighbor)
                ordered.append(node)
        
        for node in self.dependency_graph:
            if node not in visited:
                visit(node)
                
        return ordered[::-1]  # Reverse to get correct order

    def _find_valid_launch_time(self,
                              payload_id: str,
                              constraints: List[PayloadConstraint]) -> Optional[int]:
        """
        Find a launch time that satisfies all constraints for a payload.
        """
        payload = self.payload_map[payload_id]
        
        # Try each possible time in the launch window
        for time in range(payload.required_launch_window.start,
                         payload.required_launch_window.end + 1):
            
            valid = True
            for constraint in constraints:
                related_times = [
                    self.launch_timing_gene.launch_times[rid]
                    for rid in constraint.related_payload_ids
                ]
                
                if not constraint.validate_timing(time, related_times):
                    valid = False
                    break
            
            if valid:
                return time
                
        return None
    
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import datetime
from enum import Enum
import random

class LaunchSite(Enum):
    """Defines possible launch sites and their constraints"""
    KENNEDY = "Kennedy Space Center"
    VANDENBERG = "Vandenberg Space Force Base"
    BOCA_CHICA = "Boca Chica"

@dataclass
class LaunchWindow:
    """Represents a specific launch opportunity"""
    start_time: int  # Time step when window opens
    end_time: int    # Time step when window closes
    vehicle_id: str
    launch_site: LaunchSite
    
    def overlaps_with(self, other: 'LaunchWindow') -> bool:
        """Check if this launch window overlaps with another"""
        return not (self.end_time < other.start_time or 
                   self.start_time > other.end_time)

class LaunchFrequencyGene:
    """
    Manages launch timing constraints for vehicles, ensuring minimum separation
    between launches and respecting site-specific constraints.
    """
    def __init__(self,
                 vehicle_map: Dict[str, Vehicle],
                 launch_timing_gene: LaunchTimingGene,
                 payload_assignment_gene: PayloadAssignmentGene,
                 min_pad_turnaround: int = 3):  # Minimum days between launches
        
        self.vehicle_map = vehicle_map
        self.launch_timing_gene = launch_timing_gene
        self.payload_assignment_gene = payload_assignment_gene
        self.min_pad_turnaround = min_pad_turnaround
        
        # Track next available launch time for each vehicle
        self.next_available_launch = {
            vehicle_id: vehicle.availability_window.start
            for vehicle_id, vehicle in vehicle_map.items()
        }
        
        # Track scheduled launches for each vehicle
        self.scheduled_launches: Dict[str, List[LaunchWindow]] = {
            vehicle_id: [] for vehicle_id in vehicle_map
        }
        
        # Map vehicles to their primary launch sites
        self.vehicle_launch_sites = self._assign_launch_sites()
        
        # Initialize launch schedule based on current assignments
        self._initialize_launch_schedule()

    def _assign_launch_sites(self) -> Dict[str, LaunchSite]:
        """
        Assign primary launch sites to vehicles based on their characteristics.
        This could be expanded based on real-world constraints.
        """
        sites = {}
        for vehicle_id, vehicle in self.vehicle_map.items():
            # Example logic for assigning launch sites
            if vehicle.payload_capacity > 5000:  # Heavy lift vehicle
                sites[vehicle_id] = LaunchSite.KENNEDY
            elif "polar" in vehicle.operational_domains:
                sites[vehicle_id] = LaunchSite.VANDENBERG
            else:
                sites[vehicle_id] = LaunchSite.BOCA_CHICA
        return sites

    def _initialize_launch_schedule(self):
        """
        Create initial launch schedule based on current payload assignments
        and launch times.
        """
        self.scheduled_launches.clear()
        for vehicle_id in self.vehicle_map:
            self.scheduled_launches[vehicle_id] = []

        # Process all payloads to create launch windows
        for payload_id, payload in self.payload_map.items():
            vehicle_id = self.payload_assignment_gene.assignments[payload_id]
            launch_time = self.launch_timing_gene.launch_times[payload_id]
            
            launch_window = LaunchWindow(
                start_time=launch_time,
                end_time=launch_time + 1,  # Assuming 1 time step for launch
                vehicle_id=vehicle_id,
                launch_site=self.vehicle_launch_sites[vehicle_id]
            )
            
            self.scheduled_launches[vehicle_id].append(launch_window)
            
        # Sort launch windows chronologically
        for launches in self.scheduled_launches.values():
            launches.sort(key=lambda x: x.start_time)

    def is_launch_feasible(self, 
                          vehicle_id: str, 
                          proposed_time: int) -> bool:
        """
        Check if a proposed launch time is feasible given vehicle constraints
        and existing schedule.
        """
        vehicle = self.vehicle_map[vehicle_id]
        
        # Check if within vehicle's availability window
        if not vehicle.availability_window.contains(proposed_time):
            return False
            
        # Check minimum separation from other launches of same vehicle
        for window in self.scheduled_launches[vehicle_id]:
            if abs(proposed_time - window.start_time) < vehicle.launch_frequency:
                return False
        
        # Check launch site constraints
        proposed_window = LaunchWindow(
            start_time=proposed_time,
            end_time=proposed_time + 1,
            vehicle_id=vehicle_id,
            launch_site=self.vehicle_launch_sites[vehicle_id]
        )
        
        # Check for conflicts with other vehicles at same launch site
        for other_vehicle_id, launches in self.scheduled_launches.items():
            if (self.vehicle_launch_sites[other_vehicle_id] == 
                    self.vehicle_launch_sites[vehicle_id]):
                for window in launches:
                    if (window.overlaps_with(proposed_window) and 
                            abs(proposed_time - window.start_time) < 
                            self.min_pad_turnaround):
                        return False
        
        return True

    def get_next_available_time(self, 
                              vehicle_id: str, 
                              after_time: int) -> Optional[int]:
        """
        Find the next feasible launch time for a vehicle after a given time.
        """
        vehicle = self.vehicle_map[vehicle_id]
        current_time = max(after_time, self.next_available_launch[vehicle_id])
        
        while current_time <= vehicle.availability_window.end:
            if self.is_launch_feasible(vehicle_id, current_time):
                return current_time
            current_time += 1
            
        return None

    def update_launch_schedule(self, 
                             payload_id: str, 
                             new_time: int) -> bool:
        """
        Update the launch schedule when a payload's launch time changes.
        Returns True if update is successful.
        """
        vehicle_id = self.payload_assignment_gene.assignments[payload_id]
        
        # Check if new time is feasible
        if not self.is_launch_feasible(vehicle_id, new_time):
            return False
            
        # Remove old launch window if it exists
        self.scheduled_launches[vehicle_id] = [
            window for window in self.scheduled_launches[vehicle_id]
            if window.vehicle_id != payload_id
        ]
        
        # Add new launch window
        new_window = LaunchWindow(
            start_time=new_time,
            end_time=new_time + 1,
            vehicle_id=vehicle_id,
            launch_site=self.vehicle_launch_sites[vehicle_id]
        )
        
        self.scheduled_launches[vehicle_id].append(new_window)
        self.scheduled_launches[vehicle_id].sort(key=lambda x: x.start_time)
        
        # Update next available launch time
        self.next_available_launch[vehicle_id] = max(
            self.next_available_launch[vehicle_id],
            new_time + vehicle.launch_frequency
        )
        
        return True

    def repair_schedule(self) -> bool:
        """
        Attempt to repair launch schedule violations by adjusting launch times.
        Returns True if successful.
        """
        violations = self._find_schedule_violations()
        if not violations:
            return True
            
        for payload_id, current_time in violations:
            vehicle_id = self.payload_assignment_gene.assignments[payload_id]
            new_time = self.get_next_available_time(
                vehicle_id, current_time)
            
            if new_time is None:
                return False  # Couldn't find valid time
                
            self.launch_timing_gene.launch_times[payload_id] = new_time
            if not self.update_launch_schedule(payload_id, new_time):
                return False
                
        return True

    def _find_schedule_violations(self) -> List[Tuple[str, int]]:
        """
        Find all current violations of launch frequency constraints.
        Returns list of (payload_id, current_time) pairs.
        """
        violations = []
        for payload_id, launch_time in self.launch_timing_gene.launch_times.items():
            vehicle_id = self.payload_assignment_gene.assignments[payload_id]
            if not self.is_launch_feasible(vehicle_id, launch_time):
                violations.append((payload_id, launch_time))
        return violations