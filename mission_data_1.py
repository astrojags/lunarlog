from typing import Dict, List, Tuple, Set, Any
from enum import Enum
from population import TimeWindow, Payload, Vehicle, RouteArc, Route, NodeType, StackingRule, PropellantType

# 1. TimeWindows
# These will be used within other definitions, but here's the structure
early_window = TimeWindow(start=1, end=3)    # Months 1-3
mid_window = TimeWindow(start=4, end=6)      # Months 4-6
late_window = TimeWindow(start=7, end=12)    # Months 7-12
full_window = TimeWindow(start=1, end=12)    # Full year

# 2. Payloads
example_payloads: Dict[str, Payload] = {
    "LEO_SAT": Payload(
        id="LEO_SAT",
        mass=1500.0,  # 1.5 metric tons
        volume=8.0,   # 8 cubic meters
        required_launch_window=early_window,
        precursor_payloads=[],
        co_payloads=[],
        destination=NodeType.LEO  # Explicitly specify LEO destination
    ),
    "CARGO1": Payload(
        id="CARGO1",
        mass=2000.0,  # 2 metric tons
        volume=10.0,  # 10 cubic meters
        required_launch_window=early_window,
        precursor_payloads=[],
        co_payloads=[]
    ),
    "HABITAT": Payload(
        id="HABITAT",
        mass=5000.0,
        volume=30.0,
        required_launch_window=mid_window,
        precursor_payloads=["CARGO1"],  # Cargo must arrive first
        co_payloads=[]
    ),
    "CREW": Payload(
        id="CREW",
        mass=1000.0,
        volume=15.0,
        required_launch_window=late_window,
        precursor_payloads=["HABITAT"],  # Habitat must be ready
        co_payloads=["SUPPLIES"]  # Must launch with supplies
    ),
    "SUPPLIES": Payload(
        id="SUPPLIES",
        mass=3000.0,
        volume=20.0,
        required_launch_window=late_window,
        precursor_payloads=["HABITAT"],
        co_payloads=["CREW"]  # Must launch with crew
    )
}

# 3. Vehicles
example_vehicles: Dict[str, Vehicle] = {
    "HEAVY1": Vehicle(
        id="HEAVY1",
        payload_capacity=10000.0,  # 10 metric tons
        volume_capacity=50.0,      # 50 cubic meters
        specific_impulse=450.0,    # 450s Isp
        availability_window=full_window,
        launch_frequency=2,        # Minimum 2 months between launches
        operational_domains=["Earth", "LEO", "LLO"]
    ),
    "MEDIUM1": Vehicle(
        id="MEDIUM1",
        payload_capacity=5000.0,
        volume_capacity=30.0,
        specific_impulse=380.0,
        availability_window=full_window,
        launch_frequency=1,
        operational_domains=["Earth", "LEO"]
    ),
    "TANKER1": Vehicle(
        id="TANKER1",
        payload_capacity=8000.0,
        volume_capacity=40.0,
        specific_impulse=420.0,
        availability_window=full_window,
        launch_frequency=2,
        operational_domains=["Earth", "LEO", "LLO"]
    )
}

# 4. Routes and Nodes
# Adding specific LEO orbit parameters
leo_orbit_params = {
    "altitude": 400,  # km
    "inclination": 51.6,  # degrees (ISS-like orbit)
    "nodal_precession": 5.0  # degrees per day
}

example_route_arcs: List[RouteArc] = [
    RouteArc(
        origin=NodeType.EARTH,
        destination=NodeType.LEO,
        delta_v=9000.0,        # 9 km/s to LEO
        time_of_flight=0.125,  # 3 hours
        allowed_vehicles={"HEAVY1", "MEDIUM1", "TANKER1"}
    ),
    RouteArc(
        origin=NodeType.LEO,
        destination=NodeType.LLO,
        delta_v=4000.0,        # 4 km/s LEO to LLO
        time_of_flight=3.0,    # 3 days
        allowed_vehicles={"HEAVY1", "TANKER1"}
    ),
    RouteArc(
        origin=NodeType.LLO,
        destination=NodeType.LUNAR_SURFACE,
        delta_v=2000.0,        # 2 km/s LLO to surface
        time_of_flight=0.125,  # 3 hours
        allowed_vehicles={"HEAVY1"}
    )
]

# Create valid routes dictionary
# Adding specific LEO route variants for satellite deployment
example_valid_routes: Dict[Tuple[NodeType, NodeType], List[Route]] = {
    (NodeType.EARTH, NodeType.LEO): [Route(arcs=[example_route_arcs[0]])],
    (NodeType.EARTH, NodeType.LLO): [Route(arcs=[example_route_arcs[0], example_route_arcs[1]])],
    (NodeType.LEO, NodeType.LLO): [Route(arcs=[example_route_arcs[1]])],
    (NodeType.LLO, NodeType.LUNAR_SURFACE): [Route(arcs=[example_route_arcs[2]])]
}

# 5. Stacking Rules
example_stacking_rules: Dict[Tuple[str, str], StackingRule] = {
    ("MEDIUM1", "HEAVY1"): StackingRule(
        upper_vehicle="MEDIUM1",
        lower_vehicle="HEAVY1",
        max_payload_fraction=0.8,  # Can use 80% of lower vehicle capacity
        interface_mass=500.0       # 500 kg interface mass
    ),
    ("TANKER1", "HEAVY1"): StackingRule(
        upper_vehicle="TANKER1",
        lower_vehicle="HEAVY1",
        max_payload_fraction=0.9,
        interface_mass=600.0
    )
}

# 6. Launch Sites
example_launch_sites: Dict[str, Dict[str, Any]] = {
    "KENNEDY": {
        "latitude": 28.5,
        "available_vehicles": {"HEAVY1", "MEDIUM1", "TANKER1"},
        "min_inclination": 28.5,
        "max_launches_per_month": 4
    },
    "VANDENBERG": {
        "latitude": 34.4,
        "available_vehicles": {"MEDIUM1"},
        "min_inclination": 34.4,
        "max_launches_per_month": 2
    }
}

# 7. Propellant Types
example_propellants: Dict[str, PropellantType] = {
    "LOX_LH2": PropellantType(
        name="LOX_LH2",
        density=1140.0,
        specific_impulse=450.0,
        boiloff_rate=0.0016,
        oxidizer_ratio=5.5
    ),
    "NTO_MMH": PropellantType(
        name="NTO_MMH",
        density=1450.0,
        specific_impulse=320.0,
        boiloff_rate=0.0,
        oxidizer_ratio=1.6
    )
}

# 8. Campaign Parameters
# Adding LEO-specific constraints
leo_deployment_constraints = {
    "min_separation": 2,  # Minimum days between LEO deployments
    "lighting_conditions": "daylight",  # Deploy only in daylight
    "collision_avoidance": True  # Enable collision avoidance checks
}

campaign_parameters = {
    "max_campaign_duration": 12,    # 12 months
    "min_pad_turnaround": 3,       # 3 days between launches
    "propellant_margin": 0.1,      # 10% propellant margin
    "max_vehicles_per_type": {
        "HEAVY1": 2,
        "MEDIUM1": 3,
        "TANKER1": 2
    }
}