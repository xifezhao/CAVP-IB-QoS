import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import heapq  # For priority queue in Dijkstra
import json  # For saving configuration

# --- Set random seed ---
RANDOM_SEED = 42  # Choose any integer value
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Constant definitions (parameters) ---
NUM_RUNS = 5  # Reduced for demonstration and faster execution
NUM_STEPS = 20
ALGORITHM_NAMES = ['CAVP-IB-QoS', 'WFQ', 'PriorityQueueing', 'StaticAllocation']

# Network topology will be generated, so no need for default values here

BANDWIDTH_REQ_RANGE = (10, 40)
BW_CHANGE_PROB = 0.2
LATENCY_SENSITIVITIES = ['High', 'Medium', 'Low']
FLOW_PRIORITIES = ['High', 'Medium', 'Low']

# --- Network Model ---
class Network:
    def __init__(self, adjacency_list):
        """
        Initializes the network topology.

        Args:
            adjacency_list (dict): The topology represented as an adjacency list.
                Keys: Node (str)
                Values: Dictionary, Keys: Neighboring nodes (str), Values: Link properties (dict, contains 'capacity' and 'latency')
                Example:
                {
                    'A': {'B': {'capacity': 100, 'latency': 10}, 'C': {'capacity': 50, 'latency': 20}},
                    'B': {'A': {'capacity': 100, 'latency': 10}, 'D': {'capacity': 75, 'latency': 15}},
                    'C': {'A': {'capacity': 50, 'latency': 20}, 'D': {'capacity': 100, 'latency': 25}},
                    'D': {'B': {'capacity': 75, 'latency': 15}, 'C': {'capacity': 100, 'latency': 25}}
                }
        """
        self.adjacency_list = adjacency_list
        self.flows = {}  # flow_id: {'path': [node1, node2, ...], 'bandwidth': ...}

    def get_link_utilization(self, link):
        """Calculates the utilization of a given link."""
        total_bandwidth = 0
        for flow_id, flow_data in self.flows.items():
            try:
                link_index = flow_data['path'].index(link[0])  # Find index of first node in the link
                if flow_data['path'][link_index+1] == link[1]:
                    total_bandwidth += flow_data['bandwidth']  # Add flow bandwidth if link exists
            except (ValueError, IndexError):  # Handle cases where link is not in path, or at end
                continue
        return total_bandwidth / self.adjacency_list[link[0]][link[1]]['capacity']

    def get_link_cost(self, link, utilization_exponent=2):
        """Calculates the cost of a link."""
        utilization = self.get_link_utilization(link)
        base_latency = self.adjacency_list[link[0]][link[1]]['latency']
        cost = base_latency * (1 + utilization) ** utilization_exponent
        return cost

    def get_neighbors(self, node):
        """Returns the neighbors of a given node."""
        return self.adjacency_list.get(node, {}).keys()

# --- Flow Representation ---
class Flow:
    """Flow class, representing a flow in the network."""
    def __init__(self, flow_id, source, destination, bandwidth_req,
                 latency_sensitivity, priority, burstiness=0.0):
        """
        Initializes the flow.

        Args:
            flow_id (str): Flow ID.
            source (str): Source node.
            destination (str): Destination node.
            bandwidth_req (int): Bandwidth requirement.
            latency_sensitivity (str): Latency sensitivity ('High', 'Medium', 'Low').
            priority (str): Priority ('High', 'Medium', 'Low').
            burstiness (float): Burstiness (0.0 - 1.0).
        """
        self.flow_id = flow_id
        self.source = source
        self.destination = destination
        self.bandwidth_req = bandwidth_req
        self.latency_sensitivity = latency_sensitivity
        self.priority = priority
        self.path = []
        self.allocated_bandwidth = 0
        self.burstiness = burstiness  # New attribute
        self.current_bandwidth_req = bandwidth_req  # Current, potentially bursty, demand

    def update_bandwidth_req(self):
        """
        Updates the current bandwidth requirement based on burstiness.
        """
        if random.random() < self.burstiness:
            # Generate a bursty demand (e.g., increase to 2-5 times the original demand)
            self.current_bandwidth_req = self.bandwidth_req * random.uniform(2, 5)
        else:
            # Return to normal demand (possibly with small random fluctuations)
            self.current_bandwidth_req = self.bandwidth_req * random.uniform(0.9, 1.1)
    def __str__(self):  # for easier printing
      return (f"Flow(id={self.flow_id}, src={self.source}, dst={self.destination}, "
              f"bw_req={self.bandwidth_req}, latency_sens={self.latency_sensitivity}, "
              f"priority={self.priority}, burstiness={self.burstiness}, "
              f"current_bw_req={self.current_bandwidth_req})")

# --- Intent Representation ---
class Intent:
    """Intent class, representing the user's QoS requirements for the flow."""
    def __init__(self, flow_id, min_bandwidth, max_latency=None, jitter=None,
                 packet_loss_rate=None, bandwidth_guaranteed=False,
                 latency_guaranteed=False, reliability='Medium',
                 sla_tier=None, application_type=None, user_group=None, priority=2):
        """Initializes the intent."""
        self.flow_id = flow_id
        self.min_bandwidth = min_bandwidth
        self.max_latency = max_latency
        self.jitter = jitter
        self.packet_loss_rate = packet_loss_rate
        self.bandwidth_guaranteed = bandwidth_guaranteed
        self.latency_guaranteed = latency_guaranteed
        self.reliability = reliability
        self.sla_tier = sla_tier
        self.application_type = application_type
        self.user_group = user_group
        self.priority = priority  # Now an integer

    def __str__(self):
        return (f"Intent(flow_id={self.flow_id}, min_bandwidth={self.min_bandwidth}, "
                f"max_latency={self.max_latency}, jitter={self.jitter}, "
                f"packet_loss_rate={self.packet_loss_rate}, "
                f"bandwidth_guaranteed={self.bandwidth_guaranteed}, "
                f"latency_guaranteed={self.latency_guaranteed}, "
                f"reliability={self.reliability}, sla_tier={self.sla_tier}, "
                f"application_type={self.application_type}, user_group={self.user_group}, "
                f"priority={self.priority})")

# --- QoS Algorithm Implementations ---

class CAVP_IB_QoS:
    def __init__(self, network):
        self.network = network

    def generate_policy(self, flows, intents):
        """Generates the QoS policy (including routing and bandwidth allocation)."""
        for flow in flows:
            flow.path = self._dijkstra(flow.source, flow.destination)  # Calculate route
        self._bandwidth_allocation(flows, intents)  # Bandwidth allocation

    def _dijkstra(self, source, destination):
        """Dijkstra's algorithm implementation."""
        distances = {node: float('inf') for node in self.network.adjacency_list}
        previous_nodes = {node: None for node in self.network.adjacency_list}
        distances[source] = 0
        queue = [(0, source)]  # (distance, node)

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_distance > distances[current_node]:
                continue

            if current_node == destination:
                path = []
                while previous_nodes[current_node] is not None:
                    path.insert(0, current_node)
                    current_node = previous_nodes[current_node]
                path.insert(0, source)
                return path

            for neighbor, link_data in self.network.adjacency_list[current_node].items():
                link = (current_node, neighbor)
                cost = self.network.get_link_cost(link)  # Use dynamic link cost
                distance = current_distance + cost
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))
        return []  # No path found

    def _bandwidth_allocation(self, flows, intents):
      """Bandwidth allocation based on dynamic priority."""
      self.network.flows = {}  # Clear previous allocations
      # Sort flows based on dynamic priority
      sorted_flows = sorted(flows, key=lambda f: self._calculate_dynamic_priority(f, intents), reverse=True)

      for flow in sorted_flows:
          intent = self._get_intent(flow.flow_id, intents)
          if intent is None or not flow.path:  # Check flow.path
              continue

          allocated_bandwidth = 0  # Initialize allocated bandwidth
          for i in range(len(flow.path) - 1):
              link = (flow.path[i], flow.path[i+1])
              available_bw = self.network.adjacency_list[link[0]][link[1]]['capacity'] - self.network.get_link_utilization(link)
              alloc_bw = min(available_bw, intent.min_bandwidth - allocated_bandwidth)  # Use remaining required bandwidth

              if alloc_bw > 0:
                  # Update allocated bandwidth for the flow and the network
                  allocated_bandwidth += alloc_bw  # Accumulate allocated bandwidth
                  if flow.flow_id not in self.network.flows:
                      self.network.flows[flow.flow_id] = {'path': flow.path, 'bandwidth': 0}
                  self.network.flows[flow.flow_id]['bandwidth'] += alloc_bw
          flow.allocated_bandwidth = allocated_bandwidth  # Set total allocated bandwidth for the flow

    def _get_intent(self, flow_id, intents):
        """
        Gets the corresponding intent based on flow_id.

        Args:
            flow_id (str): Flow ID.
            intents (list): List of Intent objects.

        Returns:
            Intent: The corresponding Intent object, or None if not found.
        """
        for intent in intents:
            if intent.flow_id == flow_id:
                return intent
        return None

    def _get_intent_priority(self, intent):
      """Returns the priority value based on the intent."""
      return intent.priority


    def _calculate_dynamic_priority(self, flow, intents):
        """Calculates the dynamic priority of the flow."""
        intent = self._get_intent(flow.flow_id, intents)
        if intent is None:
            return 0

        priority = self._get_intent_priority(intent)

        # Dynamic adjustments based on network conditions
        if flow.path:  # Check if flow.path is not empty
            # Consider the *first hop* utilization for priority adjustment.  This
            # is a simplification; you could consider the entire path.
            first_hop_link = (flow.path[0], flow.path[1])
            if self.network.get_link_utilization(first_hop_link) > 0.8:  # High utilization on first hop
                if intent.priority > 2:  # Assuming priority is now an integer
                    priority += 1  # Boost high-priority flows
                else:
                    priority -= 1  # Reduce other flows
            if flow.latency_sensitivity == 'High' and intent.max_latency is not None:
                # Calculate *estimated* path latency.  This is an approximation
                # since we don't know the final delays yet.
                estimated_latency = 0
                for i in range(len(flow.path) -1):
                    estimated_latency += self.network.adjacency_list[flow.path[i]][flow.path[i+1]]['latency']
                if estimated_latency > intent.max_latency:
                    priority += 2
        # else:  # Optionally, handle the case where flow.path is empty
        #     priority = 0  # Or some other default value

        return priority

    def adapt_policy(self, flows, intents):
        """
        Adaptive policy adjustment (simplified: regenerate policy).

        Args:
            flows (list): List of Flow objects.
            intents (list): List of Intent objects.
        """
        self.generate_policy(flows, intents)

class WFQ(CAVP_IB_QoS):  # Inherit from CAVP_IB_QoS for routing
    def __init__(self, network):
        super().__init__(network)

    def _bandwidth_allocation(self, flows, intents):
        """Bandwidth allocation strategy based on WFQ."""
        self.network.flows = {}

        for node in self.network.adjacency_list:
            for neighbor, link_data in self.network.adjacency_list[node].items():
                link = (node, neighbor)
                flows_on_link = [flow for flow in flows if flow.path and link[0] in flow.path and link[1] in flow.path and flow.path.index(link[0])+1 == flow.path.index(link[1])]
                total_weight = sum(self._get_flow_weight(flow) for flow in flows_on_link)

                if total_weight > 0:
                    for flow in flows_on_link:
                        flow_weight = self._get_flow_weight(flow)
                        allocated_bandwidth = (flow_weight / total_weight) * link_data['capacity']
                        flow.allocated_bandwidth = min(allocated_bandwidth, flow.current_bandwidth_req)
                        if flow.flow_id not in self.network.flows and flow.allocated_bandwidth > 0 and flow.path:
                            self.network.flows[flow.flow_id] = {'path': flow.path, 'bandwidth': flow.allocated_bandwidth}
                else:
                    for flow in flows_on_link:
                        flow.allocated_bandwidth = 0

    def _get_flow_weight(self, flow):
        """Gets the weight based on flow priority."""
        if flow.priority == 'High':
            return 3
        elif flow.priority == 'Medium':
            return 2
        else:
            return 1

class PriorityQueueing(CAVP_IB_QoS):  # Inherit from CAVP_IB_QoS for routing
    def __init__(self, network):
        super().__init__(network)


    def _bandwidth_allocation(self, flows, intents):
        """Bandwidth allocation strategy based on priority queuing."""
        self.network.flows = {}
        priority_levels = {'High': 3, 'Medium': 2, 'Low': 1}
        sorted_flows = sorted(flows, key=lambda f: priority_levels[f.priority], reverse=True)

        for node in self.network.adjacency_list:
            for neighbor, link_data in self.network.adjacency_list[node].items():
                link = (node, neighbor)
                available_capacity = link_data['capacity']
                for flow in sorted_flows:
                    if flow.path and link[0] in flow.path and link[1] in flow.path and flow.path.index(link[0])+1 == flow.path.index(link[1]):
                        required_bandwidth = flow.current_bandwidth_req
                        allocated_bandwidth = min(required_bandwidth, available_capacity)
                        flow.allocated_bandwidth = allocated_bandwidth
                        available_capacity -= allocated_bandwidth
                        if flow.flow_id not in self.network.flows and flow.allocated_bandwidth > 0 and flow.path:
                            self.network.flows[flow.flow_id] = {'path': flow.path, 'bandwidth': flow.allocated_bandwidth}
                        if available_capacity <= 0:
                            break

# --- Data Generation Functions ---

def generate_network_topology():
    """Generates a sample network topology (adjacency list)."""
    topology = {
        'A': {'B': {'capacity': 100, 'latency': 10}, 'C': {'capacity': 80, 'latency': 15}},
        'B': {'A': {'capacity': 100, 'latency': 10}, 'D': {'capacity': 90, 'latency': 12}, 'E': {'capacity': 110, 'latency': 18}},
        'C': {'A': {'capacity': 80, 'latency': 15}, 'F': {'capacity': 120, 'latency': 20}},
        'D': {'B': {'capacity': 90, 'latency': 12}, 'G': {'capacity': 100, 'latency': 10}},
        'E': {'B': {'capacity': 110, 'latency': 18}, 'F': {'capacity': 95, 'latency': 14}, 'H': {'capacity': 105, 'latency': 16}},
        'F': {'C': {'capacity': 120, 'latency': 20}, 'E': {'capacity': 95, 'latency': 14}, 'I': {'capacity': 85, 'latency': 22}},
        'G': {'D': {'capacity': 100, 'latency': 10}, 'H': {'capacity': 80, 'latency': 8}},
        'H': {'E': {'capacity': 105, 'latency': 16}, 'G': {'capacity': 80, 'latency': 8}, 'I': {'capacity': 115, 'latency': 18}},
        'I': {'F': {'capacity': 85, 'latency': 22}, 'H': {'capacity': 115, 'latency': 18}}
    }
    return topology

def generate_flows(num_flows, time_step, arrival_pattern='uniform', existing_flows=[]):
    """Generates new flows for a given time step."""
    new_flows = []
    if arrival_pattern == 'poisson':
        # Poisson arrival: Number of flows per time step follows Poisson dist.
        num_new_flows = np.random.poisson(num_flows)
    elif arrival_pattern == 'periodic':
        # Periodic: Fixed number of flows every few time steps.
        num_new_flows = num_flows if time_step % 5 == 0 else 0
    elif arrival_pattern == 'batch':
        # Batch arrival: a probability of generating a large batch of flows
        num_new_flows = num_flows * 5 if random.random() < 0.1 else 0
    else:  # Default: Uniform
        num_new_flows = num_flows
    available_nodes = list(generate_network_topology().keys())  # Get nodes from the topology
    for _ in range(num_new_flows):
        source = random.choice(available_nodes)
        # Ensure destination is different from source and exists in the topology
        destination = random.choice([node for node in available_nodes if node != source])
        bandwidth_req = random.randint(*BANDWIDTH_REQ_RANGE)
        latency_sensitivity = random.choice(LATENCY_SENSITIVITIES)
        priority = random.choice(FLOW_PRIORITIES)
        burstiness = random.uniform(0, 0.5)  # Example range
        new_flow = Flow(f"flow_{time_step}_{_}", source, destination,
                        bandwidth_req, latency_sensitivity, priority, burstiness)
        new_flows.append(new_flow)

    # Simulate long-lived flows (keep some existing flows)
    for flow in existing_flows:
        if random.random() < 0.8:  # 80% chance of survival
            new_flows.append(flow)
        # else:  # No need to explicitly delete in Python, garbage collection will handle it.
        #   del flow

    return new_flows

def generate_intents(flows):
    """
    Generates corresponding intents based on flows.

    Args:
        flows (list): List of Flow objects.

    Returns:
        list: List of Intent objects.
    """
    intents = []
    for flow in flows:
        intent = Intent(
            flow_id=flow.flow_id,
            min_bandwidth=flow.bandwidth_req * 0.8,  # 80% of requested
            max_latency=50 if flow.latency_sensitivity == 'High' else None,
            jitter=10 if flow.latency_sensitivity == 'High' else None,
            packet_loss_rate=0.01 if flow.latency_sensitivity == 'High' else None,
            bandwidth_guaranteed=True if flow.priority == 'High' else False,
            latency_guaranteed= True if flow.latency_sensitivity == 'High' and flow.priority == 'High' else False,
            reliability='High' if flow.latency_sensitivity == 'High' else 'Medium',
            sla_tier='Gold' if flow.latency_sensitivity == 'High' and flow.priority == 'High' else 'Silver',
            application_type='video' if flow.latency_sensitivity == 'High' else 'web',
            user_group = 'Sales' if flow.source == 'A' else 'Engineering',
            priority = 5 if flow.latency_sensitivity == 'High' and flow.priority == 'High' else (3 if flow.latency_sensitivity == 'High' else 2)
        )
        intents.append(intent)
    return intents

# --- Simulation Run Function ---

def run_simulation(num_steps, algorithm_name='CAVP-IB-QoS', arrival_pattern='uniform'):  # Add algorithm_name parameter
    """
    Runs the QoS algorithm simulation.

    Args:
        num_steps (int): Number of simulation time steps.
        algorithm_name (str): Algorithm name ('CAVP-IB-QoS', 'WFQ', 'PriorityQueueing', 'StaticAllocation').
        arrival_pattern (str): Arrival pattern ('uniform', 'poisson', 'periodic', 'batch').

    Returns:
        pd.DataFrame: DataFrame containing the simulation results.
    """
    network = Network(generate_network_topology())
    if algorithm_name == 'CAVP-IB-QoS':
        algorithm = CAVP_IB_QoS(network)
    elif algorithm_name == 'WFQ':
        algorithm = WFQ(network)
    elif algorithm_name == 'PriorityQueueing':
        algorithm = PriorityQueueing(network)
    else:  # No algorithm object is needed for static allocation
        algorithm = None

    all_flows = []
    all_intents = []
    results = []

    for step in range(num_steps):
        print(f"--- Step {step} - {algorithm_name} ---")  # Print algorithm name
        new_flows = generate_flows(np.random.randint(1, 4), step, arrival_pattern, all_flows)  # Pass arrival pattern
        all_flows = new_flows  # Replace old flows with new + surviving

        # Update bandwidth requirements (including burstiness)
        for flow in all_flows:
            flow.update_bandwidth_req()  # Update based on burstiness
            # Also, allow for regular bandwidth changes
            if random.random() < BW_CHANGE_PROB:
                flow.bandwidth_req = random.randint(*BANDWIDTH_REQ_RANGE)
                print(f"  Flow {flow.flow_id} changed, new bw: {flow.bandwidth_req}")


        new_intents = generate_intents(all_flows)
        all_intents = new_intents

        if algorithm:
            algorithm.generate_policy(all_flows, all_intents)  # Execute policy generation
        else:  # Static Allocation
            run_static_allocation_step(network, all_flows)  # Execute a single step of static allocation

        # --- Metric Calculation ---
        total_latency = 0
        total_flows_with_path = 0
        intent_fulfillment_count = 0

        for flow in all_flows:
            if flow.path:
                # Calculate *actual* path latency (now that we have dynamic routing)
                path_latency = 0
                for i in range(len(flow.path) - 1):
                    path_latency += network.adjacency_list[flow.path[i]][flow.path[i+1]]['latency']

                total_latency += path_latency
                total_flows_with_path += 1


                intent = None
                if algorithm_name != 'StaticAllocation':  # Static allocation has no intent
                    # intent = algorithm._get_intent(flow.flow_id, all_intents) if algorithm else None  # Original code, incorrect
                    intent = next((intent for intent in all_intents if intent.flow_id == flow.flow_id), None)  # Corrected code, direct lookup
                if intent and flow.allocated_bandwidth >= intent.min_bandwidth:  # Check if intent is met (only check for non-static allocation algorithms)
                    intent_fulfillment_count += 1

        avg_latency = total_latency / total_flows_with_path if total_flows_with_path > 0 else 0
        intent_fulfillment = (intent_fulfillment_count / len(all_flows)) if len(all_flows) > 0 else 0 if algorithm_name != 'StaticAllocation' else np.nan  # Intent fulfillment rate for static allocation is set to NaN

        link_utilizations = {}
        for node in network.adjacency_list:
            for neighbor, link_data in network.adjacency_list[node].items():
                link = (node, neighbor)
                link_utilizations[str(link)] = network.get_link_utilization(link)


        results.append({
            'step': step,
            'avg_latency': avg_latency,
            'intent_fulfillment': intent_fulfillment,
            **link_utilizations,
            'algorithm': algorithm_name  # Record algorithm name
        })

        print(f"  Avg. Latency: {avg_latency:.2f} ms")
        if algorithm_name != 'StaticAllocation':
            print(f"  Intent Fulfillment: {intent_fulfillment:.2f}")
        for link, util in link_utilizations.items():
            print(f"  Link {link} Utilization: {util:.2f}")
    results_df = pd.DataFrame(results)
    results_df = results_df.rename(columns={'step': f'step_{algorithm_name}_{arrival_pattern}'})
    # Save the results of each algorithm-arrival pattern to a CSV file
    results_df.to_csv(f"results_{algorithm_name}_{arrival_pattern}.csv", index=False)

    return results_df



def run_static_allocation_step(network, flows):
    """
    Performs a single step of static bandwidth allocation.

    Args:
        network (Network): Network model instance.
        flows (list): List of Flow objects.
    """
    # Clear previous allocations
    network.flows = {}

    # First, route all flows using Dijkstra (even for static allocation)
    for flow in flows:
        flow.path = []  # Initialize path
        path = []
        dijkstra = CAVP_IB_QoS(network)  # Use a temporary instance for routing only
        path = dijkstra._dijkstra(flow.source, flow.destination)
        if path:  # Only proceed if a path was found
          flow.path = path

          # Calculate total bandwidth to allocate per link on this flow's path
          link_bandwidths = {}
          for i in range(len(flow.path) - 1):
                link = (flow.path[i], flow.path[i+1])
                if link not in link_bandwidths:
                    link_bandwidths[link] = 0

                # Count flows using this specific link *on this path*.  This is the key fix!
                flows_using_link = [
                    f for f in flows
                    if f.path and link[0] in f.path and link[1] in f.path and f.path.index(link[0]) + 1 == f.path.index(link[1])
                ]
                num_flows_on_link = len(flows_using_link)


                if num_flows_on_link > 0:
                   bandwidth_per_flow = network.adjacency_list[link[0]][link[1]]['capacity'] / num_flows_on_link
                   link_bandwidths[link] = min(bandwidth_per_flow, flow.current_bandwidth_req) #per-link, per-flow bandwidth


          #Allocate calculated bandwidth, by link.
          for i in range(len(flow.path) - 1):
                link = (flow.path[i], flow.path[i + 1])
                if link in link_bandwidths:  # flow might not use all links in its path.
                    if flow.flow_id not in network.flows:
                         network.flows[flow.flow_id] = {'path': flow.path, 'bandwidth': 0}
                    flow.allocated_bandwidth += link_bandwidths[link]
                    network.flows[flow.flow_id]['bandwidth'] += link_bandwidths[link]

        else:
            flow.allocated_bandwidth = 0

# --- Run Simulation and Visualization ---

NUM_RUNS = 20  # Number of runs
NUM_STEPS = 20
ALGORITHM_NAMES = ['CAVP-IB-QoS', 'WFQ', 'PriorityQueueing', 'StaticAllocation']

# Store results of multiple runs
all_results = {}
for algorithm_name in ALGORITHM_NAMES:
    all_results[algorithm_name] = {}
    for arrival_pattern in ['uniform', 'poisson', 'periodic', 'batch']:
        all_results[algorithm_name][arrival_pattern] = []


for run in range(NUM_RUNS):
    print(f"----- Run {run + 1} -----")
    for algorithm_name in ALGORITHM_NAMES:
        for arrival_pattern in ['uniform', 'poisson', 'periodic', 'batch']:
            print(f"Running: {algorithm_name}, Arrival: {arrival_pattern}")
            results_df = run_simulation(NUM_STEPS, algorithm_name, arrival_pattern=arrival_pattern)
            all_results[algorithm_name][arrival_pattern].append(results_df)

# Combine results (by algorithm and arrival pattern)
combined_results = {}
for algorithm_name in ALGORITHM_NAMES:
    combined_results[algorithm_name] = {}
    for arrival_pattern in ['uniform', 'poisson', 'periodic', 'batch']:
        combined_results[algorithm_name][arrival_pattern] = pd.concat(
            all_results[algorithm_name][arrival_pattern]
        )
        # Save combined data
        combined_results[algorithm_name][arrival_pattern].to_csv(f"combined_results_{algorithm_name}_{arrival_pattern}.csv", index=False)


# Group and calculate mean and standard deviation (by algorithm, arrival pattern, and time step)
grouped_results = {}
for algorithm_name in ALGORITHM_NAMES:
    grouped_results[algorithm_name] = {}
    for arrival_pattern in ['uniform', 'poisson', 'periodic', 'batch']:
        df = combined_results[algorithm_name][arrival_pattern]

        # 1. Ensure the 'step' column name is unique (prevent duplicates)
        step_col_name = f'step_{algorithm_name}_{arrival_pattern}'

        # 2. Select necessary columns, and perform grouping and aggregation
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [col for col in numeric_cols if 'step' not in col]  # Remove step-like columns.
        grouped_results[algorithm_name][arrival_pattern] = (
            df[[step_col_name] + numeric_cols]  # Use the correct 'step' column name
            .groupby(step_col_name)
            .agg(['mean', 'std'])
            .reset_index()  # Reset index after aggregation
        )

# --- Quantitative Results ---
print("\n--- Quantitative Results ---")

# Create an empty DataFrame to store the results
quant_results_df = pd.DataFrame(columns=['Algorithm', 'Arrival Pattern', 'Avg. Latency (ms)',
                                         'Intent Fulfillment', 'Avg. Link Utilization',
                                         'Max Link Utilization'])  # Add new columns

for algorithm_name in ALGORITHM_NAMES:
    for arrival_pattern in ['uniform', 'poisson', 'periodic', 'batch']:
        df = combined_results[algorithm_name][arrival_pattern]

        # Calculate average latency
        avg_latency = df['avg_latency'].mean()

        # Calculate intent fulfillment rate
        intent_fulfillment = df['intent_fulfillment'].mean()

        # Calculate average link utilization
        link_util_cols = [col for col in df.columns if isinstance(col, str) and col.startswith("('")]
        avg_link_utilization = df[link_util_cols].values.mean()

        # Calculate maximum link utilization
        max_link_utilization = df[link_util_cols].values.max()

        # Add results to the DataFrame
        new_row = pd.DataFrame([{
            'Algorithm': algorithm_name,
            'Arrival Pattern': arrival_pattern,
            'Avg. Latency (ms)': f"{avg_latency:.2f}",
            'Intent Fulfillment': f"{intent_fulfillment:.2f}",
            'Avg. Link Utilization': f"{avg_link_utilization:.2f}",
            'Max Link Utilization': f"{max_link_utilization:.2f}"
        }])
        quant_results_df = pd.concat([quant_results_df, new_row], ignore_index=True)


# Print the table
print(quant_results_df)
# You can also use the to_markdown() method to generate a Markdown table:
# print(quant_results_df.to_markdown(index=False))

# Save table data to a CSV file
quant_results_df.to_csv("quantitative_results.csv", index=False)

# --- Visualization --- (Example: Compare average latency of CAVP-IB-QoS under different arrival patterns)
plt.figure(figsize=(10, 6))
colors = {'uniform': 'blue', 'poisson': 'red', 'periodic': 'green', 'batch': 'orange'}

for arrival_pattern in ['uniform', 'poisson', 'periodic', 'batch']:
    step_data = grouped_results['CAVP-IB-QoS'][arrival_pattern][f'step_{"CAVP-IB-QoS"}_{arrival_pattern}'].to_numpy()
    latency_mean_data = grouped_results['CAVP-IB-QoS'][arrival_pattern][('avg_latency', 'mean')].to_numpy()
    latency_std_data = grouped_results['CAVP-IB-QoS'][arrival_pattern][('avg_latency', 'std')].to_numpy()

    plt.plot(step_data, latency_mean_data, label=f'{arrival_pattern} (Mean)', color=colors[arrival_pattern])
    plt.fill_between(step_data, latency_mean_data - latency_std_data, latency_mean_data + latency_std_data,
                     color=colors[arrival_pattern], alpha=0.2, label=f'{arrival_pattern} (Std Dev)')

plt.xlabel('Time Step')
plt.ylabel('Average Latency (ms)')
plt.title('CAVP-IB-QoS: Average Latency vs. Arrival Pattern')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cavp_latency_vs_arrival.pdf")  # Save as PDF
plt.show()
plt.close()


# --- Visualization --- (Example: Compare average latency of different algorithms under uniform arrival pattern)
plt.figure(figsize=(10, 6))

for i, algorithm_name in enumerate(ALGORITHM_NAMES):
    if algorithm_name == "StaticAllocation":  # skip static
        continue
    step_data = grouped_results[algorithm_name]['uniform'][f'step_{algorithm_name}_uniform'].to_numpy()
    latency_mean_data = grouped_results[algorithm_name]['uniform'][('avg_latency', 'mean')].to_numpy()
    latency_std_data = grouped_results[algorithm_name]['uniform'][('avg_latency', 'std')].to_numpy()

    plt.plot(step_data, latency_mean_data, label=f'{algorithm_name} (Mean)', color=colors[list(colors.keys())[i]])
    plt.fill_between(step_data, latency_mean_data - latency_std_data, latency_mean_data + latency_std_data,
                     color=colors[list(colors.keys())[i]], alpha=0.2, label=f'{algorithm_name} (Std Dev)')

plt.xlabel('Time Step')
plt.ylabel('Average Latency (ms)')
plt.title('Average Latency Comparison (Uniform Arrival)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_comparison_uniform.pdf")  # Save as PDF.
plt.show()
plt.close()


# (Example: Compare average latency of different algorithms under uniform arrival pattern)
# Average latency comparison across algorithms and arrival patterns
plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange']
arrival_patterns = ['uniform', 'poisson', 'periodic', 'batch']
num_algorithms = len(ALGORITHM_NAMES)
bar_width = 0.2
index = np.arange(len(arrival_patterns))

for i, algorithm_name in enumerate(ALGORITHM_NAMES):
    means = []
    for arrival_pattern in arrival_patterns:
        mean_latency = grouped_results[algorithm_name][arrival_pattern][('avg_latency', 'mean')].mean()
        means.append(mean_latency)

    plt.bar(index + i * bar_width, means, bar_width, label=algorithm_name, color=colors[i%len(colors)])

plt.xlabel('Arrival Pattern')
plt.ylabel('Average Latency (ms)')
plt.title('Average Latency Comparison across Algorithms and Arrival Patterns')
plt.xticks(index + bar_width * (num_algorithms - 1) / 2, arrival_patterns)
plt.legend()
plt.tight_layout()
plt.savefig("latency_comparison_all.pdf")  # Save as PDF
plt.show()
plt.close()

# Add more visualization charts...
# Intent fulfillment comparison across algorithms and arrival patterns
plt.figure(figsize=(12, 8))
for i, algorithm_name in enumerate(ALGORITHM_NAMES):
    means = []
    for arrival_pattern in arrival_patterns:
        mean_intent_fulfillment = grouped_results[algorithm_name][arrival_pattern][('intent_fulfillment', 'mean')].mean()
        means.append(mean_intent_fulfillment)

    plt.bar(index + i * bar_width, means, bar_width, label=algorithm_name, color=colors[i % len(colors)])

plt.xlabel('Arrival Pattern')
plt.ylabel('Intent Fulfillment Rate')
plt.title('Intent Fulfillment Comparison across Algorithms and Arrival Patterns')
plt.xticks(index + bar_width * (num_algorithms - 1) / 2, arrival_patterns)
plt.legend()
plt.tight_layout()
plt.savefig("intent_fulfillment_comparison_all.pdf")
plt.show()
plt.close()

# Example: Link utilization over time for a specific link (CAVP-IB-QoS, uniform arrival)
# Iterate over all possible links
for link in [
    (node1, node2)
    for node1 in generate_network_topology()
    for node2 in generate_network_topology()[node1]
    ]:
    plt.figure(figsize=(10, 6))
    link_str = str(link)
    for i, algorithm_name in enumerate(ALGORITHM_NAMES):
        try:  # Try to get data, skip if the algorithm didn't use the link
            step_data = grouped_results[algorithm_name]['uniform'][f'step_{algorithm_name}_uniform'].to_numpy()
            utilization_mean_data = grouped_results[algorithm_name]['uniform'][(link_str, 'mean')].to_numpy()
            utilization_std_data = grouped_results[algorithm_name]['uniform'][(link_str, 'std')].to_numpy()

            plt.plot(step_data, utilization_mean_data, label=f'{algorithm_name} (Mean)', color=colors[i%len(colors)])
            plt.fill_between(step_data, utilization_mean_data - utilization_std_data,
                         utilization_mean_data + utilization_std_data, color=colors[i%len(colors)], alpha=0.2)
        except KeyError:
            print(f"Skipping link {link_str} for {algorithm_name} (no data)")
            continue


    plt.xlabel('Time Step')
    plt.ylabel('Link Utilization')
    plt.title(f'Link Utilization Over Time: {link_str} (Uniform Arrival)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"link_utilization_{link_str.replace(' ', '').replace(',','_')}_uniform.pdf")  # Filename includes link information
    plt.show()
    plt.close()


# --- Configuration Information ---
config = {
    'NUM_RUNS': NUM_RUNS,
    'NUM_STEPS': NUM_STEPS,
    'ALGORITHM_NAMES': ALGORITHM_NAMES,
    'BANDWIDTH_REQ_RANGE': BANDWIDTH_REQ_RANGE,
    'BW_CHANGE_PROB': BW_CHANGE_PROB,
    'LATENCY_SENSITIVITIES': LATENCY_SENSITIVITIES,
    'FLOW_PRIORITIES': FLOW_PRIORITIES,
    'RANDOM_SEED': RANDOM_SEED
}

# Add the network topology to the configuration as well.  Because the adjacency list is a nested dictionary, printing it directly might be hard to read, so format it using json here.
config['NETWORK_TOPOLOGY'] = generate_network_topology()
# Write to JSON file
with open('simulation_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("\nSimulation configuration saved to simulation_config.json")
