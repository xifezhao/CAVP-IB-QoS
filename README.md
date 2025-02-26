```markdown
# Network QoS Simulation with CAVP-IB-QoS and Other Algorithms

This repository contains a Python simulation for evaluating different Quality of Service (QoS) algorithms in a network, including a novel algorithm called CAVP-IB-QoS (Congestion-Aware, Variable-Priority, Intent-Based QoS), Weighted Fair Queueing (WFQ), Priority Queueing, and Static Allocation.  The simulation focuses on dynamic network conditions, including varying traffic arrival patterns, changing bandwidth requirements, and flow burstiness.

## Table of Contents

- [Introduction](#introduction)
- [Algorithms Implemented](#algorithms-implemented)
- [Network Model](#network-model)
- [Flow and Intent Representation](#flow-and-intent-representation)
- [Simulation Setup](#simulation-setup)
- [Running the Simulation](#running-the-simulation)
- [Results and Visualization](#results-and-visualization)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [How to Contribute](#how-to-contribute)
- [License](#license)

## Introduction

Providing Quality of Service (QoS) in networks is crucial for ensuring that different types of traffic receive the appropriate resources (bandwidth, latency, jitter, etc.) to meet their performance requirements. This simulation compares the performance of several QoS algorithms under various dynamic conditions.  The key contribution is the CAVP-IB-QoS algorithm, which combines congestion-aware routing (using a modified Dijkstra's algorithm with dynamic link costs), variable-priority scheduling, and intent-based QoS management.  The other algorithms (WFQ, Priority Queueing, and Static Allocation) serve as baselines for comparison.

## Algorithms Implemented

The following algorithms are implemented and compared in this simulation:

1.  **CAVP-IB-QoS:**  This is the core algorithm. It uses Dijkstra's algorithm for path finding, with link costs dynamically calculated based on latency and utilization.  Bandwidth allocation is priority-based, taking into account flow priorities, latency sensitivities, and user-defined intents. The algorithm also adapts to changing network conditions by recalculating routes and bandwidth allocations.

2.  **WFQ (Weighted Fair Queueing):**  A classic scheduling algorithm that provides fair bandwidth allocation among flows based on assigned weights.  In this implementation, weights are derived from flow priorities (High=3, Medium=2, Low=1).

3.  **Priority Queueing:** A simple algorithm where flows are served strictly based on their priority level (High, Medium, Low).  Higher priority flows are always served before lower priority flows.

4.  **Static Allocation:**  A baseline algorithm where bandwidth is allocated equally among all flows sharing a link.  This serves as a simple comparison point and does not adapt to flow priorities or intents.  Routing is still performed using Dijkstra's algorithm, but bandwidth is allocated statically once the path is determined.

## Network Model

The network is modeled as a directed graph, represented by an adjacency list.  Each link in the network has a capacity (bandwidth) and a base latency. The `Network` class in the code encapsulates the network topology and provides methods for:

-   Calculating link utilization.
-   Calculating dynamic link costs (used by CAVP-IB-QoS).
-   Accessing neighbors of a node.

A sample network topology is generated by the `generate_network_topology()` function, but this can be easily modified or replaced with a custom topology.

## Flow and Intent Representation

-   **Flow:**  The `Flow` class represents a network flow, characterized by:
    -   `flow_id`: A unique identifier.
    -   `source`: The source node.
    -   `destination`: The destination node.
    -   `bandwidth_req`:  The initial bandwidth requirement.
    -   `latency_sensitivity`:  The sensitivity to latency (High, Medium, Low).
    -   `priority`: The priority of the flow (High, Medium, Low).
    -   `burstiness`:  A value between 0.0 and 1.0 representing the burstiness of the flow.
    -   `current_bandwidth_req`: The *current* bandwidth requirement, which can vary due to burstiness.
    -   `path`: The path assigned to the flow (calculated by the routing algorithm).
    -   `allocated_bandwidth`: The total bandwidth allocated to the flow.

-   **Intent:** The `Intent` class represents a user's QoS requirements for a flow.  It includes:
    -   `flow_id`: The ID of the flow the intent applies to.
    -   `min_bandwidth`: The minimum bandwidth required.
    -   `max_latency`:  The maximum acceptable latency (optional).
    -   `jitter`:  The maximum acceptable jitter (optional).
    -   `packet_loss_rate`: The maximum acceptable packet loss rate (optional).
    -   `bandwidth_guaranteed`:  Whether bandwidth is guaranteed.
    -   `latency_guaranteed`: Whether latency is guaranteed.
    -   `reliability`:  The required reliability level.
    -   `sla_tier`:  The Service Level Agreement (SLA) tier (e.g., Gold, Silver).
    -   `application_type`: The type of application (e.g., video, web).
    -   `user_group`: The user group the flow belongs to.
    -   `priority`: An integer representing priority (higher value = higher priority).

## Simulation Setup

The simulation runs for a specified number of time steps (`NUM_STEPS`).  In each step:

1.  **Flow Generation:**  New flows are generated using the `generate_flows()` function.  The number of flows generated per step can follow different arrival patterns:
    -   `uniform`: A random number of flows are generated.
    -   `poisson`: The number of flows follows a Poisson distribution.
    -   `periodic`:  A fixed number of flows arrive at regular intervals.
    -   `batch`: A large batch of flows may arrive with a small probability.

    Existing flows can also persist between steps, simulating long-lived flows.

2.  **Intent Generation:**  Intents are generated for each flow using the `generate_intents()` function.  The intent parameters are derived from the flow's characteristics.

3.  **QoS Policy Generation/Adaptation:** The selected QoS algorithm (CAVP-IB-QoS, WFQ, PriorityQueueing, or Static Allocation) generates or adapts its policy based on the current flows and intents.  This involves routing (finding paths for flows) and bandwidth allocation.

4.  **Metrics Calculation:**  The simulation calculates the following metrics:
    -   Average latency of flows with paths.
    -   Intent fulfillment rate (percentage of flows whose intents are met).
    -   Link utilization for each link.

5. **Dynamic Updates:** Flow bandwidth may change due to bw_change_prob or burstiness.

## Running the Simulation

The simulation is run by executing the Python script.  The `run_simulation()` function performs a single simulation run for a given algorithm and arrival pattern.  The main part of the script runs the simulation multiple times (`NUM_RUNS`) for each combination of algorithm and arrival pattern. The main part of the simulation is wrapped in loops, like so:

```python
for run in range(NUM_RUNS):
    for algorithm_name in ALGORITHM_NAMES:
        for arrival_pattern in ['uniform', 'poisson', 'periodic', 'batch']:
            results_df = run_simulation(NUM_STEPS, algorithm_name, arrival_pattern=arrival_pattern)
```
This will run the simulation `NUM_RUNS` times for each possible combination. The results are stored in pandas dataframes.

## Results and Visualization

The simulation produces the following outputs:

-   **CSV Files:**
    -   `results_[algorithm_name]_[arrival_pattern].csv`:  Contains the results of each individual simulation run for a specific algorithm and arrival pattern.
    -   `combined_results_[algorithm_name]_[arrival_pattern].csv`: Contains the combined results of all runs for a specific algorithm and arrival pattern.
    -   `quantitative_results.csv`:  A summary table of key metrics (average latency, intent fulfillment, average/max link utilization) for each algorithm and arrival pattern.
    -   `simulation_config.json`: Contains the parameter settings used in the simulation.

-   **PDF Plots:**
    -   `cavp_latency_vs_arrival.pdf`: Compares average latency for CAVP-IB-QoS across different arrival patterns.
    -   `latency_comparison_uniform.pdf`: Compares average latency for different algorithms under the uniform arrival pattern.
    -   `latency_comparison_all.pdf`:  Compares average latency for different algorithms across all arrival patterns (bar chart).
    -   `intent_fulfillment_comparison_all.pdf`: Compares intent fulfillment rates for different algorithms across all arrival patterns (bar chart).
    -   `link_utilization_[link]_uniform.pdf`: Shows link utilization over time for a specific link under the uniform arrival pattern for all algorithms that use that link. The `[link]` part of the filename is replaced with a string representation of the link, e.g., `link_utilization_A_B_uniform.pdf`.

The visualization code uses `matplotlib` to generate plots showing:

-   Average latency vs. time step for different algorithms and arrival patterns.
-   Intent fulfillment rate vs. time step for different algorithms and arrival patterns.
-   Link utilization vs. time step for specific links.

The quantitative results are printed to the console and saved to a CSV file.  The plots are saved as PDF files.

## Dependencies

The code requires the following Python libraries:

-   `numpy`: For numerical operations.
-   `pandas`: For data manipulation and analysis.
-   `matplotlib`: For plotting.
-   `seaborn`: For enhanced plotting (optional, but used in the provided code).
-   `heapq`: For the priority queue implementation in Dijkstra's algorithm.
-   `json` : For writing config file.

You can install these using `pip`:

```bash
pip install numpy pandas matplotlib seaborn
```

## Configuration

The simulation parameters are defined as constants at the beginning of the script:

-   `RANDOM_SEED`:  The random seed for reproducibility.
-   `NUM_RUNS`: The number of simulation runs to perform.
-   `NUM_STEPS`: The number of time steps in each simulation run.
-   `ALGORITHM_NAMES`:  A list of the algorithms to be evaluated.
-   `BANDWIDTH_REQ_RANGE`: The range for generating random bandwidth requirements.
-   `BW_CHANGE_PROB`: The probability of a flow's bandwidth requirement changing in a time step.
-   `LATENCY_SENSITIVITIES`:  The possible latency sensitivity levels.
-   `FLOW_PRIORITIES`: The possible flow priority levels.

These parameters, along with the generated network topology, are saved to a `simulation_config.json` file for reference.  You can modify these parameters directly in the script or load them from the JSON file.

## File Structure

-   **`your_script_name.py`** (replace with the actual name of your Python file): Contains the entire simulation code.
-   **`results_[algorithm_name]_[arrival_pattern].csv`**: CSV files containing the raw simulation results.
-   **`combined_results_[algorithm_name]_[arrival_pattern].csv`**:  CSV files containing the combined results.
-   **`quantitative_results.csv`**: CSV file with the summarized quantitative results.
-   **`simulation_config.json`**: JSON file storing the simulation configuration.
-   **`*.pdf`**:  PDF files containing the generated plots.

## How to Contribute

1.  **Fork the repository.**
2.  **Create a new branch:** `git checkout -b your-feature-branch`
3.  **Make your changes and commit them:** `git commit -m "Add some feature"`
4.  **Push to the branch:** `git push origin your-feature-branch`
5.  **Submit a pull request.**

Contributions such as bug fixes, improvements to the algorithms, new visualization methods, or extensions to the simulation (e.g., adding different network topologies, traffic models, or QoS algorithms) are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (you'll need to create a LICENSE file and put the MIT license text in it).  If you don't create a LICENSE file, you should at least state the license clearly in the README.
```

**Key improvements and explanations in this README:**

*   **Clear and Concise Introduction:**  Explains the purpose of the simulation and highlights the novel CAVP-IB-QoS algorithm.
*   **Detailed Algorithm Descriptions:**  Provides clear explanations of each algorithm, including the key differences and how they handle routing and bandwidth allocation.
*   **Comprehensive Simulation Setup:**  Explains the flow generation, intent generation, policy execution, and metrics calculation steps.  Also explains the different arrival patterns.
*   **Well-Defined Network Model:**  Clearly describes the network representation and the functions of the `Network` class.
*   **Thorough Flow and Intent Representation:**  Explains the attributes of the `Flow` and `Intent` classes and their significance.
*   **Clear Instructions on Running the Simulation:** Explains how the simulation is run and how the results are generated.
*   **Detailed Results and Visualization Section:**  Lists the output files (CSV and PDF) and describes the generated plots.
*   **Dependencies:** Lists the required Python libraries and provides instructions for installing them.
*   **Configuration:** Explains the simulation parameters and how to modify them.
*   **File Structure:** Describes the organization of the project files.
*   **How to Contribute:** Provides guidelines for contributing to the project.
*   **License:** Specifies the license under which the project is released (crucial for open-source projects).
*   **Table of Contents:** Makes the README easier to navigate.
*   **Code Snippets:** Includes a small code snippet to show how the main simulation loop is structured.
* **Markdown Formatting:** Uses Markdown effectively for readability and organization, including headings, lists, code blocks, and links.

This comprehensive README provides all the necessary information for someone to understand, run, and contribute to the network QoS simulation project. It's well-organized, clearly written, and covers all the essential aspects of the project.  It also explains *why* things are done the way they are, which is important for understanding the code.
