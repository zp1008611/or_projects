# %%
import numpy as np
import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time
import networkx as nx
from cspy import BiDirectional
import os
from utils_gurobi import (create_pricing_problem_graph,
                   solve_pricing_problem,
                   compute_route_cost,
                   add_new_route_to_master,
                   rebuild_customer_constraints,
                   solve_master_problem,
                   process_fractional_solution,
                   extract_dual_prices
                )

# %% [markdown]
# # Read data

# %%


def preprocess_cvrp(data_file):

    """
    该函数通过读取、解析数据文件，计算距离矩阵，并将数据合并为一个字典，完成了 CVRP 数据的预处理工作
    """
    # Step 1: Read and parse the data
    with open(data_file, "r") as file:
        lines = file.readlines()
    
    # Extract sections
    node_coord_section = []
    demand_section = []
    capacity = 0
    depot = 0
    
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            current_section = "NODE_COORD"
        elif line.startswith("DEMAND_SECTION"):
            current_section = "DEMAND"
        elif line.startswith("DEPOT_SECTION"):
            current_section = "DEPOT"
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1])
        elif current_section == "NODE_COORD" and line and not line.startswith("EOF"):
            parts = list(map(int, line.split()))
            node_coord_section.append((parts[0], parts[1], parts[2]))
        elif current_section == "DEMAND" and line and not line.startswith("EOF"):
            parts = list(map(int, line.split()))
            demand_section.append((parts[0], parts[1]))
        elif current_section == "DEPOT" and line and not line.startswith("EOF"):
            if depot == 1:
                continue
            depot = int(line)
            # print(depot)

    # Convert to pandas DataFrames
    nodes_df = pd.DataFrame(node_coord_section, columns=["Node", "X", "Y"])
    demands_df = pd.DataFrame(demand_section, columns=["Node", "Demand"])

    # Step 2: Compute distance matrix
    num_nodes = len(nodes_df)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x1, y1 = nodes_df.loc[i, ["X", "Y"]]
                x2, y2 = nodes_df.loc[j, ["X", "Y"]]
                distance_matrix[i, j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Step 3: Combine and prepare data
    nodes_df = nodes_df.merge(demands_df, on="Node")
    return {
        "nodes": nodes_df,
        "distance_matrix": distance_matrix,
        "capacity": capacity,
        "depot": depot
    }



# %% [markdown]
# # Inital feasible routes

# %%
def create_initial_feasible_solution_with_nodes(data):
    """
    该函数通过分离客户节点、生成初始路线并合并，为 CVRP 问题创建了初始的可行解
    """
    depot = data["depot"]  # Depot node (1 in this case)
    capacity = data["capacity"]  # Vehicle capacity
    nodes = data["nodes"]  # Node DataFrame with 'demand' and other details
    distance_matrix = data["distance_matrix"]  # Distance matrix
    customer_nodes = list(data["nodes"]["Node"])  # Extract all node numbers

    # Exclude the depot and segregate into odd and even customer nodes
    odd_customers = [node for node in customer_nodes if node != depot and node % 2 != 0]
    even_customers = [node for node in customer_nodes if node != depot and node % 2 == 0]
    

    # Create initial routes for each customer set
    def generate_routes(customer_set, route_type):
        initial_routes = {}
        for customer in customer_set:
            # route_id = f"Route_{customer}_{route_type}"  # Unique route ID
            route_id = f"Route_{customer}"  # Unique route ID
            route_cost = (
                2 * distance_matrix[depot - 1][customer - 1]
            )  # Depot -> Customer -> Depot
            route_demand = nodes.loc[customer -1, "Demand"]  # Demand of the customer
            initial_routes[route_id] = {
                "cost": route_cost,
                "route": [depot, customer, depot],  # Depot -> Customer -> Depot
                "Type" : route_type,
            }
        return initial_routes

    # Generate routes for odd and even customers
    initial_routes_typeI = generate_routes(odd_customers, "TypeI")
    initial_routes_typeII = generate_routes(even_customers, "TypeII")

    # Combine all routes
    initial_routes = {**initial_routes_typeI, **initial_routes_typeII}

    return initial_routes

# %% [markdown]
# # Initial feasible solution

# %%
def initialize_and_solve_master_problem(routes, data, vehicle_types, customers, a_ir, route_costs):
    """
    Initializes and solves the Master Problem for the VRP using Gurobi.
    
    Args:
        routes (dict): Dictionary containing route information with the structure:
                       {
                           route_id: {
                               "Type": route_type,
                               "cost": route_cost,
                               "route": [list_of_customers]
                           },
                           ...
                       }
        data (dict): Dictionary containing problem data, specifically `nodes` which
                     includes the customers and depot information.
        vehicle_types (list): List of vehicle types (e.g., ["TypeI", "TypeII"]).
    
    Returns:
        tuple: A tuple containing:
               - solution (dict): Includes:
                   - 'Objective': Objective value of the solution.
                   - 'Variables': Selected route variables and their values.
                   - 'Status': Optimization status (e.g., Optimal, Infeasible).
               - dual_prices (dict): Dictionary of dual prices in the format {customer: dual_price}.
    """
    # Extract customers excluding the depot (Node 1)
    

    for route_id, route_info in routes.items():
        route_type = route_info["Type"]
        route_costs[(route_id, route_type)] = route_info["cost"]
        a_ir[route_type][route_id] = {
            i: 1 if i in route_info["route"] else 0 for i in customers
        }

    # Initialize the Gurobi model
    master = gp.Model("MasterProblem")
    master.setParam('OutputFlag', 0)
    # Add decision variables for route selection
    lambda_vars = {}
    for route_id, route_info in routes.items():
        route_type = route_info["Type"]
        lambda_vars[(route_id, route_type)] = master.addVar(
            obj=route_info["cost"], vtype=GRB.CONTINUOUS,
            name=f"lambda_{route_id}_{route_type}", lb=0, ub=1
        )

    # Add customer coverage constraints
    customer_constraints = {}
    for customer in customers:
        customer_constraints[customer] = master.addConstr(
            gp.quicksum(
                lambda_vars[(route_id, route_type)] * a_ir[route_type][route_id][customer]
                for route_id, route_info in routes.items()
                for route_type in vehicle_types
                if route_type == route_info["Type"]
            ) == 1,
            name=f"{customer}"
        )

    # Optimize the Master Problem
    master.optimize()

    # Initialize the solution dictionary
    solution = {}
    dual_prices = {}

    if master.Status == GRB.OPTIMAL:
        # Store the objective value
        solution["Objective"] = master.ObjVal

        # Store the variables with their values
        solution["Variables"] = {
            var.VarName: var.X for var in master.getVars() if var.X > 0
        }

        # Store the dual prices for each customer constraint
        for constr in master.getConstrs():
            dual_prices[int(constr.ConstrName)] = constr.Pi

        # Set the status
        solution["Status"] = "Optimal"
    else:
        # If the model isn't optimal, return empty values
        solution["Status"] = "Infeasible or Suboptimal"
        solution["Objective"] = None
        solution["Variables"] = {}

    # Return the solution dictionary and dual prices
    return solution, dual_prices, master, lambda_vars

# %% [markdown]
# # Pricing Problem

# %%

def build_arcs_and_demands(customers, max_customers, depot, distance_matrix, nodes):
    """
    Build arcs and demands for a given set of customers.
    """
    arcs = {}
    demands = {}

    for i in customers:
        if i < max_customers:
            arcs[("Source", i)] = distance_matrix[depot - 1][i - 1]
            arcs[(i, "Sink")] = distance_matrix[i - 1][depot - 1]

    for i in customers:
        for j in customers:
            if i != j and i < max_customers and j < max_customers:
                arcs[(i, j)] = distance_matrix[i - 1][j - 1]
                arcs[(j, i)] = distance_matrix[j - 1][i - 1]

    for i in customers:
        if i < max_customers:
            demands[i] = nodes.loc[i - 1, "Demand"]

    return arcs, demands


def process_customer_set(
    customer_set,
    max_customers,
    depot,
    distance_matrix,
    nodes,
    dual_prices,
    capacity,
    routes,
    lambda_vars,
    vehicle_types,
    master,
    a_ir,
    customers,
):
    """
    Process a set of customers (odd or even) by building the pricing problem,
    solving it, and adding the resulting route to the master problem.
    """
    start_pricing = time.time()

    arcs, demands = build_arcs_and_demands(
        customer_set, max_customers, depot, distance_matrix, nodes
    )

    # Solve the pricing problem
    G = create_pricing_problem_graph(arcs, dual_prices, demands, capacity)
    bidirec = solve_pricing_problem(G, max_res=[len(arcs), capacity], min_res=[0, 0])

    pricing_time = time.time() - start_pricing

    if bidirec.total_cost >= -1e-6:
        return master, pricing_time
    # Compute route details
    new_route_cost = compute_route_cost(bidirec.path, depot, distance_matrix)
    covered_customers = [
        node for node in bidirec.path if node != "Source" and node != "Sink"
    ]
    new_route_type = "TypeI" if covered_customers[0]%2 != 0 else "TypeII"
    new_route_id = f"Extra_Route{len(lambda_vars) + 3}"
    # print(f"New route {new_route_id} with cost {new_route_cost} and customers {covered_customers} and type {new_route_type}")

    # Add the new route to the master problem
    master = add_new_route_to_master(
        master,
        lambda_vars,
        routes,
        a_ir,
        new_route_id,
        new_route_type,
        new_route_cost,
        covered_customers,
        vehicle_types,
        customers,
    )

    return master, pricing_time

# %% [markdown]
# # Master Problem

# %%
def solve_and_update_master_problem(master, lambda_vars, a_ir, routes, vehicle_types, customers):
    """
    Rebuild customer constraints and solve the master problem.
    """
    start_master = time.time()

    master = rebuild_customer_constraints(master, lambda_vars, a_ir, routes, vehicle_types, customers)
    obj_val, variable_assignments = solve_master_problem(master)

    master_time = time.time() - start_master
    # print(f"Master Problem Objective Value = {obj_val}")
    return master, obj_val, variable_assignments, master_time

# %% [markdown]
# # Column Generation

# %%
def run_column_generation(
    master,
    odd_customers,
    even_customers,
    depot,
    distance_matrix,
    nodes,
    dual_prices,
    capacity,
    a_ir,
    routes,
    vehicle_types,
    customers,
    lambda_vars,
    prev_sol,
    increment_odd,
    increment_even,
    time_limit,
):
    """
    Run the column generation process for a fixed time limit.
    """
    start_time = time.time()
    iteration = 0
    max_customers_odd, max_customers_even = increment_odd + 23, increment_even + 22
    max_customers_odd = min(max_customers_odd, len(customers))
    max_customers_even = min(max_customers_even, len(customers))

    total_pricing_time = 0
    total_master_time = 0

    while time.time() - start_time < time_limit:
        iteration += 1
        # print(f"Iteration: {iteration}")

        # Process odd customers
        master, pricing_time_odd = process_customer_set(
            odd_customers,
            max_customers_odd,
            depot,
            distance_matrix,
            nodes,
            dual_prices,
            capacity,
            routes,
            lambda_vars,
            vehicle_types,
            master,
            a_ir,
            customers,
        )
        total_pricing_time += pricing_time_odd

        # Process even customers
        master, pricing_time_even = process_customer_set(
            even_customers,
            max_customers_even,
            depot,
            distance_matrix,
            nodes,
            dual_prices,
            capacity,
            routes,
            lambda_vars,
            vehicle_types,
            master,
            a_ir,
            customers,
        )
        total_pricing_time += pricing_time_even

        # Solve and update master problem
        master, obj_val, variable_assignments, master_time = solve_and_update_master_problem(
            master, lambda_vars, a_ir, routes, vehicle_types, customers
        )
        total_master_time += master_time

        # Process fractional solutions in master problem
        mip_model, gap = process_fractional_solution(master)
        dual_prices = extract_dual_prices(mip_model)

        # Print MIP objective and gap
        # print("MIP Objective Value:", mip_model.ObjVal)
        # print("MIP Gap:", gap)

        # Increment customer limits
        if max_customers_odd == len(odd_customers) or max_customers_even == len(even_customers):
            prev_sol = obj_val
        else:
            max_customers_odd += increment_odd
            max_customers_even += increment_even
            prev_sol = obj_val

    # Output final results
    final_obj_val = mip_model.ObjVal
    final_gap = gap

    print(f"Final Objective Value: {final_obj_val}")
    print(f"Total Pricing Time: {total_pricing_time:.2f} seconds")
    print(f"Total Master Time: {total_master_time:.2f} seconds")
    print(f"Final MIP Gap: {final_gap}")

    return final_obj_val, total_pricing_time, total_master_time, final_gap

# %% [markdown]
# # Single Instance

# %%
def process_instance(file_path):
    """
    Process a single CVRP instance file and return results for the instance.
    """
    # Preprocess the data and initialize the problem
    start_time = time.time()
    
    data = preprocess_cvrp(file_path)
    routes = create_initial_feasible_solution_with_nodes(data)
    customers = list(range(2, len(data["nodes"]) + 1))
    vehicle_types = ["TypeI", "TypeII"]

    # Precompute feasibility matrix a_ir and route costs
    a_ir = {t: {} for t in vehicle_types}
    route_costs = {}
    solution, dual_prices, master, lambda_vars = initialize_and_solve_master_problem(
        routes, data, vehicle_types, customers, a_ir, route_costs
    )

    # print(f"dual_prices: {dual_prices}")
    print(f"initial_solution: {solution['Objective']}")
    prev_sol = solution["Objective"]
    # break
    # Extract problem-specific parameters
    depot = data["depot"]
    capacity = data["capacity"]
    nodes = data["nodes"]
    distance_matrix = data["distance_matrix"]
    customer_nodes = list(data["nodes"]["Node"])

    # Segregate customer nodes into odd and even
    odd_customers = [node for node in customer_nodes if node != depot and node % 2 != 0]
    even_customers = [node for node in customer_nodes if node != depot and node % 2 == 0]

    end_time = time.time()
    # Set column generation time limit
    time_limit = 300  - (end_time - start_time)

    # Run column generation and retrieve results
    final_obj_val, total_pricing_time, total_master_time, final_gap = run_column_generation(
        master,
        odd_customers,
        even_customers,
        depot,
        distance_matrix,
        nodes,
        dual_prices,
        capacity,
        a_ir,
        routes,
        vehicle_types,
        customers,
        lambda_vars,
        prev_sol,
        increment_odd=2,
        increment_even=2,
        time_limit=time_limit,
    )

    # Prepare results dictionary
    result = {
        "Instance": os.path.basename(file_path),  # Use the file name as the instance name
        "Root Node LP Objective": solution["Objective"],
        "Best PnB IP Solution": final_obj_val,
        "GAP": final_gap,
        "Time for Column Generation": total_pricing_time,
        "Time for IP": total_master_time,
    }
    return result

# %% [markdown]
# # Main

# %%

def main():
    """
    Main function to process all VRP instance files in the specified directory
    and return a consolidated DataFrame of results.
    """
    results = []  # Initialize an empty list to store results

    for root, dirs, files in os.walk("./Uchoa-Vrp-Set-X"):
        files = sorted(files)
        for file in files:
            if file.endswith(".vrp"):  # Process only .vrp files
                print(f"----------------{file}----------------")
                file_path = os.path.join(root, file)
                try:
                    result = process_instance(file_path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                # break

    # Convert results to a Pandas DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file (optional)
    results_df.to_csv("cvrp_results_scenario_1.csv", index=False)

    return results_df




# %%
if __name__ == "__main__":
    results_df = main()
    print(results_df)



