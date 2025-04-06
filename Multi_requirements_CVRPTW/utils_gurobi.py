import networkx as nx
from cspy import BiDirectional
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

def create_pricing_problem_graph(arcs, dual_prices, demands, capacity):
    """
    Create a directed graph for the pricing problem with resource constraints.
    """
    G = nx.DiGraph(directed=True, n_res=2)  # Two resources
    for (i, j), cost in arcs.items():
        demand = demands.get(j, 0)
        res_cost = np.array([1, demand])
        adjusted_cost = cost - dual_prices.get(j, 0)
        G.add_edge(i, j, res_cost=res_cost, weight=adjusted_cost)
    return G


def solve_pricing_problem(G, max_res, min_res):
    """
    Solve the pricing problem using BiDirectional algorithm.
    """
    bidirec = BiDirectional(G, max_res, min_res, direction="both", elementary=True, time_limit=5)
    bidirec.run()
    return bidirec


def compute_route_cost(path, depot, distance_matrix):
    """
    Compute the cost of a new route based on the path.
    """
    cost = 0
    for i in range(len(path) - 1):
        if path[i] == "Source":
            cost += distance_matrix[depot - 1][path[i + 1] - 1]
        elif path[i + 1] == "Sink":
            cost += distance_matrix[path[i] - 1][depot - 1]
        else:
            cost += distance_matrix[path[i] - 1][path[i + 1] - 1]
    return cost


def add_new_route_to_master(master, lambda_vars, routes, a_ir, new_route_id, new_route_type, new_route_cost, covered_customers, vehicle_types, customers):
    """
    Add a new route to the master problem.
    """
    # Update routes and a_ir
    a_ir[new_route_type][new_route_id] = {i: 1 if i in covered_customers else 0 for i in customers}
    routes[new_route_id] = {
        "cost": new_route_cost,
        "route": covered_customers,
        "Type": new_route_type,
    }

    # Add new lambda variable
    lambda_vars[(new_route_id, new_route_type)] = master.addVar(
        obj=new_route_cost, vtype=GRB.CONTINUOUS, name=f"lambda_{new_route_id}_{new_route_type}", lb=0, ub=1
    )

    master.update()

    return master


def rebuild_customer_constraints(master, lambda_vars, a_ir, routes, vehicle_types, customers):
    """
    Rebuild customer coverage constraints for the master problem.
    """
    # Remove existing constraints
    for constr in master.getConstrs():
        master.remove(constr)
    customer_constraints = {}
    for customer in customers:
        customer_constraints[customer] = master.addConstr(
            gp.quicksum(
                lambda_vars[(route_id, route_type)] * a_ir[route_type][route_id][customer]
                for route_id, route_info in routes.items()
                for route_type in vehicle_types
                if route_type == route_info["Type"]
            ) == 1,
            name=f"{customer}",
        )
    master.update()
    return master


def solve_master_problem(master):
    """
    Solve the master problem and return the optimal value and variable assignments.
    """
    master.setParam('OutputFlag', 0)
    master.optimize()
    if master.Status == GRB.OPTIMAL:
        return master.objVal, {var.VarName: var.X for var in master.getVars()}
    else:
        raise Exception(f"Master problem optimization failed with status {master.Status}")


def process_fractional_solution(master, model_path="master.lp"):
    """
    Process fractional solutions by converting the master problem to a MIP.
    """
    master.write(model_path)
    model = gp.read(model_path)
    model.setParam('OutputFlag', 0)
    model.setParam("MIPFocus", 1) 

    # Change variables to binary
    for var in model.getVars():
        var.vtype = GRB.BINARY
    model.update()
    model.optimize()
    gap = model.MIPGap

    for var in master.getVars():
    # if variable is fractional, then get the value from the MIP model
        if var.X < 1- 1e-4 and var.X > 0 + 1e-4:
            for var_mip in model.getVars():
                if var.VarName == var_mip.VarName:
                    # print(f"{var.VarName} = {var_mip.X}")
                    var_mip.lb = var_mip.X
                # var_mip.ub = GRB.INFINITY
    model.write("model.lp")

    for var in model.getVars():
        var.vtype = GRB.CONTINUOUS
    model.update()
    model.optimize()

    return model, gap


def extract_dual_prices(model):
    """
    Extract dual prices from the model.
    """
    return {int(constr.ConstrName): constr.Pi for constr in model.getConstrs()}