import networkx as nx
from cspy import BiDirectional
import coptpy as cp
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
        obj=new_route_cost, vtype=cp.COPT.CONTINUOUS, name=f"lambda_{new_route_id}_{new_route_type}", lb=0, ub=1
    )


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
            cp.quicksum(
                lambda_vars[(route_id, route_type)] * a_ir[route_type][route_id][customer]
                for route_id, route_info in routes.items()
                for route_type in vehicle_types
                if route_type == route_info["Type"]
            ) == 1,
            name=f"{customer}",
        )

    return master


def solve_master_problem(master):
    """
    Solve the master problem and return the optimal value and variable assignments.
    """
    # model.setParam('Logging', 0)
    master.solve()
    if master.status == cp.COPT.OPTIMAL:
        return master.objVal, {var.name: var.x for var in master.getVars()}
    else:
        raise Exception(f"Master problem optimization failed with status {master.Status}")


def process_fractional_solution(master, model_path="master.lp"):
    """
    Process fractional solutions by converting the master problem to a MIP.
    """
    # master.write(model_path)
    # for constr in master.getConstrs():
    #     print(constr.name)
    model = master.clone()
    # model = env.createModel("fractional")
    # model.read(model_path)
    # for constr in model.getConstrs():
    #     print(constr)
    #     print(constr.name)

    
    model.setParam('Logging', 0)
    

    # Change variables to binary
    for var in model.getVars():
        model.setVarType(var, cp.COPT.BINARY)
    model.solve()
    gap = model.getParam('RelGap')

    master_vars = master.getVars()
    model_vars = model.getVars()

    for i,var in enumerate(master_vars):
    # if variable is fractional, then get the value from the MIP model
        if var.x < 1- 1e-4 and var.x > 0 + 1e-4:
            for j,var_mip in enumerate(model_vars):
                try:
                    # if var.name == var_mip.name:
                    if i==j:
                        # print(f"{var.VarName} = {var_mip.X}")
                        # print(f"var.name = {var.name}")
                        # print(f"var_mip.name = {var_mip.name}")
                        # print(f"var_mip.x = {var_mip.x}")
                        # print(f"var_mip.lb = {var_mip.lb}")
                        # var_mip.setInfo(cp.COPT.Info.LB, var_mip.x)
                        model.setInfo(cp.COPT.Info.UB, var_mip, model.getValues()[j])
                # var_mip.ub = GRB.INFINITY
                except Exception as e:
                    print(e)

    model.write("model.lp")

    for var in model.getVars():
        model.setVarType(var, cp.COPT.CONTINUOUS)
    model.solve()

    return model, gap


def extract_dual_prices(model):
    """
    Extract dual prices from the model.
    """
    # return {int(constr.name): constr.pi for constr in model.getConstrs()}
    return {int(id+2): constr.pi for id,constr in enumerate(model.getConstrs())}
