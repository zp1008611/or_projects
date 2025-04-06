import networkx as nx
from cspy import BiDirectional
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

def create_pricing_problem_graph(arcs, dual_prices, demands, capacity):
    """
    Create a directed graph for the pricing problem with resource constraints.
    该函数通过遍历弧的信息，为每个弧计算资源成本和调整后的成本，    
    并将其添加到有向图中，最终返回一个带有资源约束的有向图。
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
    该函数通过调用 BiDirectional 算法，在给定的有向图上搜索满足资源约束的最优路径，并返回算法的运行结果
    """
    bidirec = BiDirectional(G, max_res, min_res, direction="both", elementary=True, time_limit=5)
    bidirec.run()
    return bidirec


def compute_route_cost(path, depot, distance_matrix):
    """
    Compute the cost of a new route based on the path.
    该函数通过遍历路径中的每个节点对，根据节点的类型和距离矩阵计算路线的总成本，并返回该成本
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
    该函数通过更新路线信息和添加新的变量，将一条新的路线添加到主问题中，并返回更新后的模型
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
    该函数通过设置求解参数、求解主问题、检查求解状态，并根据结果返回最优值和变量赋值，或者抛出异常提示优化失败
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
    该函数通过将主问题转换为 MIP 模型，求解 MIP 模型，处理分数变量，
    最后将变量类型改回连续型并再次求解，以获得更好的解
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
    该函数通过遍历模型中的所有约束条件，提取每个约束条件的对偶价格，并将其存储在一个字典中返回
    """
    return {int(constr.ConstrName): constr.Pi for constr in model.getConstrs()}