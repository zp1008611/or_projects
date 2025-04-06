import coptpy as cp
# from pyomo.environ import *
# from pyomo.opt import SolverFactory  
from VRPTW import Customer, Graph, Path, Instance
import numpy as np
import math
from typing import List, Dict
from itertools import product


def solve_cvrptw(g,Q,K):
    c_ij = g.distance_matrix
    N = list(g.all_nodes.keys())
    C = list(g.all_customers.keys())
    M_ij = {(i, j): 100 for i in N for j in N}  # 大 M 值
    K = list(range(K))
    # 创建 COPT 环境和模型
    env = cp.Envr()
    model = env.createModel("CVRPTW")


    # 定义决策变量 x_ijk
    x_ijk = {}
    w_ik = {}
    for k in K:
        for i in N:
            for j in N:
                # x_ijk[i, j, k] = model.addVar(vtype=cp.COPT.BINARY, name=f"x_{i}_{j}_{k}")
                x_ijk[i, j, k] = model.addVar(vtype=cp.COPT.CONTINUOUS, name=f"x_{i}_{j}_{k}")
            w_ik[i] = model.addVar(vtype=cp.COPT.CONTINUOUS, lb=0, name=f"w_{i}")

    # 目标函数：最小化总配送成本
    obj = cp.LinExpr()
    for k in K:
        for i in N:
            for j in N:
                if c_ij[i,j]==float('inf'):
                    obj += cp.COPT.INFINITY * x_ijk[i, j, k]
                else:
                    obj += c_ij[(i, j)] * x_ijk[i, j, k]
    model.setObjective(obj, sense=cp.COPT.MINIMIZE)

    # 约束式(2)：每个客户只能被一辆车服务一次
    for i in C:
        expr = cp.LinExpr()
        for k in K:
            for j in N:
                expr += x_ijk[i, j, k]
        model.addConstr(expr == 1, f"constr2_{i}")

    # 约束式(3)：车辆必须从配送中心 0 点出发
    for k in K:
        expr = cp.LinExpr()
        for j in C:
            expr += x_ijk[0, j, k]
        model.addConstr(expr == 1, f"constr3_{k}")

    # 约束式(4)：流量平衡约束
    for k in K:
        for h in C:
            in_expr = cp.LinExpr()
            out_expr = cp.LinExpr()
            for i in N:
                in_expr += x_ijk[i, h, k]
            for j in N:
                out_expr += x_ijk[h, j, k]
            model.addConstr(in_expr - out_expr == 0, f"constr4_{k}_{j}")

    # 约束式(5)：每辆车都必须停留在配送中心 n+1
    for k in K:
        expr = cp.LinExpr()
        for i in N:
            expr += x_ijk[i, len(g.all_nodes.keys())-1, k]
        model.addConstr(expr == 1, f"constr5_{k}")

    # 约束式(6)：时间约束
    for k in K:
        for i in C:
            for j in C:
                expr = cp.LinExpr()
                expr += w_ik[i] + c_ij[(i, j)] + g.all_nodes[i].servicet - M_ij[(i, j)] * (1 - x_ijk[i, j, k])
                model.addConstr(expr <= w_ik[j], f"constr6_{k}_{i}_{j}")

    # 约束式(7)：时间窗约束
    for k in K:
        for i in C:
            sum_enter = cp.LinExpr()
            for j in N:
                if j == i:
                    sum_enter += x_ijk[i, j, k]
            model.addConstr(g.all_nodes[i].start_tw * sum_enter <= w_ik[i], f"constr7a_{k}_{i}")
            model.addConstr(w_ik[i] <= g.all_nodes[i].end_tw * sum_enter, f"constr7b_{k}_{i}")

    # 约束式(8)：车辆时间约束（配送中心时间）
    for k in K:
        model.addConstr(g.depot_start.start_tw <= w_ik[0], f"constr8a_{k}")
        model.addConstr(w_ik[0] <= g.depot_start.start_tw, f"constr8b_{k}")
        model.addConstr(g.depot_end.start_tw <= w_ik[len(g.all_nodes.keys())-1], f"constr8c_{k}")
        model.addConstr(w_ik[len(g.all_nodes.keys())-1] <= g.depot_end.end_tw, f"constr8d_{k}")

    # 约束式(9)：车辆容量约束
    for k in K:
        expr = cp.LinExpr()
        for i in C:
            for j in N:
                expr += g.all_nodes[i].demand * x_ijk[i, j, k]
        model.addConstr(expr <= Q, f"constr9_{k}")

    # 求解模型
    model.solve()



def create_default_paths(g, paths, lambda_r,K):
    env = cp.Envr() 
    model = env.createModel("RMP_Model")
    for c in list(g.all_customers.values()):
        new_path = [g.depot_start.id, c.id, g.depot_end.id]
        add_new_column(model, Path(new_path, paths, g), lambda_r)

    # 3. 设置目标函数：最小化总成本
    model.setObjective(cp.quicksum(path.cost * lambda_r[path.id] for path in paths), sense=cp.COPT.MINIMIZE)

    # 4. 添加约束（3.7）：每个客户节点恰好被访问一次
    a_ri = np.zeros([len(paths),len(list(g.all_customers.values()))])
    for path in paths:
        for c in list(g.all_customers.values()):
            r = path.id
            # 客户的编号从1开始
            i = c.id-1 
            a_ri[r,i] = path.if_contains_cus(c)
    u = {}
    for c in list(g.all_customers.values()):
        i = c.id-2
        u[c] = model.addConstr(cp.quicksum(a_ri[path.id, i] * lambda_r[path.id] for path in paths) == 1, f"constr_client_{i}")

    # 5. 添加约束（3.8）：车辆数量限制
    constr_vehicle_num = model.addConstr(cp.quicksum(lambda_r[path.id] for path in paths) <= K, "constr_vehicle_num")

    return model,a_ri,constr_vehicle_num,u

def add_new_column(model, path, lambda_r):
    # 2. 添加决策变量
    r = path.id
    lambda_r[r] = model.addVar(vtype=cp.COPT.CONTINUOUS, lb=0, ub =1, name=f"lambda_{r}")



def create_subproblem_model(g, Q, u, sigma):
    N = g.all_nodes.keys()
    C = g.all_customers.keys()
    c_ij = g.distance_matrix  # 弧的成本
    M_ij = {(i, j): 100 for i in N for j in N}  # 大 M 值
    

    # 创建 COPT 环境和模型
    env = cp.Envr()
    model = env.createModel()

    # 定义变量
    x_ij = {}
    s_i = {}
    for i in N:
        for j in N:
            x_ij[i, j] = model.addVar(vtype=cp.COPT.CONTINUOUS, lb=0, name=f"x_{i}_{j}")
        s_i[i] = model.addVar(vtype=cp.COPT.CONTINUOUS, lb=0, name=f"s_{i}")

    # 目标函数
    obj_expr = cp.LinExpr()
    for i in N:
        for j in N:
            if c_ij[i,j]==float('inf'):
                obj_expr += cp.COPT.INFINITY  * x_ij[i, j]
            else:
                obj_expr += (c_ij[i,j] - u[g.all_nodes[i]]) * x_ij[i, j]
                # obj_expr += c_ij[i,j] * x_ij[i, j]
    obj_expr -= sigma
    model.setObjective(obj_expr, sense=cp.COPT.MINIMIZE)

    # 容量约束
    cap_expr = cp.LinExpr()
    for i in C:
        for j in N:
            cap_expr += g.all_customers[i].demand * x_ij[i, j]
    model.addConstr(cap_expr <= Q, "capacity_constraint")


    # 从起点出发约束
    start_expr = cp.LinExpr()
    for j in N:
        start_expr += x_ij[0, j]
    model.addConstr(start_expr == 1, "start_constraint")

    # 流量守恒约束
    for h in C:
        in_expr = cp.LinExpr()
        out_expr = cp.LinExpr()
        for i in N:
            in_expr += x_ij[i, h]
        for j in N:
            out_expr += x_ij[h, j]
        model.addConstr(in_expr == out_expr, f"flow_conservation_{h}")

    # 到达终点约束
    end_expr = cp.LinExpr()
    for i in N:
        end_expr += x_ij[i, max(N)]
    model.addConstr(end_expr == 1, "end_constraint")

    # 时间约束
    for i in N:
        for j in N:
            time_expr = cp.LinExpr()
            if c_ij[i,j]==float('inf'):
                time_expr += s_i[i] + cp.COPT.INFINITY - s_i[j]
            else:
                time_expr += s_i[i] + c_ij[(i, j)] - s_i[j]
            
            model.addConstr(time_expr <= M_ij[(i, j)] * (1 - x_ij[i, j]), f"time_constraint_{i}_{j}")

    # 时间窗约束
    for i in N:
        model.addConstr(s_i[i] >= g.all_nodes[i].start_tw, f"time_window_lower_{i}")
        model.addConstr(s_i[i] <= g.all_nodes[i].end_tw, f"time_window_upper_{i}")

    # 节点自身约束
    for i in N:
        model.addConstr(x_ij[i, i] == 0, f"self_loop_constraint_{i}")

    # 求解模型
    model.solve()

    # 输出结果
    if model.status == cp.COPT.OPTIMAL:
        print("最优解找到，目标值为:", model.getObjVal())
        print("最优路径选择：")
        for i in N:
            for j in N:
                if model.getValue(x_ij[i, j]) > 0.5:
                    print(f"弧 ({i}, {j}) 被选中")
    else:
        print("未找到最优解，状态码:", model.status)

    
class Label:
    def __init__(self):
        self.best_cost = 0
        self.time_consumption = 0
        self.capacity_consumption = 0
        self.vis = False
        self.path = []

    def __init__(self, bc, tc, cc, path):
        self.best_cost = bc
        self.time_consumption = tc
        self.capacity_consumption = cc
        self.path = path
        self.vis = True


class Espprc:
    def __init__(self,g, u, sigma, step, lt, ut, max_capacity):
        self.g = g
        self.n = len(g.all_nodes)
        self.start = g.depot_start.id
        self.end = g.depot_end.id
        self.step = step
        self.lower_time = lt
        self.upper_time = ut
        self.max_capacity = max_capacity
        self.u_ = u
        self.sigma_ = sigma
        self.lower_bound_matrix = [[Label(math.inf, 0, 0, []) for _ in range(self.n + 10)] for _ in range((ut - lt) // step + 10)]
        self.naive_dual_bound = math.inf
        self.overall_best_cost = 0.0
        self.primal_bound = 0.0
        self.is_visited = [False] * len(g.all_nodes)

    def reduced_cost(self, path):
        if not path:
            return 0.0
        total_cost = 0.0
        for i in range(len(path) - 1):
            current_node = self.g.all_nodes[path[i]]
            next_node = self.g.all_nodes[path[i+1]]
            travel_time = current_node.time_to_node(next_node)
            total_cost += travel_time
            
            if current_node.id != self.g.depot_start.id:
                total_cost -= self.u_.get(current_node, 0.0)
        
        total_cost -= self.sigma_
        return total_cost

    def capacity_consumption(self, path):
        total_demand = 0.0
        for i in range(1, len(path.customers)):
            total_demand += path.customers[i].demand
        return total_demand

    def time_consumption(self, path):
        if not path:
            return 0.0
        total_time = 0.0
        for i in range(len(path) - 1):
            total_time += path.customers[i].time_to_node(path.customers[i+1]) + path.customers[i].servicet
            total_time = max(total_time, path.customers[i+1].start_tw)
        return total_time

    def cal_naive_dual_bound(self):
        valid_mask = (self.g.distance_matrix != 0) & ~np.isinf(self.g.distance_matrix)
        valid_numbers = self.g.distance_matrix[valid_mask]
        if valid_numbers.size > 0:
            self.naive_dual_bound = np.min(valid_numbers/valid_numbers)

    def bound_order(self):
        self.bound_generation_order = list(range(1, self.n))

    def is_feasible(self, cur: int, capacity: float, time: float) -> bool:
        if self.is_visited[cur]:
            return False
        
        node = self.g.all_nodes[cur]
        if (capacity + node.get_demand() > self.max_capacity or
            time > node.get_end_tw()):
            return False
        
        return True

    def check_bounds(self, root, cur, time, cost, flag):
        lower_bound = 0
        err = 1e-9
        if time < self.time_incumbent + self.step:
            diff_time = self.time_incumbent + self.step - time
            if diff_time > 0:
                lower_bound = diff_time * self.naive_dual_bound + self.overall_best_cost
            else:
                print("Contradictory!")
        else:
            try:
                lb = self.lower_bound_matrix[(self.upper_time - int(time)) // self.step][cur]
                if not lb.vis:
                    return True
                lower_bound = lb.best_cost
                if abs(lower_bound - math.inf) < err:
                    return False
            except IndexError:
                print("Out of range error!")
        best_cost = self.primal_bound if flag else self.g.all_nodes[root].best_cost
        if cost + lower_bound >= best_cost:
            return False
        return True

    def rollback(self, cur, cost, path):
        if len(path) < 2 or self.get_edge(path[-2], cur) == -1:
            return True
        alt_path = path[:-1] + [cur]
        if cost >= self.reduced_cost(alt_path):
            return False
        return True

    def intersection(self, path1, path2):
        set1 = set(path1)
        set2 = set(path2)
        return len(set1.intersection(set2)) == 0

    def concat(self, root, cur, time, cost, capacity, path, flag):
        ix = (self.upper_time - int(time)) // self.step
        lb = self.lower_bound_matrix[ix][cur] if time >= self.time_incumbent + self.step else Label(math.inf, 0, 0, [])
        if lb.vis and ix > 0 and capacity + lb.capacity_consumption <= self.max_capacity and \
                self.intersection(path, lb.path) and lb.best_cost == self.lower_bound_matrix[ix - 1][cur].best_cost:
            if flag:
                self.primal_bound = cost + lb.best_cost
            else:
                self.g.all_nodes[root].best_cost = cost + lb.best_cost
            path.extend(lb.path)
            return True
        return False

    def dynamic_update(self, cur, opt_path):
        start_index = opt_path.index(cur)
        path_cost = self.reduced_cost(opt_path[start_index:])
        if path_cost < self.g.all_nodes[cur].best_cost:
            self.g.all_nodes[cur].best_cost = path_cost

    def pulse_procedure(self, root, cur, cost, capacity, time, path, flag):
        time = max(time, self.g.all_nodes[cur].start_tw)
        if not self.is_feasible(cur, capacity, time) or not self.check_bounds(root, cur, time, cost, flag) or not self.rollback(cur, cost, path):
            return
        if not self.concat(root, cur, time, cost, capacity, path, flag):
            opt_path = []
            path.append(cur)
            nx_cost = 0.0
            nx_capacity = capacity + self.g.all_nodes[cur].demand
            nx_time = 0.0
            for successor in np.argsort(self.g.distance_matrix[cur,:]):
                new_path = path.copy()
                nx_cost = cost + self.g.distance_matrix[cur,successor].cost
                nx_time = max(self.g.all_nodes[successor].twl, time + self.g.all_nodes[cur].service_time + self.G['edge_list'][edge_ix].time)
                if self.g.all_nodes[successor].label:
                    self.g.all_nodes[successor].label = False
                    self.pulse_procedure(root, successor, nx_cost, nx_capacity, nx_time, new_path, flag)
                    self.g.all_nodes[successor].label = True
                if new_path and new_path[-1] == self.end and (not opt_path or self.reduced_cost(new_path) < self.reduced_cost(opt_path)):
                    opt_path = new_path
                    self.dynamic_update(cur, opt_path)
            if path and path[-1] != self.end:
                path = opt_path
        if path and path[-1] == self.end:
            tmp = self.reduced_cost(path)
            self.g.all_nodes[root].best_cost = min(self.g.all_nodes[root].best_cost, tmp)
            if tmp < self.primal_bound:
                self.primal_bound = tmp

    def bounding_scheme(self):
        self.bound_order()
        self.cal_naive_dual_bound()
        bound_index = 0
        self.time_incumbent = self.upper_time - self.step
        pre_path = [[] for _ in range(self.n)]
        while self.time_incumbent >= self.lower_time:
            for root in self.bound_generation_order:
                for node in list(self.g.all_nodes.values()):
                    node.label = True
                path = []
                self.g.all_nodes[root].label = False
                self.pulse_procedure(root, root, 0.0, 0.0, self.time_incumbent, path, False)
                if pre_path[root] and not path:
                    path = pre_path[root]
                self.lower_bound_matrix[bound_index][root] = Label(
                    self.g.all_nodes[root].best_cost,
                    self.time_consumption(path),
                    self.capacity_consumption(path),
                    path
                )
                pre_path[root] = path
            self.overall_best_cost = self.primal_bound
            self.time_incumbent -= self.step
            bound_index += 1

    def espprc(self):
        self.bounding_scheme()
        for node in self.g.all_nodes.values():
            node.label = True
        self.g.all_nodes[self.start].label = False
        self.primal_bound = math.inf
        opt_path = []
        self.pulse_procedure(self.start, self.start, 0.0, 0.0, 0.0, opt_path, True)
        print(f"min cost: {self.reduced_cost(opt_path)}")
        print("optimal path:", end=" ")
        for i in opt_path:
            print(i, end=" ")
        print()

    def print_para(self):
        print(f"n,start,end,step,lt,ut,max_capacity: {self.n} {self.start} {self.end} {self.step} {self.lower_time} {self.upper_time} {self.max_capacity}")

class SubProblem_Pulse:
    def __init__(self, u_: Dict[Customer, float],sigma_:float, g: Graph, instance: Instance, 
                 lower_time: float, upper_time: float, step: float):
        self.u_ = u_
        self.sigma_ = sigma_
        self.g = g
        self.instance = instance
        self.lower_time = lower_time
        self.upper_time = upper_time
        self.step = step
        
        # 初始化下界矩阵和状态数组
        self.lower_bound_matrix = [
            [ -math.inf for _ in range(int((upper_time - lower_time) / step) + 1) ]
            for _ in range(len(g.all_nodes))
        ]
        self.is_visited = [False] * len(g.all_nodes)
        self.best_cost_cus = [math.inf] * len(g.all_nodes)
        self.time_incumbent = upper_time - step
        self.obj_value = 0.0

    def bounding_scheme(self):
        bound_index = 0
        self.is_visited[self.g.depot_start.id] = True
        
        while self.time_incumbent >= self.lower_time:
            for root in self.g.all_customers.keys():
                path = []
                self.pulse_procedure(root, root, 0.0, 0.0, self.time_incumbent, path)
                self.lower_bound_matrix[root][bound_index] = self.best_cost_cus[root]
            
            self.time_incumbent -= self.step
            bound_index += 1
        
        self.is_visited[self.g.depot_start.id] = False

    def reduced_cost(self, path: List[int]) -> float:
        if not path or len(path) == 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            current_node = self.g.all_nodes[path[i]]
            next_node = self.g.all_nodes[path[i+1]]
            travel_time = current_node.time_to_node(next_node)
            total_cost += travel_time
            
            if current_node.id != self.g.depot_start.id:
                total_cost -= self.u_.get(current_node, 0.0)
        
        total_cost -= self.sigma_

        return total_cost

    def is_feasible(self, cur: int, capacity: float, time: float) -> bool:
        if self.is_visited[cur]:
            return False
        
        node = self.g.all_nodes[cur]
        if (capacity + node.get_demand() > self.instance.max_capacity or
            time > node.get_end_tw()):
            return False
        
        return True

    def check_bounds(self, root: int, cur: int, time: float, cost: float) -> bool:
        idx = int((self.upper_time - time) / self.step)
        lower_bound = self.lower_bound_matrix[cur][idx] if idx >=0 else -math.inf
        if cost + lower_bound >= self.best_cost_cus[root]:
            return False
        return True

    def rollback(self, cur: int, cost: float, path: List[int]) -> bool:
        if len(path) < 2:
            return True
        
        alt_path = path.copy()
        alt_path.pop()
        alt_path.append(cur)
        alt_cost = self.reduced_cost(alt_path)
        
        if cost >= alt_cost:
            return False
        return True

    def pulse_procedure(self, root: int, cur: int, cost: float, capacity: float, 
                        time: float, path: List[int]):
        node = self.g.all_nodes[cur]
        if time < node.get_start_tw():
            time = node.get_start_tw()
        
        if not self.is_feasible(cur, capacity, time) or \
           not self.check_bounds(root, cur, time, cost) or \
           not self.rollback(cur, cost, path):
            return
        
        new_path = path.copy()
        new_path.append(cur)
        self.is_visited[cur] = True
        
        nx_cost = 0.0
        nx_capacity = capacity + node.get_demand()
        nx_time = 0.0
        opt_path = []
        
        if cur != self.g.depot_end.id:
            for nx in self.g.all_nodes:
                if nx == cur:
                    continue
                
                next_node = self.g.all_nodes[nx]
                travel_time = node.time_to_node(next_node)
                
                if node.id == self.g.depot_start.id:
                    nx_cost = travel_time
                else:
                    nx_cost = cost + travel_time - self.u_.get(node, 0.0)
                
                service_time = node.time_at_node()
                nx_time = max(next_node.get_start_tw(), time + service_time + travel_time)
                
                if not self.is_visited[nx]:
                    self.pulse_procedure(root, nx, nx_cost, nx_capacity, nx_time, new_path)
                
                # 更新最优路径
                if new_path and new_path[-1] == self.g.depot_end.id:
                    current_reduced_cost = self.reduced_cost(new_path)
                    if not opt_path or current_reduced_cost < self.reduced_cost(opt_path):
                        opt_path = new_path.copy()
        
        # 处理 depot_end 的情况
        if new_path and new_path[-1] != self.g.depot_end.id:
            new_path = opt_path
        
        if new_path and new_path[-1] == self.g.depot_end.id:
            tmp_cost = self.reduced_cost(new_path)
            if tmp_cost < self.best_cost_cus[root]:
                self.best_cost_cus[root] = tmp_cost
        
        self.is_visited[cur] = False

    def run_pulse_algorithm(self) -> List[int]:
        self.bounding_scheme()
        opt_path = []
        self.pulse_procedure(self.g.depot_start.id, self.g.depot_start.id, 
                            0.0, 0.0, 0.0, opt_path)
        self.obj_value = self.reduced_cost(opt_path)
        return opt_path

    
if __name__ == "__main__":

    instance = Instance("input/Solomon/25_customer/c101.txt")
    g = instance.read_data_from_file()

    paths = []
    lambda_r = {}
    K = 25

    model,a_ri,constr_vehicle_num,u= create_default_paths(g,paths,lambda_r,K)
    # 求解模型
    model.solveLP()

    # print(model.getLpSolution())
    for key,constr in u.items():
        u[key] = model.getInfo(cp.COPT.Info.Dual, constr)
    u[g.depot_end] = u[g.depot_start] = 0.0
    sigma = model.getInfo(cp.COPT.Info.Dual, constr_vehicle_num)
    create_subproblem_model(g, instance.max_capacity, u, sigma)
    solve_cvrptw(g,instance.max_capacity,K)
    # boundStep = 4
    # subproblem = SubProblem_Pulse(
    #             u,
    #             sigma,
    #             g,
    #             instance,
    #             g.depot_start.start_tw,
    #             g.depot_start.end_tw,
    #             (g.depot_start.end_tw - g.depot_start.start_tw) / boundStep
    #         )
    # path = subproblem.run_pulse_algorithm()
    # print(path)
    # step = 10
    # lt = g.depot_start.start_tw
    # ut = g.depot_start.end_tw
    # max_capacity = instance.max_capacity
    # Espprc(g, u, sigma, step, lt, ut, max_capacity).espprc()
    
    