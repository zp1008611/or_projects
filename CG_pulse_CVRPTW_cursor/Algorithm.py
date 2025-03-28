import time
import coptpy as copt
from VRPTW import Customer, Graph, Path, Instance
from Parameters import Parameters
import sys
import math
from typing import List, Dict

class ColumnGen:
    def __init__(self, instance):
        self.instance = Instance(instance)
        self.g = self.instance.read_data_from_file()
        self.paths = []
        self.masterproblem = MasterProblem(self.g, self.paths)

    def run_column_generation(self):
        iteration_counter = 0
        start = time.time()
        while True:
            iteration_counter += 1
            self.masterproblem.solve_relaxation()
            self.subproblem = SubProblemPulse(
                self.masterproblem.lambda_,
                self.g,
                self.instance,
                self.g.depot_start.start_tw,
                self.g.depot_start.end_tw,
                (self.g.depot_start.end_tw - self.g.depot_start.start_tw) / Parameters.ColGen.boundStep
            )
            path = self.subproblem.run_pulse_algorithm()
            self.masterproblem.add_new_column(Path(path, self.paths, self.g))
            self.display_iteration(iteration_counter)
            if self.subproblem.obj_value >= Parameters.ColGen.zero_reduced_cost_AbortColGen:
                break

        self.masterproblem.solve_mip()
        self.masterproblem.display_solution()
        end = time.time()
        print(f"Time used: {end - start}s")

    def display_iteration(self, iter):
        if iter % 20 == 0 or iter == 1:
            print()
            print("Iteration", end="")
            print("   nPaths", end="")
            print("       MP lb", end="")
            print("      SB int")
        print(f"{iter:9.0f}", end="")
        print(f"{len(self.paths):9.0f}", end="")
        print(f"{self.masterproblem.lastObjValue:15.2f}", end="")
        print(f"{self.subproblem.obj_value:12.4f}")





class MasterProblem:
    def __init__(self, g, paths):
        self.g = g
        self.paths = paths
        self.create_model()
        self.create_default_paths()
        Parameters.configure_copt(self)

    def create_model(self):
        self.model = copt.Envr().createModel()
        self.model.setObjectiveSense(copt.COPT.MINIMIZE)
        self.row_customers = {}
        self.lambda_ = {}
        self.mip_conversion = []

        for customer in self.g.all_customers.values():
            lin_expr = self.model.LinExpr()
            constr = self.model.addConstr(lin_expr == 1, name=f"cust {customer.id}")
            self.row_customers[customer] = constr

    def create_default_paths(self):
        for c in self.g.all_customers.values():
            new_path = [self.g.depot_start.id, c.id, self.g.depot_end.id]
            self.add_new_column(Path(new_path, self.paths, self.g))

    def add_new_column(self, path):
        col_coeff = [path.cost]
        col_indices = []
        col_values = []
        for c in self.g.all_customers.values():
            col_indices.append(self.row_customers[c])
            col_values.append(path.if_contains_cus(c))

        # 创建列表达式
        col_expr = copt.ColExpr()
        for index, value in zip(col_indices, col_values):
            col_expr += value * index

        # 添加新变量
        var = self.model.addVar(obj=col_coeff[0], lb=0, ub=1, name=f"theta.{path.id}", col=col_expr)
        path.theta = var

    def save_dual_values(self):
        for c in self.g.all_customers.values():
            self.lambda_[c] = self.row_customers[c].getDual()

    def solve_relaxation(self):
        try:
            self.model.solve()
            if self.model.getStatus() == copt.COPT.OPTIMAL:
                self.save_dual_values()
                self.lastObjValue = self.model.getObjVal()
        except copt.CoptError as e:
            print("COPT exception caught: ", e)

    def convert_to_mip(self):
        for path in self.paths:
            path.theta.setType(copt.COPT.BINARY)

    def solve_mip(self):
        try:
            self.convert_to_mip()
            self.model.solve()
            if self.model.getStatus() == copt.COPT.OPTIMAL:
                self.display_solution()
            else:
                print("Integer solution not found")
        except copt.CoptError as e:
            print("COPT exception caught: ", e)

    def display_solution(self):
        try:
            total_cost = 0
            print("\n" + "--- Solution >>> ------------------------------")
            for path in self.paths:
                if path.theta.getVal() > 0.99999:
                    total_cost += path.display_info()
            print("Total cost = ", total_cost)
            print("\n" + "--- Solution <<< ------------------------------")
        except copt.CoptError as e:
            print("COPT exception caught: ", e)


class SubProblemPulse:
    def __init__(self, lambda_: Dict[Customer, float], g: Graph, instance: Instance, 
                 lower_time: float, upper_time: float, step: float):
        self.lambda_ = lambda_
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
                total_cost -= self.lambda_.get(current_node, 0.0)
        
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
        if not self.is_feasible(cur, capacity, time):
            return

        if cur == self.g.depot_end.id:
            if cost < self.best_cost_cus[root]:
                self.best_cost_cus[root] = cost
            return

        if not self.check_bounds(root, cur, time, cost):
            return

        if not self.rollback(cur, cost, path):
            return

        self.is_visited[cur] = True
        path.append(cur)

        for next_node in self.g.all_nodes.values():
            if next_node.id != cur:
                travel_time = node.time_to_node(next_node)
                self.pulse_procedure(root, next_node.id, cost + travel_time,
                                   capacity + node.get_demand(), time + travel_time + node.time_at_node(),
                                   path)

        path.pop()
        self.is_visited[cur] = False

    def run_pulse_algorithm(self) -> List[int]:
        self.bounding_scheme()
        best_path = []
        best_cost = math.inf

        for root in self.g.all_customers.keys():
            path = []
            self.pulse_procedure(root, root, 0.0, 0.0, self.time_incumbent, path)
            if self.best_cost_cus[root] < best_cost:
                best_cost = self.best_cost_cus[root]
                best_path = path

        self.obj_value = best_cost
        return best_path