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
        self.g = self.instance.ReadDataFromFile()
        self.paths = []
        self.masterproblem = MasterProblem(self.g, self.paths)

    def runColumnGeneration(self):
        iteration_counter = 0
        start = time.time()
        while True:
            iteration_counter += 1
            self.masterproblem.solveRelaxation()
            self.subproblem = SubProblem_Pulse(
                self.masterproblem.lambda_,
                self.g,
                self.instance,
                self.g.depot_start.startTw,
                self.g.depot_start.endTw,
                (self.g.depot_start.endTw - self.g.depot_start.startTw) / Parameters.ColGen.boundStep
            )
            path = self.subproblem.runPulseAlgorithm()
            self.masterproblem.addNewColumn(Path(path, self.paths, self.g))
            self.displayIteration(iteration_counter)
            if self.subproblem.objValue >= Parameters.ColGen.zero_reduced_cost_AbortColGen:
                break

        # self.masterproblem.solveMIP()
        self.masterproblem.solveRelaxation()
        self.masterproblem.displaySolution()
        end = time.time()
        print(f"Time used: {end - start}s")

    def displayIteration(self, iter):
        if iter % 20 == 0 or iter == 1:
            print()
            print("Iteration", end="")
            print("   nPaths", end="")
            print("       MP lb", end="")
            print("      SB int")
        print(f"{iter:9.0f}", end="")
        print(f"{len(self.paths):9.0f}", end="")
        print(f"{self.masterproblem.lastObjValue:15.2f}", end="")
        print(f"{self.subproblem.objValue:12.4f}")





class MasterProblem:
    def __init__(self, g, paths):
        self.g = g
        self.paths = paths
        self.create_model()
        self.create_default_paths()
        Parameters.configure_copt(self)

    def create_model(self):
        # self.cplex = cplex.Cplex()
        self.model = copt.Envr().createModel()
        # self.cplex.objective.set_sense(self.cplex.objective.sense.minimize)
        self.model.setObjectiveSense(copt.MINIMIZE)
        self.row_customers = {}
        self.lambda_ = {}
        self.mipConversion = []

        for customer in self.g.all_customers.values():
            # self.row_customers[customer] = self.cplex.linear_constraints.add(
            #     lin_expr=[cplex.SparsePair(ind=[], val=[])],
            #     senses=["E"],
            #     rhs=[1],
            #     range_values=[0],
            #     names=["cust " + str(customer.id)]
            # )
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

        # new_col = self.cplex.SparsePair(ind=col_indices, val=col_values)
        # self.cplex.variables.add(obj=col_coeff, lb=[0], ub=[1], names=["theta." + str(path.id)], columns=[new_col])
        # path.theta = self.cplex.variables.get_numvar_by_name("theta." + str(path.id))
        vars = [self.model.addVar() for _ in range(max(col_indices) + 1)]
        # 创建列表达式
        col_expr = copt.ColExpr()
        for index, value in zip(col_indices, col_values):
            col_expr += value * vars[index]
        # 添加新变量
        self.model.addVar(obj=col_coeff, lb=0, ub=1, name=f"theta.{path.id}", col=col_expr)
        self.model.update()
        # 获取变量引用
        path.theta = self.model.getVarByName(f"theta.{path.id}")

    def save_dual_values(self):
        for c in self.g.all_customers.values():
            self.lambda_[c] = self.cplex.solution.get_dual_values(self.row_customers[c])

    def solve_relaxation(self):
        # try:
        #     self.cplex.solve()
        #     if self.cplex.solution.get_status() == self.cplex.solution.status.optimal:
        #         self.save_dual_values()
        #         self.lastObjValue = self.cplex.solution.get_objective_value()
        # except cplex.exceptions.CplexError as e:
        #     print("CPLEX exception caught: ", e)

        try:
            self.model.solve()
            if self.model.getStatus() == copt.COPT.OPTIMAL:
                self.save_dual_values()
                self.lastObjValue = self.model.getObjVal()
        except copt.CoptError as e:
            print("COPT exception caught: ", e)



    def convert_to_mip(self):
        for path in self.paths:
            var_index = self.cplex.variables.get_numvar_index("theta." + str(path.id))
            self.cplex.variables.set_types(var_index, self.cplex.variables.type.binary)

    def solve_mip(self):
        try:
            self.convert_to_mip()
            self.cplex.solve()
            if self.cplex.solution.get_status() == self.cplex.solution.status.optimal:
                self.display_solution()
            else:
                print("Integer solution not found")
        except cplex.exceptions.CplexError as e:
            print("CPLEX exception caught: ", e)

    def display_solution(self):
        try:
            total_cost = 0
            print("\n" + "--- Solution >>> ------------------------------")
            for path in self.paths:
                if self.cplex.solution.get_values(path.theta) > 0.99999:
                    total_cost += path.display_info()
            print("Total cost = ", total_cost)
            print("\n" + "--- Solution <<< ------------------------------")
        except cplex.exceptions.CplexError as e:
            print("CPLEX exception caught: ", e)


class SubProblem_Pulse:
    def __init__(self, u_: Dict[Customer, float],sigma_:float, g: Graph, instance: Instance, 
                 lower_time: float, upper_time: float, step: float):
        self.lambda_ = u_
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
        if (capacity + node.getDemand() > self.instance.max_capacity or
            time > node.getEndTw()):
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
        if time < node.getStartTw():
            time = node.getStartTw()
        
        if not self.is_feasible(cur, capacity, time) or \
           not self.check_bounds(root, cur, time, cost) or \
           not self.rollback(cur, cost, path):
            return
        
        new_path = path.copy()
        new_path.append(cur)
        self.is_visited[cur] = True
        
        nx_cost = 0.0
        nx_capacity = capacity + node.getDemand()
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
                nx_time = max(next_node.getStartTw(), time + service_time + travel_time)
                
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