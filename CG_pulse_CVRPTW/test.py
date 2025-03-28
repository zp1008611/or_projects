import coptpy as cp
# from pyomo.environ import *
# from pyomo.opt import SolverFactory  
from VRPTW import Customer, Graph, Path, Instance
import numpy as np

instance = Instance("input/Solomon/25_customer/c101.txt")
g = instance.read_data_from_file()

paths = []
K = 1000

model = cp.Model("VRP_SP")
objective = 0
lambda_r = {}



# def create_default_paths():
#     for c in g.all_customers.values():
#         new_path = [g.depot_start.id, c.id, g.depot_end.id]
#         add_new_column(model, Path(new_path, paths, g))

# def add_new_column(model, path):

#     # 2. 添加决策变量
#     lambda_r[path.id] = model.add_var(vtype=cp.BINARY, name=f"lambda_{path.id}")

#     # 3. 设置目标函数：最小化总成本
#     model.set_objective(objective + path.cost * lambda_r[path.id], sense=cp.MINIMIZE)

#     # 4. 添加约束（3.2）：每个客户节点恰好被访问一次
#     for i in g.all_customers.values():
        
#         model.add_constr(cp.quicksum(a_ri[(r, i)] * lambda_r[r] for r in R) == 1, f"constr_client_{i}")

#     # 5. 添加约束（3.3）：车辆数量限制
#     model.add_constr(cp.quicksum(lambda_r[r] for r in R) <= K, "constr_vehicle_num")

#     # 6. 求解模型
#     model.solve()

# def add_new_column(self, path):
#     col_coeff = [path.cost]
#     col_indices = []
#     col_values = []
#     for c in g.all_customers.values():
#         col_indices.append(self.row_customers[c])
#         col_values.append(path.if_contains_cus(c))

#     # new_col = self.cplex.SparsePair(ind=col_indices, val=col_values)
#     # self.cplex.variables.add(obj=col_coeff, lb=[0], ub=[1], names=["theta." + str(path.id)], columns=[new_col])
#     # path.theta = self.cplex.variables.get_numvar_by_name("theta." + str(path.id))
#     vars = [self.model.addVar() for _ in range(max(col_indices) + 1)]
#     # 创建列表达式
#     col_expr = copt.ColExpr()
#     for index, value in zip(col_indices, col_values):
#         col_expr += value * vars[index]
#     # 添加新变量
#     self.model.addVar(obj=col_coeff, lb=0, ub=1, name=f"theta.{path.id}", col=col_expr)
#     self.model.update()
#     # 获取变量引用
#     path.theta = self.model.getVarByName(f"theta.{path.id}")


# def create_default_paths():
#     for c in g.all_customers.values():
#         new_path = [g.depot_start.id, c.id, g.depot_end.id]
#         add_new_column(Path(new_path, paths, g))


# # 假设已有问题参数
# R = [...]  # 路径集合（需定义具体路径标识）
# C = [...]  # 客户节点集合
# K = ...   # 车辆最大数量
# c_r = {r: ... for r in R}  # 各路径成本
# a_ri = {(r, i): ... for r in R for i in C}  # 路径-客户关联矩阵



# # 7. 输出结果
# if model.status == cp.OPTIMAL:
#     print("最优解：")
#     for r in R:
#         if lambda_r[r].x > 0.5:
#             print(f"选择路径 {r}，变量值：{lambda_r[r].x}")
#     print(f"最小总成本：{model.obj_val}")
# else:
#     print("未找到最优解，状态：", model.status)