import numpy as np
from coptpy import *

class CVRPModel:
    def __init__(self, distances, demands, capacity, n_vehicles):
        """
        初始化CVRP模型
        
        Args:
            distances: 距离矩阵 (n+1) x (n+1)，包含depot
            demands: 客户需求量数组
            capacity: 车辆容量
            n_vehicles: 可用车辆数量
        """
        self.distances = distances
        self.demands = demands
        self.capacity = capacity
        self.n_vehicles = n_vehicles
        self.n_customers = len(demands)
        
        # 创建COPT环境
        self.env = Envr()
        self.model = self.env.createModel("CVRP")
        
    def create_restricted_master_problem(self, initial_routes):
        """
        创建受限主问题(RMP)
        
        Args:
            initial_routes: 初始可行路径列表
        """
        # 清空模型
        self.model.clear()
        
        # 添加变量
        self.lambda_vars = []
        for r in initial_routes:
            var = self.model.addVar(name=f"lambda_{r}")
            self.lambda_vars.append(var)
            
        # 添加约束
        # 1. 每个客户必须被访问一次
        for i in range(self.n_customers):
            constr = self.model.addConstr(
                quicksum(self.lambda_vars[j] for j, r in enumerate(initial_routes) if i in r) == 1,
                name=f"customer_{i}"
            )
            
        # 2. 车辆数量限制
        self.model.addConstr(
            quicksum(self.lambda_vars) <= self.n_vehicles,
            name="vehicle_limit"
        )
        
        # 设置目标函数
        self.model.setObjective(
            quicksum(self.lambda_vars[i] * self._calculate_route_cost(r) 
                    for i, r in enumerate(initial_routes)),
            COPT.MINIMIZE
        )
        
    def _calculate_route_cost(self, route):
        """计算路径成本"""
        cost = 0
        for i in range(len(route)-1):
            cost += self.distances[route[i], route[i+1]]
        return cost
        
    def solve_rmp(self):
        """求解受限主问题"""
        self.model.solve()
        
        # 检查求解状态
        if self.model.status == COPT.INFEASIBLE:
            raise ValueError("受限主问题不可行，请检查初始路径集合")
            
        # 只有在模型可行时才获取对偶值
        if self.model.status == COPT.OPTIMAL:
            return {
                'objective': self.model.objval,
                'dual_values': [c.pi for c in self.model.getConstrs()]
            }
        else:
            raise ValueError(f"模型求解失败，状态码: {self.model.status}")
        
    def get_dual_values(self):
        """获取对偶变量值"""
        return [c.pi for c in self.model.getConstrs()] 