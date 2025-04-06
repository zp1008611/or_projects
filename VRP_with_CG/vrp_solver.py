import numpy as np
import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time
import networkx as nx
from cspy import BiDirectional
import os
import logging
from typing import Dict, List, Tuple, Any, Union
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VRPData:
    """存储VRP问题数据的类"""
    nodes: pd.DataFrame
    distance_matrix: np.ndarray
    capacity: int
    depot: int

@dataclass
class Route:
    """存储路径信息的类"""
    id: str
    cost: float
    route: List[int]
    type: str

class VRPSolver:
    """VRP求解器主类"""
    
    def __init__(self, data_file: str):
        """初始化VRP求解器"""
        self.data = self._preprocess_cvrp(data_file)
        self.routes = {}
        self.vehicle_types = ["TypeI", "TypeII"]
        self.customers = list(range(2, len(self.data.nodes) + 1))
        self.a_ir = {t: {} for t in self.vehicle_types}
        self.route_costs = {}
        self.lambda_vars = {}
        
    def _preprocess_cvrp(self, data_file: str) -> VRPData:
        """预处理CVRP数据"""
        logger.info(f"开始预处理数据文件: {data_file}")
        with open(data_file, "r") as file:
            lines = file.readlines()
        
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
                depot = int(line)
        
        nodes_df = pd.DataFrame(node_coord_section, columns=["Node", "X", "Y"])
        demands_df = pd.DataFrame(demand_section, columns=["Node", "Demand"])
        
        num_nodes = len(nodes_df)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    x1, y1 = nodes_df.loc[i, ["X", "Y"]]
                    x2, y2 = nodes_df.loc[j, ["X", "Y"]]
                    distance_matrix[i, j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        nodes_df = nodes_df.merge(demands_df, on="Node")
        return VRPData(nodes=nodes_df, distance_matrix=distance_matrix, capacity=capacity, depot=depot)
    
    def create_initial_feasible_solution(self) -> Dict[str, Route]:
        """创建初始可行解"""
        logger.info("开始创建初始可行解")
        customer_nodes = list(self.data.nodes["Node"])
        odd_customers = [node for node in customer_nodes if node != self.data.depot and node % 2 != 0]
        even_customers = [node for node in customer_nodes if node != self.data.depot and node % 2 == 0]
        
        def generate_routes(customer_set: List[int], route_type: str) -> Dict[str, Route]:
            routes = {}
            for customer in customer_set:
                route_id = f"Route_{customer}"
                route_cost = 2 * self.data.distance_matrix[self.data.depot - 1][customer - 1]
                route_demand = self.data.nodes.loc[customer - 1, "Demand"]
                routes[route_id] = Route(
                    id=route_id,
                    cost=route_cost,
                    route=[self.data.depot, customer, self.data.depot],
                    type=route_type
                )
            return routes
        
        initial_routes_typeI = generate_routes(odd_customers, "TypeI")
        initial_routes_typeII = generate_routes(even_customers, "TypeII")
        return {**initial_routes_typeI, **initial_routes_typeII}
    
    def initialize_master_problem(self) -> Tuple[gp.Model, Dict[int, float]]:
        """初始化主问题"""
        logger.info("初始化主问题")
        master = gp.Model("MasterProblem")
        master.setParam('OutputFlag', 0)
        
        # 添加决策变量
        for route_id, route in self.routes.items():
            self.lambda_vars[(route_id, route.type)] = master.addVar(
                obj=route.cost,
                vtype=GRB.CONTINUOUS,
                name=f"lambda_{route_id}_{route.type}",
                lb=0,
                ub=1
            )
        
        # 添加客户覆盖约束
        customer_constraints = {}
        for customer in self.customers:
            customer_constraints[customer] = master.addConstr(
                gp.quicksum(
                    self.lambda_vars[(route_id, route.type)] * 
                    (1 if customer in route.route else 0)
                    for route_id, route in self.routes.items()
                ) == 1,
                name=f"{customer}"
            )
        
        master.optimize()
        
        if master.Status == GRB.OPTIMAL:
            dual_prices = {int(constr.ConstrName): constr.Pi for constr in master.getConstrs()}
            return master, dual_prices
        else:
            raise Exception("主问题初始化失败")
    
    def solve(self, time_limit: int = 300) -> Dict[str, Any]:
        """求解VRP问题"""
        logger.info("开始求解VRP问题")
        start_time = time.time()
        
        # 创建初始解
        self.routes = self.create_initial_feasible_solution()
        
        # 初始化主问题
        master, dual_prices = self.initialize_master_problem()
        
        # 运行列生成
        final_obj_val, total_pricing_time, total_master_time, final_gap = self._run_column_generation(
            master=master,
            dual_prices=dual_prices,
            time_limit=time_limit
        )
        
        # 准备结果
        result = {
            "Instance": os.path.basename(self.data_file),
            "Root Node LP Objective": master.ObjVal,
            "Best PnB IP Solution": final_obj_val,
            "GAP": final_gap,
            "Time for Column Generation": total_pricing_time,
            "Time for IP": total_master_time,
            "Total Time": time.time() - start_time
        }
        
        logger.info(f"求解完成，总耗时: {result['Total Time']:.2f}秒")
        return result
    
    def _run_column_generation(self, master: gp.Model, dual_prices: Dict[int, float], time_limit: int) -> Tuple[float, float, float, float]:
        """运行列生成过程"""
        # 实现列生成逻辑
        pass

def main():
    """主函数"""
    results = []
    data_dir = "./Uchoa-Vrp-Set-X"
    
    for root, _, files in os.walk(data_dir):
        files = sorted(files)
        for file in files:
            if file.endswith(".vrp"):
                logger.info(f"处理实例: {file}")
                file_path = os.path.join(root, file)
                try:
                    solver = VRPSolver(file_path)
                    result = solver.solve()
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理实例 {file} 时出错: {str(e)}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv("cvrp_results.csv", index=False)
    logger.info("所有实例处理完成")
    return results_df

if __name__ == "__main__":
    results_df = main()
    print(results_df) 