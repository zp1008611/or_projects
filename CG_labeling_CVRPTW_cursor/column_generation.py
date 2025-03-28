import numpy as np
from espprc import ESPPRC

class ColumnGeneration:
    def __init__(self, model):
        """
        初始化列生成算法
        
        Args:
            model: CVRP模型实例
        """
        self.model = model
        self.espprc = ESPPRC(model)
        
    def solve(self, max_iterations=100, tolerance=1e-6):
        """
        使用列生成算法求解CVRP
        
        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            dict: 包含最优解信息的字典
        """
        # 生成初始可行解
        initial_routes = self._generate_initial_routes()
        
        # 创建并求解受限主问题
        self.model.create_restricted_master_problem(initial_routes)
        current_routes = initial_routes.copy()
        
        for iteration in range(max_iterations):
            print(f"\n迭代 {iteration + 1}")
            
            # 求解RMP
            rmp_result = self.model.solve_rmp()
            print(f"当前目标值: {rmp_result['objective']}")
            
            # 获取对偶变量
            dual_values = self.model.get_dual_values()
            
            # 求解子问题(ESPPRC)
            new_route = self.espprc.solve(dual_values)
            
            if new_route is None:
                print("未找到新的负成本路径，算法收敛")
                break
                
            # 计算新路径的reduced cost
            reduced_cost = self._calculate_reduced_cost(new_route, dual_values)
            
            if reduced_cost > -tolerance:
                print("未找到新的负成本路径，算法收敛")
                break
                
            # 添加新路径到RMP
            current_routes.append(new_route)
            self.model.create_restricted_master_problem(current_routes)
            
        # 获取最终解
        final_solution = self._get_final_solution(current_routes)
        return final_solution
        
    def _generate_initial_routes(self):
        """生成初始可行路径"""
        routes = []
        unvisited = list(range(self.model.n_customers))
        current_load = 0
        current_route = [0]  # 从depot开始
        
        # 按需求量从大到小排序客户
        sorted_customers = sorted(unvisited, 
                                key=lambda x: self.model.demands[x], 
                                reverse=True)
        
        # 首先处理需求量大的客户
        for customer in sorted_customers:
            if customer not in unvisited:
                continue
                
            # 如果当前路径已满，开始新路径
            if current_load + self.model.demands[customer] > self.model.capacity:
                if len(routes) >= self.model.n_vehicles - 1:  # 只剩最后一辆车
                    # 将剩余客户都放在最后一条路径上
                    remaining_customers = [c for c in unvisited if c != customer]
                    if remaining_customers:
                        current_route.extend(remaining_customers)
                        current_route.append(0)
                        routes.append(current_route)
                        break
                    else:
                        # 如果当前客户无法放入任何路径，说明问题无解
                        raise ValueError("无法生成可行解，请增加车辆容量或减少客户数量")
                else:
                    # 开始新路径
                    current_route.append(0)
                    routes.append(current_route)
                    current_route = [0]
                    current_load = 0
            
            # 添加客户到当前路径
            current_route.append(customer)
            current_load += self.model.demands[customer]
            unvisited.remove(customer)
        
        # 添加最后一条路径
        if len(current_route) > 1:  # 如果最后一条路径不为空
            current_route.append(0)
            routes.append(current_route)
        
        # 验证所有客户是否都被访问
        if unvisited:
            raise ValueError("无法访问所有客户，请增加车辆容量或减少客户数量")
            
        return routes
        
    def _calculate_reduced_cost(self, route, dual_values):
        """计算路径的reduced cost"""
        # 路径成本
        route_cost = self.model._calculate_route_cost(route)
        
        # 减去对偶变量
        reduced_cost = route_cost
        for i in range(1, len(route)-1):  # 不包括depot
            customer = route[i]
            reduced_cost -= dual_values[customer]
            
        return reduced_cost
        
    def _get_final_solution(self, routes):
        """获取最终解"""
        # 获取变量的值
        var_values = [var.x for var in self.model.lambda_vars]
        
        # 构建最终路径
        final_routes = []
        for i, value in enumerate(var_values):
            if value > 1e-6:  # 只保留非零路径
                final_routes.append(routes[i])
                
        return {
            'objective': self.model.model.objval,
            'routes': final_routes
        } 