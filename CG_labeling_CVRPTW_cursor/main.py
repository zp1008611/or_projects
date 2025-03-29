import numpy as np
from model import CVRPModel
from column_generation import ColumnGeneration

def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 创建CVRP问题实例
    n_customers = 10  # 客户数量
    n_vehicles = 3    # 车辆数量
    capacity = 100    # 车辆容量
    
    # 生成随机客户位置和需求量
    depot = np.array([0, 0])
    customers = np.random.rand(n_customers, 2) * 100  # 客户坐标
    demands = np.random.randint(10, 30, n_customers)  # 客户需求量
    
    # 计算距离矩阵
    locations = np.vstack([depot, customers])
    distances = np.zeros((n_customers + 1, n_customers + 1))
    for i in range(n_customers + 1):
        for j in range(n_customers + 1):
            distances[i,j] = np.linalg.norm(locations[i] - locations[j])
    
    # 创建CVRP模型
    model = CVRPModel(
        distances=distances,
        demands=demands,
        capacity=capacity,
        n_vehicles=n_vehicles
    )
    
    # 创建列生成求解器
    solver = ColumnGeneration(model)
    
    # 求解CVRP
    solution = solver.solve()
    
    # 输出结果
    print("\n最优解:")
    print(f"总成本: {solution['objective']}")
    print("\n路径:")
    for i, route in enumerate(solution['routes']):
        print(f"车辆 {i+1}: {route}")

if __name__ == "__main__":
    main() 