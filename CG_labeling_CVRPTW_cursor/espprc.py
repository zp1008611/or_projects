import numpy as np
from collections import defaultdict

class Label:
    def __init__(self, node, cost, load, path):
        self.node = node
        self.cost = cost
        self.load = load
        self.path = path
        
    def dominates(self, other):
        """检查当前标签是否支配另一个标签"""
        return (self.node == other.node and
                self.cost <= other.cost and
                self.load <= other.load)
                
    def __str__(self):
        return f"Label(node={self.node}, cost={self.cost:.2f}, load={self.load})"

class ESPPRC:
    def __init__(self, model):
        """
        初始化ESPPRC求解器
        
        Args:
            model: CVRP模型实例
        """
        self.model = model
        
    def solve(self, dual_values):
        """
        求解资源受限基本最短路问题
        
        Args:
            dual_values: 对偶变量值
            
        Returns:
            list: 最优路径，如果未找到可行解则返回None
        """
        # 初始化标签列表
        labels = defaultdict(list)
        labels[0].append(Label(0, 0, 0, [0]))  # 从depot开始
        
        # 初始化未处理节点集合
        unprocessed = {0}
        
        while unprocessed:
            current_node = unprocessed.pop()
            current_labels = labels[current_node]
            
            # 扩展当前节点的所有标签
            for label in current_labels:
                # 尝试访问所有可能的下一节点
                for next_node in range(self.model.n_customers + 1):
                    if next_node == current_node:
                        continue
                        
                    # 计算新的载重
                    new_load = label.load
                    if next_node > 0:  # 如果不是depot
                        new_load += self.model.demands[next_node-1]
                        
                    # 检查容量约束
                    if new_load > self.model.capacity:
                        continue
                        
                    # 计算新的成本
                    new_cost = (label.cost + 
                              self.model.distances[current_node, next_node] - 
                              (dual_values[next_node-1] if next_node > 0 else 0))
                    
                    # 创建新标签
                    new_path = label.path + [next_node]
                    new_label = Label(next_node, new_cost, new_load, new_path)
                    
                    # 检查是否应该添加新标签
                    should_add = True
                    for existing_label in labels[next_node]:
                        if existing_label.dominates(new_label):
                            should_add = False
                            break
                        if new_label.dominates(existing_label):
                            labels[next_node].remove(existing_label)
                            
                    if should_add:
                        labels[next_node].append(new_label)
                        unprocessed.add(next_node)
        
        # 找到返回depot的最优路径
        best_cost = float('inf')
        best_path = None
        
        for label in labels[0]:
            if label.cost < best_cost and len(label.path) > 1:  # 确保路径至少包含一个客户
                best_cost = label.cost
                best_path = label.path
                
        return best_path if best_path is not None else None 