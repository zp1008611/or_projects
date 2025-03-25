import math
from typing import Dict
from typing import TYPE_CHECKING


class Graph:
    def __init__(self):
        self.depot_start: Depot = None
        self.depot_end: Depot = None
        self.all_customers: Dict[int, Customer] = {}
        self.all_nodes: Dict[int, Node] = {}
        self.distance_matrix: list[list[float]] = []

    def build_matrix(self):
        node_ids = list(self.all_nodes.keys())
        self.distance_matrix = [[0.0 for _ in node_ids] for _ in node_ids]
        
        for i in node_ids:
            node_i = self.all_nodes[i]
            for j in node_ids:
                node_j = self.all_nodes[j]
                
                # 处理depot_end到所有节点的距离
                if node_i.id == self.depot_end.id:
                    self.distance_matrix[i][j] = math.inf
                    continue
                # 处理其他节点到depot_start的距离
                if node_j.id == self.depot_start.id:
                    self.distance_matrix[i][j] = math.inf
                    continue
                # 处理同一节点的情况
                if node_i.id_external == node_j.id_external:
                    self.distance_matrix[i][j] = math.inf
                    continue
                
                # 计算欧几里得距离
                dx = node_i.xcoord - node_j.xcoord
                dy = node_i.ycoord - node_j.ycoord
                self.distance_matrix[i][j] = math.sqrt(dx**2 + dy**2)
                self.distance_matrix[j][i] = self.distance_matrix[i][j]



class Node:
    def __init__(self, graph: Graph, external_id: int, x: float, y: float, demand: float, start_tw: float, end_tw: float, service_time: float):
        self.id = len(graph.all_nodes)
        self.id_external = external_id
        self.xcoord = x
        self.ycoord = y
        self.demand = demand
        self.start_tw = start_tw
        self.end_tw = end_tw
        self.servicet = service_time
        graph.all_nodes[self.id] = self
        self.g = graph

    def time_to_node(self, node_to: 'Node') -> float:
        return self.g.distance_matrix[self.id][node_to.id]

    def time_at_node(self) -> float:
        return self.servicet

    def get_start_tw(self) -> float:
        return self.start_tw

    def set_start_tw(self, start_tw: float) -> None:
        self.start_tw = start_tw

    def get_end_tw(self) -> float:
        return self.end_tw

    def set_end_tw(self, end_tw: float) -> None:
        self.end_tw = end_tw

    def get_demand(self) -> float:
        return self.demand

    def set_demand(self, demand: float) -> None:
        self.demand = demand



class Path:
    def __init__(self, stops_new_path: list, paths: list, g: Graph):
        self.paths = paths
        self.g = g
        self.customers = []
        for i in range(1, len(stops_new_path) - 1):
            self.customers.append(g.all_customers[stops_new_path[i]])
        self.calculate_cost()
        self.id = len(self.paths)
        self.paths.append(self)
        self.theta = None

    def calculate_cost(self):
        if len(self.customers) > 0:
            self.cost = self.g.depot_start.time_to_node(self.customers[0])
            for i in range(1, len(self.customers)):
                self.cost += self.customers[i - 1].time_to_node(self.customers[i])
            self.cost += self.customers[-1].time_to_node(self.g.depot_end)
        else:
            self.cost = 0

    def if_contains_cus(self, customer: Customer):
        return 1 if customer in self.customers else 0

    def display_info(self):
        travel_cost = 0
        print(f"Path id   : {self.id}")
        print("Stops     : depot->", end="")
        travel_cost += self.g.depot_start.time_to_node(self.customers[0])
        for i in range(1, len(self.customers)):
            travel_cost += self.customers[i - 1].time_to_node(self.customers[i])
        travel_cost += self.customers[-1].time_to_node(self.g.depot_end)
        for c in self.customers:
            print(f"{c.id_external}->", end="")
        print("depot")
        print(f"trvalCost : {travel_cost}")
        return travel_cost
    





class Customer(Node):
    def __init__(self, graph: Graph, external_id: int, x: float, y: float, 
                 demand: float, startTw: float, endTw: float, service_time: float):
        super().__init__(graph, external_id, x, y, demand, startTw, endTw, service_time)
        graph.all_customers[self.id] = self


class Depot(Node):
    def __init__(self, graph: Graph, external_id: int, x: float, y: float, 
                 startTw: float, endTw: float):
        super().__init__(graph, external_id, x, y, 0, startTw, endTw, 0)





