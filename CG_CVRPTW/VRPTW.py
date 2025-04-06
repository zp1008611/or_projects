import numpy as np
from typing import Dict,Optional
from typing import TYPE_CHECKING


class Graph:
    def __init__(self):
        self.depot_start: Depot = None
        self.depot_end: Depot = None
        self.all_customers: Dict[int, Customer] = {}
        self.all_nodes: Dict[int, Node] = {}
        self.distance_matrix: Optional[np.ndarray] = None

    def build_matrix(self):
        node_ids = list(self.all_nodes.keys())
        n_nodes = len(self.all_nodes)
        self.distance_matrix = np.full((n_nodes, n_nodes), float('inf'))
        
        for i in node_ids:
            node_i = self.all_nodes[i]
            # 处理depot_end到所有节点的距离
            if node_i.id == self.depot_end.id:
                self.distance_matrix[i, :] = float('inf')
            else: 
                for j in node_ids:
                    node_j = self.all_nodes[j]
                    if node_i.id == self.depot_start.id and node_j.id == self.depot_start.id:
                        self.distance_matrix[i,j] = 0
                    # 处理其他节点到depot_start的距离
                    elif node_i.id == self.depot_start.id and node_j.id == self.depot_end.id:
                        self.distance_matrix[i,j] = 0
                    elif node_j.id == self.depot_start.id:
                        self.distance_matrix[i,j] = float('inf')
                    # 处理同一节点的情况
                    elif node_i.id_external == node_j.id_external:
                        # self.distance_matrix[i,j] = self.distance_matrix[j,i] = float('inf')
                        self.distance_matrix[i,j] = self.distance_matrix[j,i] = 0
                    else:
                        # 计算欧氏距离
                        dist = np.sqrt(
                            (self.all_nodes[i].xcoord - self.all_nodes[j].xcoord) ** 2 +
                            (self.all_nodes[i].ycoord - self.all_nodes[j].ycoord) ** 2
                        )
                        if node_j.id == self.depot_end.id:
                            self.distance_matrix[i, j] = dist 
                        else:
                            self.distance_matrix[i, j] = self.distance_matrix[j, i] = dist 



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
        self.label = False
        self.best_cost = 0

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

class Customer(Node):
    def __init__(self, graph: Graph, external_id: int, x: float, y: float, 
                 demand: float, startTw: float, endTw: float, service_time: float):
        super().__init__(graph, external_id, x, y, demand, startTw, endTw, service_time)
        graph.all_customers[self.id] = self


class Depot(Node):
    def __init__(self, graph: Graph, external_id: int, x: float, y: float, 
                 startTw: float, endTw: float):
        super().__init__(graph, external_id, x, y, 0, startTw, endTw, 0)


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

    def if_contains_cus(self, customer:Customer):
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
    

class Instance:
    def __init__(self, instance_name):
        self.instance_name = instance_name
        self.graph = Graph()
        self.max_capacity = 0

    def read_data_from_file(self):
        try:
            with open(self.instance_name, 'r') as file:
                lines = [line.strip() for line in file if line.strip()]
                idx = 0

                # Skip header lines
                idx += 3  # Skip filename, empty line, and vehicle line
                vehicles_line = lines[idx]
                vehicles_nr = int(vehicles_line.split()[0].strip())
                vehicles_capacity = float(vehicles_line.split()[1].strip())
                self.max_capacity = vehicles_capacity

                # Skip more headers
                idx += 3
                # Read depot data
                depot_data = list(map(float, lines[idx].split()))
                depot_id = int(depot_data[0])
                d_x_coord = depot_data[1]
                d_y_coord = depot_data[2]
                d_start_tw = int(depot_data[4])
                d_end_tw = int(depot_data[5])
                idx += 1

                # Create depots
                self.graph.depot_start = Depot(self.graph, depot_id, d_x_coord, d_y_coord, d_start_tw, d_end_tw)
                
                # Read customers
                while idx < len(lines):
                    parts = list(map(float, lines[idx].split()))
                    if len(parts) < 7:
                        break
                    customer_id = int(parts[0])
                    x = parts[1]
                    y = parts[2]
                    # demand = parts[3]
                    demand = 1
                    # start_tw = int(parts[4])
                    start_tw = 0
                    # end_tw = int(parts[5])
                    end_tw = 1236
                    # service_time = parts[6]
                    service_time = 0
                    Customer(self.graph, customer_id, x, y, demand, start_tw, end_tw, service_time)
                    idx += 1

                self.graph.depot_end = Depot(self.graph, depot_id, d_x_coord, d_y_coord, d_start_tw, d_end_tw)

                # Build distance matrix
                self.graph.build_matrix()

        except FileNotFoundError:
            print("File not found!")
            exit(-1)

        return self.graph











