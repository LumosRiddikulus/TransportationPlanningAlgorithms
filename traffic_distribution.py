import json
import math
import heapq
from typing import List, Dict, Tuple, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


class Graph:
    """简单的图类"""
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.adjacency = {}
    
    def add_node(self, node_id: str, x: float, y: float):
        self.nodes[node_id] = {'x': x, 'y': y}
        self.adjacency[node_id] = {}
    
    def add_edge(self, from_node: str, to_node: str, weight: float, **attrs):
        edge_id = f"{from_node}{to_node}"
        self.edges[edge_id] = {
            'from': from_node,
            'to': to_node,
            'weight': weight,
            **attrs
        }
        self.adjacency[from_node][to_node] = weight
    
    def dijkstra(self, start: str, end: str) -> List[str]:
        """Dijkstra算法求最短路径"""
        if start not in self.nodes or end not in self.nodes:
            return []
            
        distances = {node: float('inf') for node in self.nodes}
        previous = {node: None for node in self.nodes}
        distances[start] = 0
        
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            
            if distances[current] == float('inf'):
                break
                
            if current == end:
                break
                
            unvisited.remove(current)
            
            for neighbor, weight in self.adjacency[current].items():
                if neighbor in unvisited:
                    new_distance = distances[current] + weight
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        
        path.reverse()
        
        if not path or path[0] != start:
            return []
            
        return path
    
    def dijkstra_all_paths(self, start_node: str, weight_type: str = 't') -> Dict[str, List[str]]:
        """改进的Dijkstra算法，返回所有节点的最短路径"""
        if start_node not in self.nodes:
            return {}
        
        shortest_paths = {start_node: [start_node]}
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0
        
        heap = []
        heapq.heappush(heap, (0, start_node))
        
        predecessors = defaultdict(list)
        
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            
            if current_dist > distances[current_node]:
                continue
                
            for neighbor in self.adjacency[current_node]:
                w = self.adjacency[current_node][neighbor]
                new_dist = current_dist + w
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = [current_node]
                    heapq.heappush(heap, (new_dist, neighbor))
                    
                elif abs(new_dist - distances[neighbor]) < 1e-10:
                    predecessors[neighbor].append(current_node)
        
        paths = {}
        for node in self.nodes:
            if distances[node] == float('inf'):
                continue
            path = []
            current = node
            while current != start_node:
                path.insert(0, current)
                if predecessors[current]:
                    current = predecessors[current][0]
                else:
                    break
            path.insert(0, start_node)
            paths[node] = path
            
        return paths


class Polynomial:
    """多项式类，不使用外部库"""
    
    def __init__(self, variable: str = "x", factors: List[float] = None):
        self.variable = variable
        self.factors = factors if factors is not None else [0]
        while len(self.factors) > 1 and abs(self.factors[-1]) < 1e-10:
            self.factors.pop()
    
    def evaluate(self, x: float) -> float:
        """计算多项式在x处的值"""
        result = 0
        for i, coeff in enumerate(self.factors):
            result += coeff * (x ** i)
        return result
    
    def derivative(self) -> 'Polynomial':
        """求导，返回新的多项式"""
        if len(self.factors) <= 1:
            return Polynomial(self.variable, [0])
        
        new_factors = []
        for i in range(1, len(self.factors)):
            new_factors.append(i * self.factors[i])
        
        return Polynomial(self.variable, new_factors)
    
    def integral(self) -> 'Polynomial':
        """积分，返回新的多项式"""
        new_factors = [0]
        for i, coeff in enumerate(self.factors):
            new_factors.append(coeff / (i + 1))
        
        return Polynomial(self.variable, new_factors)
    
    def __str__(self) -> str:
        terms = []
        for i, coeff in enumerate(self.factors):
            if abs(coeff) > 1e-10:
                if i == 0:
                    terms.append(f"{coeff:.4f}")
                elif i == 1:
                    terms.append(f"{coeff:.4f}{self.variable}")
                else:
                    terms.append(f"{coeff:.4f}{self.variable}^{i}")
        
        if not terms:
            return "0"
        
        return " + ".join(terms).replace("+ -", "- ")


class RoadNetwork:
    """道路网络类"""    
    def __init__(self):
        self.nodes = {}
        self.links = {}
        self.graph = Graph()
        self.demands = []
    
    def load_network(self, network_file: str):
        """读取路网文件"""
        try:
            with open(network_file, 'r') as f:
                data = json.load(f)
            
            if 'nodes' not in data or 'links' not in data:
                raise ValueError("JSON文件结构错误，缺少'nodes'或'links'字段")
            
            for i, name in enumerate(data['nodes']['name']):
                x = data['nodes']['x'][i]
                y = data['nodes']['y'][i]
                self.add_node(name, x, y)
            
            for i, between in enumerate(data['links']['between']):
                capacity = data['links']['capacity'][i]
                speed_max = data['links']['speedmax'][i]
                from_node = between[0]
                to_node = between[1]
                
                x1, y1 = self.nodes[from_node]['x'], self.nodes[from_node]['y']
                x2, y2 = self.nodes[to_node]['x'], self.nodes[to_node]['y']
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                self.add_link(f"{from_node}{to_node}", from_node, to_node, 
                             length, capacity, speed_max)
                self.add_link(f"{to_node}{from_node}", to_node, from_node, 
                             length, capacity, speed_max)
            
            print(f"成功加载路网: {len(self.nodes)}个节点, {len(self.links)}条路段")
            
        except Exception as e:
            print(f"加载路网文件失败: {e}")
            raise
    
    def load_demand(self, demand_file: str):
        """读取需求文件"""
        try:
            with open(demand_file, 'r') as f:
                data = json.load(f)
            
            self.demands = []
            for i in range(len(data['from'])):
                self.demands.append({
                    'from': data['from'][i],
                    'to': data['to'][i],
                    'amount': data['amount'][i],
                    'id': i+1  # 添加OD对编号
                })
            
            print(f"成功加载交通需求: {len(self.demands)}个OD对")
            
            total_demand = sum(d['amount'] for d in self.demands)
            print(f"总交通需求: {total_demand} 辆/小时")
            
        except Exception as e:
            print(f"加载需求文件失败: {e}")
            raise
    
    def add_node(self, node_id: str, x: float, y: float):
        """添加节点"""
        self.nodes[node_id] = {'x': x, 'y': y}
        self.graph.add_node(node_id, x, y)
    
    def add_link(self, link_id: str, from_node: str, to_node: str, 
                length: float, capacity: float, speed_max: float):
        """添加路段"""
        free_flow_time = length / speed_max
        
        t0 = free_flow_time
        factors = [t0, 2*t0/capacity, t0/(capacity**2)]
        time_function = Polynomial("q", factors)
        
        self.links[link_id] = {
            'from': from_node,
            'to': to_node,
            'length': length,
            'capacity': capacity,
            'speed_max': speed_max,
            'free_flow_time': free_flow_time,
            'time_function': time_function,
            'flow': 0
        }
        
        self.graph.add_edge(from_node, to_node, free_flow_time,
                          length=length,
                          capacity=capacity,
                          free_flow_time=free_flow_time,
                          time_function=time_function,
                          flow=0)
    
    def calculate_travel_time(self, link_id: str, flow: float) -> float:
        """计算路段行程时间"""
        return self.links[link_id]['time_function'].evaluate(flow)
    
    def update_graph_weights(self, weight_type: str = "free_flow"):
        """更新图的权重"""
        for link_id, link in self.links.items():
            if weight_type == "free_flow":
                weight = link['free_flow_time']
            elif weight_type == "current":
                weight = self.calculate_travel_time(link_id, link['flow'])
            else:
                weight = link['free_flow_time']
            
            self.graph.adjacency[link['from']][link['to']] = weight
    
    def reset_flows(self):
        """重置所有路段流量为0"""
        for link_id in self.links:
            self.links[link_id]['flow'] = 0
    
    def find_all_paths(self, start: str, end: str, max_paths: int = 10) -> List[List[str]]:
        """查找所有可能的路径"""
        if start not in self.nodes or end not in self.nodes:
            return []
            
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[str]):
            if current == end:
                paths.append(path.copy())
                return
            
            if len(paths) >= max_paths:
                return
                
            visited.add(current)
            
            for neighbor in self.graph.adjacency[current]:
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
            
            visited.remove(current)
        
        dfs(start, [start])
        return paths
    
    # ========== 全有全无分配算法 ==========
    
    def all_or_nothing_single_od(self, origin: str, destination: str, amount: float) -> Dict:
        """单个OD对的全有全无分配"""
        # 临时保存当前流量
        temp_flows = {link_id: link['flow'] for link_id, link in self.links.items()}
        
        # 重置为0
        for link_id in self.links:
            self.links[link_id]['flow'] = 0
        
        self.update_graph_weights("free_flow")
        
        used_paths = {}
        
        shortest_path = self.graph.dijkstra(origin, destination)
        
        if not shortest_path:
            print(f"警告: 找不到从 {origin} 到 {destination} 的路径")
            # 恢复流量
            for link_id, flow in temp_flows.items():
                self.links[link_id]['flow'] = flow
            return {'used_paths': {}, 'link_flows': {}}
        
        path_key = "->".join(shortest_path)
        used_paths[path_key] = {
            'path': shortest_path,
            'flow': amount,
            'travel_time': 0
        }
        
        for i in range(len(shortest_path)-1):
            from_node = shortest_path[i]
            to_node = shortest_path[i+1]
            link_id = f"{from_node}{to_node}"
            self.links[link_id]['flow'] = amount
        
        # 计算路径行程时间
        path_travel_time = 0
        for i in range(len(shortest_path)-1):
            from_node = shortest_path[i]
            to_node = shortest_path[i+1]
            link_id = f"{from_node}{to_node}"
            path_travel_time += self.calculate_travel_time(link_id, self.links[link_id]['flow'])
        used_paths[path_key]['travel_time'] = path_travel_time
        
        # 记录单OD分配的路段流量
        link_flows = {}
        for link_id in self.links:
            if self.links[link_id]['flow'] > 0:
                link_flows[link_id] = self.links[link_id]['flow']
        
        # 恢复整体流量
        for link_id, flow in temp_flows.items():
            self.links[link_id]['flow'] = flow
        
        return {
            'used_paths': used_paths,
            'link_flows': link_flows
        }
    
    def all_or_nothing_assignment(self) -> Dict:
        """全有全无分配算法（整体）"""
        self.reset_flows()
        self.update_graph_weights("free_flow")
        
        total_travel_time = 0
        used_paths = {}
        
        print("执行全有全无分配...")
        
        for demand in self.demands:
            origin = demand['from']
            destination = demand['to']
            amount = demand['amount']
            
            shortest_path = self.graph.dijkstra(origin, destination)
            
            if not shortest_path:
                print(f"警告: 找不到从 {origin} 到 {destination} 的路径")
                continue
            
            path_key = "->".join(shortest_path)
            
            if path_key not in used_paths:
                used_paths[path_key] = {
                    'path': shortest_path,
                    'flow': 0,
                    'travel_time': 0
                }
            used_paths[path_key]['flow'] += amount
            
            for i in range(len(shortest_path)-1):
                from_node = shortest_path[i]
                to_node = shortest_path[i+1]
                link_id = f"{from_node}{to_node}"
                self.links[link_id]['flow'] += amount
        
        for link_id in self.links:
            travel_time = self.calculate_travel_time(link_id, self.links[link_id]['flow'])
            total_travel_time += self.links[link_id]['flow'] * travel_time
        
        for path_key in used_paths:
            path = used_paths[path_key]['path']
            path_travel_time = 0
            for i in range(len(path)-1):
                from_node = path[i]
                to_node = path[i+1]
                link_id = f"{from_node}{to_node}"
                path_travel_time += self.calculate_travel_time(link_id, self.links[link_id]['flow'])
            used_paths[path_key]['travel_time'] = path_travel_time
        
        return {
            'total_travel_time': total_travel_time,
            'used_paths': used_paths
        }
    
    # ========== 增量分配算法 ==========
    
    def incremental_single_od(self, origin: str, destination: str, amount: float, increments: int = 4) -> Dict:
        """单个OD对的增量分配"""
        # 临时保存当前流量
        temp_flows = {link_id: link['flow'] for link_id, link in self.links.items()}
        
        # 重置为0
        for link_id in self.links:
            self.links[link_id]['flow'] = 0
        
        used_paths = {}
        
        for step in range(increments):
            fraction = 1.0 / increments
            step_amount = amount * fraction
            
            self.update_graph_weights("current")
            
            shortest_path = self.graph.dijkstra(origin, destination)
            
            if not shortest_path:
                continue
            
            path_key = "->".join(shortest_path)
            
            if path_key not in used_paths:
                used_paths[path_key] = {
                    'path': shortest_path,
                    'flow': 0,
                    'travel_time': 0
                }
            used_paths[path_key]['flow'] += step_amount
            
            for i in range(len(shortest_path)-1):
                from_node = shortest_path[i]
                to_node = shortest_path[i+1]
                link_id = f"{from_node}{to_node}"
                self.links[link_id]['flow'] += step_amount
        
        # 计算路径行程时间
        for path_key, path_info in used_paths.items():
            path = path_info['path']
            path_travel_time = 0
            for i in range(len(path)-1):
                from_node = path[i]
                to_node = path[i+1]
                link_id = f"{from_node}{to_node}"
                path_travel_time += self.calculate_travel_time(link_id, self.links[link_id]['flow'])
            used_paths[path_key]['travel_time'] = path_travel_time
        
        # 记录单OD分配的路段流量
        link_flows = {}
        for link_id in self.links:
            if self.links[link_id]['flow'] > 0:
                link_flows[link_id] = self.links[link_id]['flow']
        
        # 恢复整体流量
        for link_id, flow in temp_flows.items():
            self.links[link_id]['flow'] = flow
        
        return {
            'used_paths': used_paths,
            'link_flows': link_flows
        }
    
    def incremental_assignment(self, increments: int = 4) -> Dict:
        """增量分配算法（整体）"""
        self.reset_flows()
        
        total_travel_time = 0
        used_paths = {}
        
        print(f"执行增量分配，分{increments}步加载...")
        
        for step in range(increments):
            fraction = 1.0 / increments
            
            for demand in self.demands:
                origin = demand['from']
                destination = demand['to']
                amount = demand['amount'] * fraction
                
                self.update_graph_weights("current")
                
                shortest_path = self.graph.dijkstra(origin, destination)
                
                if not shortest_path:
                    continue
                
                path_key = "->".join(shortest_path)
                
                if path_key not in used_paths:
                    used_paths[path_key] = {
                        'path': shortest_path,
                        'flow': 0,
                        'travel_time': 0
                    }
                used_paths[path_key]['flow'] += amount
                
                for i in range(len(shortest_path)-1):
                    from_node = shortest_path[i]
                    to_node = shortest_path[i+1]
                    link_id = f"{from_node}{to_node}"
                    self.links[link_id]['flow'] += amount
        
        for link_id in self.links:
            travel_time = self.calculate_travel_time(link_id, self.links[link_id]['flow'])
            total_travel_time += self.links[link_id]['flow'] * travel_time
        
        for path_key in used_paths:
            path = used_paths[path_key]['path']
            path_travel_time = 0
            for i in range(len(path)-1):
                from_node = path[i]
                to_node = path[i+1]
                link_id = f"{from_node}{to_node}"
                path_travel_time += self.calculate_travel_time(link_id, self.links[link_id]['flow'])
            used_paths[path_key]['travel_time'] = path_travel_time
        
        return {
            'total_travel_time': total_travel_time,
            'used_paths': used_paths
        }
    
    # ========== Frank-Wolfe用户均衡分配算法 ==========
    
    def frank_wolfe_single_od(self, origin: str, destination: str, amount: float, 
                            max_iterations: int = 50, tolerance: float = 1e-4) -> Dict:
        """单个OD对的Frank-Wolfe用户均衡分配"""
        # 临时保存当前流量
        temp_flows = {link_id: link['flow'] for link_id, link in self.links.items()}
        
        # 重置为0
        for link_id in self.links:
            self.links[link_id]['flow'] = 0
        
        # 初始化流量
        x = {link_id: 0.0 for link_id in self.links}
        
        # 简单的Frank-Wolfe迭代（针对单个OD对）
        for iteration in range(max_iterations):
            # 使用当前流量更新图权重
            for link_id, flow in x.items():
                travel_time = self.calculate_travel_time(link_id, flow)
                from_node = self.links[link_id]['from']
                to_node = self.links[link_id]['to']
                self.graph.adjacency[from_node][to_node] = travel_time
            
            # 计算辅助流量模式y（全有全无分配）
            y = {link_id: 0.0 for link_id in self.links}
            
            shortest_path = self.graph.dijkstra(origin, destination)
            if shortest_path:
                for i in range(len(shortest_path)-1):
                    from_node = shortest_path[i]
                    to_node = shortest_path[i+1]
                    link_id = f"{from_node}{to_node}"
                    y[link_id] = amount
            
            # 计算对偶间隙
            gap = 0
            total_travel_time_x = 0
            
            for link_id in self.links:
                travel_time = self.calculate_travel_time(link_id, x[link_id])
                gap += travel_time * (y[link_id] - x[link_id])
                total_travel_time_x += travel_time * x[link_id]
            
            # 相对对偶间隙
            if total_travel_time_x > 0:
                relative_gap = abs(gap) / total_travel_time_x
            else:
                relative_gap = float('inf')
            
            # 收敛检查
            if iteration >= 5 and relative_gap < tolerance:
                break
            
            # 使用简单步长
            alpha = 1.0 / (iteration + 2)
            
            # 更新流量
            for link_id in self.links:
                x[link_id] = (1 - alpha) * x[link_id] + alpha * y[link_id]
        
        # 将流量应用到网络
        for link_id in self.links:
            self.links[link_id]['flow'] = x[link_id]
        
        # 查找使用的路径
        used_paths = {}
        all_paths = self.find_all_paths(origin, destination, max_paths=10)
        
        if all_paths:
            # 计算每条路径的行程时间
            path_data = []
            for path in all_paths:
                path_travel_time = 0
                for i in range(len(path)-1):
                    from_node = path[i]
                    to_node = path[i+1]
                    link_id = f"{from_node}{to_node}"
                    path_travel_time += self.calculate_travel_time(link_id, self.links[link_id]['flow'])
                
                path_data.append({
                    'path': path,
                    'travel_time': path_travel_time,
                    'key': "->".join(path)
                })
            
            # 找到最短路径时间
            min_travel_time = min(p['travel_time'] for p in path_data)
            
            # 选择行程时间接近最短路径的路径
            used_paths_for_od = []
            for path_info in path_data:
                if abs(path_info['travel_time'] - min_travel_time) < 0.1:  # 0.1小时容差
                    used_paths_for_od.append(path_info)
            
            # 如果找到多条路径，按行程时间分配流量
            if used_paths_for_od:
                # 计算权重（行程时间越短，权重越大）
                total_weight = 0
                for path_info in used_paths_for_od:
                    weight = math.exp(-path_info['travel_time'])
                    path_info['weight'] = weight
                    total_weight += weight
                
                # 按权重分配流量
                for path_info in used_paths_for_od:
                    path_flow = amount * (path_info['weight'] / total_weight)
                    used_paths[path_info['key']] = {
                        'path': path_info['path'],
                        'flow': path_flow,
                        'travel_time': path_info['travel_time']
                    }
            else:
                # 如果没有找到多条路径，使用最短路径
                shortest_path_info = min(path_data, key=lambda x: x['travel_time'])
                used_paths[shortest_path_info['key']] = {
                    'path': shortest_path_info['path'],
                    'flow': amount,
                    'travel_time': shortest_path_info['travel_time']
                }
        
        # 记录单OD分配的路段流量
        link_flows = {}
        for link_id in self.links:
            if self.links[link_id]['flow'] > 0:
                link_flows[link_id] = self.links[link_id]['flow']
        
        # 恢复整体流量
        for link_id, flow in temp_flows.items():
            self.links[link_id]['flow'] = flow
        
        return {
            'used_paths': used_paths,
            'link_flows': link_flows
        }
    
    def golden_section_search(self, obj_func: Callable[[float], float], 
                            bounds: Tuple[float, float] = (0, 1), 
                            tol: float = 1e-4, max_iter: int = 20) -> float:
        """黄金分割法寻找最优步长"""
        a, b = bounds
        golden_ratio = 0.618033988749895
        
        # 初始四个点
        x1 = b - golden_ratio * (b - a)
        x2 = a + golden_ratio * (b - a)
        
        f1 = obj_func(x1)
        f2 = obj_func(x2)
        
        for _ in range(max_iter):
            if abs(b - a) < tol:
                break
                
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - golden_ratio * (b - a)
                f1 = obj_func(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + golden_ratio * (b - a)
                f2 = obj_func(x2)
        
        return (a + b) / 2
    
    def frank_wolfe_equilibrium(self, max_iterations: int = 100, tolerance: float = 1e-4) -> Dict:
        """改进的Frank-Wolfe用户均衡分配算法（整体）"""
        print("执行Frank-Wolfe用户均衡分配...")
        
        # 重置流量
        self.reset_flows()
        
        # 初始化：使用全有全无分配作为起点
        initial_result = self.all_or_nothing_assignment()
        x = {link_id: self.links[link_id]['flow'] for link_id in self.links}
        
        print(f"初始总出行时间: {initial_result['total_travel_time']:.2f} 车辆-小时")
        
        best_x = x.copy()
        best_objective = self.calculate_objective_function(x)
        best_travel_time = self.calculate_total_travel_time(x)
        
        convergence_history = []
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n--- 迭代 {iteration} ---")
            
            # 使用当前流量更新图权重
            for link_id, flow in x.items():
                travel_time = self.calculate_travel_time(link_id, flow)
                from_node = self.links[link_id]['from']
                to_node = self.links[link_id]['to']
                self.graph.adjacency[from_node][to_node] = travel_time
            
            # 计算辅助流量模式y（全有全无分配）
            y = {link_id: 0.0 for link_id in self.links}
            
            total_demand_assigned = 0
            for demand in self.demands:
                origin = demand['from']
                destination = demand['to']
                amount = demand['amount']
                
                shortest_path = self.graph.dijkstra(origin, destination)
                if not shortest_path:
                    print(f"警告: 找不到从 {origin} 到 {destination} 的路径")
                    continue
                
                # 将交通需求分配到最短路径上
                for i in range(len(shortest_path)-1):
                    from_node = shortest_path[i]
                    to_node = shortest_path[i+1]
                    link_id = f"{from_node}{to_node}"
                    y[link_id] += amount
                    total_demand_assigned += amount
            
            print(f"辅助流量模式总流量: {sum(y.values()):.0f}")
            print(f"当前流量模式总流量: {sum(x.values()):.0f}")
            
            # 计算对偶间隙
            gap = 0
            total_travel_time_x = 0
            
            for link_id in self.links:
                travel_time = self.calculate_travel_time(link_id, x[link_id])
                gap += travel_time * (y[link_id] - x[link_id])
                total_travel_time_x += travel_time * x[link_id]
            
            # 相对对偶间隙
            if total_travel_time_x > 0:
                relative_gap = abs(gap) / total_travel_time_x
            else:
                relative_gap = float('inf')
            
            print(f"相对对偶间隙: {relative_gap:.6f}")
            
            # 收敛检查
            if iteration >= 5 and relative_gap < tolerance:
                print(f"算法在 {iteration} 次迭代后收敛")
                break
            
            # 定义目标函数用于步长搜索
            def objective_function(alpha: float) -> float:
                z = {}
                for link_id in self.links:
                    z[link_id] = (1 - alpha) * x[link_id] + alpha * y[link_id]
                return self.calculate_objective_function(z)
            
            # 使用黄金分割法寻找最优步长
            try:
                alpha = self.golden_section_search(objective_function, bounds=(0, 1), tol=1e-4, max_iter=20)
                print(f"找到最优步长: {alpha:.6f}")
                
                # 如果步长过小，但相对间隙还比较大，强制最小步长
                if alpha < 0.01 and relative_gap > 0.01 and iteration < 20:
                    alpha = 0.1
                    print(f"步长过小但间隙仍大，调整为: {alpha:.6f}")
            except:
                alpha = 0.5
                print(f"步长搜索失败，使用默认步长: {alpha:.6f}")
            
            # 更新流量
            for link_id in self.links:
                x[link_id] = (1 - alpha) * x[link_id] + alpha * y[link_id]
            
            # 计算当前目标函数值和总出行时间
            current_objective = self.calculate_objective_function(x)
            current_travel_time = self.calculate_total_travel_time(x)
            
            print(f"目标函数: {current_objective:.2f}")
            print(f"总出行时间: {current_travel_time:.2f} 车辆-小时")
            
            # 更新历史最佳解
            if current_objective < best_objective:
                best_objective = current_objective
                best_x = x.copy()
                best_travel_time = current_travel_time
                print(f"找到改进解!")
            
            convergence_history.append({
                'iteration': iteration,
                'relative_gap': relative_gap,
                'objective': current_objective,
                'travel_time': current_travel_time
            })
            
            # 强制至少迭代5次
            if iteration < 5:
                continue
            
            # 检查目标函数变化
            if iteration > 5:
                prev_obj = convergence_history[iteration-2]['objective']
                obj_change = abs(current_objective - prev_obj) / (abs(prev_obj) + 1e-10)
                if obj_change < tolerance/10 and iteration >= 10:
                    print(f"目标函数变化收敛: {obj_change:.6f} < {tolerance/10}")
                    break
        
        # 使用历史最佳解
        for link_id in self.links:
            self.links[link_id]['flow'] = best_x[link_id]
        
        # 计算最终总出行时间和使用路径
        final_travel_time = self.calculate_total_travel_time(best_x)
        used_paths = self.find_all_used_paths_accurate(best_x)
        
        print(f"\n最终结果:")
        print(f"总出行时间: {final_travel_time:.2f} 车辆-小时")
        print(f"最佳总出行时间: {best_travel_time:.2f} 车辆-小时")
        
        return {
            'total_travel_time': final_travel_time,
            'used_paths': used_paths,
            'iterations': iteration,
            'convergence_history': convergence_history
        }

    def calculate_objective_function(self, flows: Dict) -> float:
        """计算目标函数值（总出行时间积分）"""
        total = 0
        for link_id, flow in flows.items():
            t0 = self.links[link_id]['free_flow_time']
            cap = self.links[link_id]['capacity']
            
            # ∫₀^flow t0 * (1 + w/cap)^2 dw = t0 * [flow + (flow^2)/cap + (flow^3)/(3*cap^2)]
            if flow > 0:
                integral_value = t0 * (flow + (flow ** 2) / cap + (flow ** 3) / (3 * cap ** 2))
                total += integral_value
        
        return total
    
    def calculate_total_travel_time(self, flows: Dict) -> float:
        """计算总出行时间"""
        total = 0
        for link_id, flow in flows.items():
            travel_time = self.calculate_travel_time(link_id, flow)
            total += flow * travel_time
        return total
    
    def find_all_used_paths_accurate(self, flows: Dict[str, float]) -> Dict:
        """准确的路径查找方法 - 基于OD对的流量分配"""
        used_paths = {}
        
        # 使用最终流量更新图权重
        for link_id, flow in flows.items():
            travel_time = self.calculate_travel_time(link_id, flow)
            from_node = self.links[link_id]['from']
            to_node = self.links[link_id]['to']
            self.graph.adjacency[from_node][to_node] = travel_time
        
        # 为每个OD对计算路径流量
        for demand in self.demands:
            origin = demand['from']
            destination = demand['to']
            amount = demand['amount']
            
            # 查找所有可能的路径
            all_paths = self.find_all_paths(origin, destination, max_paths=10)
            
            # 计算每条路径的行程时间
            path_data = []
            for path in all_paths:
                path_travel_time = 0
                for i in range(len(path)-1):
                    from_node = path[i]
                    to_node = path[i+1]
                    link_id = f"{from_node}{to_node}"
                    path_travel_time += self.calculate_travel_time(link_id, flows[link_id])
                
                path_data.append({
                    'path': path,
                    'travel_time': path_travel_time,
                    'key': "->".join(path)
                })
            
            # 找到最短路径时间
            if path_data:
                min_travel_time = min(p['travel_time'] for p in path_data)
                
                # 选择行程时间接近最短路径的路径
                used_paths_for_od = []
                for path_info in path_data:
                    # 如果路径行程时间接近最短路径，则认为被使用
                    if abs(path_info['travel_time'] - min_travel_time) < 0.1:  # 0.1小时容差
                        used_paths_for_od.append(path_info)
                
                # 如果找到多条路径，按行程时间分配流量
                if used_paths_for_od:
                    # 计算权重（行程时间越短，权重越大）
                    total_weight = 0
                    for path_info in used_paths_for_od:
                        # 使用负指数函数作为权重
                        weight = math.exp(-path_info['travel_time'])
                        path_info['weight'] = weight
                        total_weight += weight
                    
                    # 按权重分配流量
                    for path_info in used_paths_for_od:
                        path_flow = amount * (path_info['weight'] / total_weight)
                        used_paths[path_info['key']] = {
                            'path': path_info['path'],
                            'flow': path_flow,
                            'travel_time': path_info['travel_time']
                        }
                else:
                    # 如果没有找到多条路径，使用最短路径
                    shortest_path_info = min(path_data, key=lambda x: x['travel_time'])
                    used_paths[shortest_path_info['key']] = {
                        'path': shortest_path_info['path'],
                        'flow': amount,
                        'travel_time': shortest_path_info['travel_time']
                    }
        
        return used_paths
    
    def print_assignment_results(self, results: Dict, algorithm_name: str):
        """打印分配结果"""
        print(f"\n{'='*60}")
        print(f"{algorithm_name} 分配结果")
        print(f"{'='*60}")
        
        print(f"总出行时间: {results['total_travel_time']:.2f} 车辆-小时")
        
        print(f"\n路段流量分布:")
        print(f"{'路段':<8} {'流量(辆/小时)':<12} {'容量':<8} {'饱和度':<8} {'行程时间(小时)':<12}")
        print(f"{'-'*60}")
        
        total_flow = 0
        for link_id in sorted(self.links.keys()):
            link = self.links[link_id]
            flow = link['flow']
            capacity = link['capacity']
            saturation = flow / capacity if capacity > 0 else 0
            travel_time = self.calculate_travel_time(link_id, flow)
            
            print(f"{link_id:<8} {flow:<12.0f} {capacity:<8} {saturation:<8.2f} {travel_time:<12.2f}")
            total_flow += flow
        
        total_demand = sum(d['amount'] for d in self.demands)
        print(f"总分配流量: {total_flow:.0f} 辆/小时")
        print(f"总交通需求: {total_demand:.0f} 辆/小时")
        
        if 'used_paths' in results and results['used_paths']:
            print(f"\n使用的路径:")
            total_path_flow = 0
            for i, (path_key, path_info) in enumerate(results['used_paths'].items(), 1):
                print(f"路径 {i}: {path_key}")
                print(f"  流量: {path_info['flow']:.0f} 辆/小时, 行程时间: {path_info['travel_time']:.2f} 小时")
                total_path_flow += path_info['flow']
            
            # 检查路径流量总和是否合理
            if abs(total_path_flow - total_demand) / total_demand > 0.1:
                print(f"⚠️ 注意: 路径流量总和({total_path_flow:.0f})与总交通需求({total_demand:.0f})差异较大")
            else:
                print(f"✅ 路径流量总和({total_path_flow:.0f})与总交通需求({total_demand:.0f})匹配良好")
        else:
            print(f"\n使用的路径: 无路径被识别")
        
        if 'iterations' in results:
            print(f"\n收敛迭代次数: {results['iterations']}")
    
    def print_single_od_results(self, algorithm_name: str):
        """打印所有单OD对分配结果"""
        print(f"\n{'='*60}")
        print(f"{algorithm_name} - 单OD对分配结果")
        print(f"{'='*60}")
        
        for demand in self.demands:
            od_id = demand['id']
            origin = demand['from']
            destination = demand['to']
            amount = demand['amount']
            
            print(f"\nOD对 {od_id}: {origin} -> {destination}, 需求: {amount} 辆/小时")
            
            # 根据算法名称调用对应的单OD分配函数
            if algorithm_name == "全有全无分配":
                result = self.all_or_nothing_single_od(origin, destination, amount)
            elif algorithm_name == "增量分配":
                result = self.incremental_single_od(origin, destination, amount, increments=4)
            elif algorithm_name == "Frank-Wolfe用户均衡分配":
                result = self.frank_wolfe_single_od(origin, destination, amount, max_iterations=20, tolerance=1e-3)
            else:
                print(f"  未知算法: {algorithm_name}")
                continue
            
            # 打印路径信息
            if 'used_paths' in result and result['used_paths']:
                print(f"  使用的路径:")
                for i, (path_key, path_info) in enumerate(result['used_paths'].items(), 1):
                    print(f"    路径 {i}: {path_key}")
                    print(f"      流量: {path_info['flow']:.0f} 辆/小时, 行程时间: {path_info['travel_time']:.2f} 小时")
            else:
                print(f"  没有找到可行路径")
            
            # 打印路段流量
            if 'link_flows' in result and result['link_flows']:
                print(f"  路段流量分配:")
                for link_id, flow in sorted(result['link_flows'].items()):
                    print(f"    {link_id}: {flow:.0f} 辆/小时")
            else:
                print(f"  无路段流量")
            
            print("-" * 40)
    
    def text_visualization(self):
        """文本方式可视化路网"""
        print(f"\n{'='*40}")
        print("路网结构可视化")
        print(f"{'='*40}")
        
        print("节点坐标:")
        for node_id, node in self.nodes.items():
            print(f"  {node_id}: ({node['x']}, {node['y']})")
        
        print(f"\n路段信息:")
        for link_id in sorted(self.links.keys()):
            link = self.links[link_id]
            travel_time = self.calculate_travel_time(link_id, link['flow'])
            print(f"  {link_id}: {link['from']}->{link['to']}, "
                  f"长度={link['length']:.2f}km, 容量={link['capacity']}辆/小时, "
                  f"流量={link['flow']:.0f}辆/小时, 时间={travel_time:.2f}小时")
    
    def compare_algorithms(self, aon_result: Dict, inc_result: Dict, fw_result: Dict):
        """比较三种算法的结果"""
        print(f"\n{'='*80}")
        print("三种交通分配算法对比分析")
        print(f"{'='*80}")
        
        print(f"\n算法性能对比:")
        print(f"{'算法名称':<25} {'总出行时间(车辆-小时)':<25} {'计算复杂度':<15} {'收敛性':<10}")
        print(f"{'-'*80}")
        
        algorithms = [
            ("全有全无分配", aon_result['total_travel_time'], "低", "不收敛"),
            ("增量分配", inc_result['total_travel_time'], "中", "部分收敛"), 
            ("Frank-Wolfe均衡", fw_result['total_travel_time'], "高", "收敛")
        ]
        
        for name, travel_time, complexity, convergence in algorithms:
            print(f"{name:<25} {travel_time:<25.2f} {complexity:<15} {convergence:<10}")
        
        aon_time = aon_result['total_travel_time']
        inc_time = inc_result['total_travel_time']
        fw_time = fw_result['total_travel_time']
        
        print(f"\n效率提升分析:")
        if aon_time > 0 and inc_time > 0:
            improvement_inc = (aon_time - inc_time) / aon_time * 100
            print(f"增量分配相比全有全无分配提升: {improvement_inc:.2f}%")
        
        if aon_time > 0 and fw_time > 0:
            improvement_fw = (aon_time - fw_time) / aon_time * 100
            print(f"Frank-Wolfe相比全有全无分配提升: {improvement_fw:.2f}%")
        
        if inc_time > 0 and fw_time > 0:
            improvement = (inc_time - fw_time) / inc_time * 100
            print(f"Frank-Wolfe相比增量分配提升: {improvement:.2f}%")
            
            if improvement < -2:  # 允许2%的误差
                print("⚠️ 警告: Frank-Wolfe结果比增量分配差，可能存在收敛问题")
            elif improvement > 0:
                print("✅ Frank-Wolfe达到预期效果，优于增量分配")
            else:
                print("ℹ️ Frank-Wolfe与增量分配结果相近")

    
# ========== 新增的图表展示功能 ==========
    
    def visualize_network_structure(self, save_path: str = "./outputs/network_structure.png"):
        """可视化路网结构"""
        plt.figure(figsize=(12, 10))
        
        # 创建NetworkX图
        G = nx.DiGraph()
        
        # 添加节点
        for node_id, node in self.nodes.items():
            G.add_node(node_id, pos=(node['x'], node['y']))
        
        # 添加边
        for link_id, link in self.links.items():
            if link['flow'] == 0:  # 只显示有流量的边
                continue
            G.add_edge(link['from'], link['to'], 
                      weight=link['length'],
                      capacity=link['capacity'],
                      flow=link['flow'])
        
        # 获取位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', 
                              edgecolors='black', linewidths=2)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
        
        # 绘制边
        edge_colors = []
        edge_widths = []
        
        for (u, v, data) in G.edges(data=True):
            # 根据饱和度决定颜色
            saturation = data['flow'] / data['capacity'] if data['capacity'] > 0 else 0
            if saturation < 0.5:
                edge_colors.append('green')
            elif saturation < 0.8:
                edge_colors.append('orange')
            else:
                edge_colors.append('red')
            
            # 根据流量决定线宽
            edge_widths.append(2 + 3 * saturation)
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              width=edge_widths, arrows=True, 
                              arrowsize=20, arrowstyle='->')
        
        # 添加边标签（流量/容量）
        edge_labels = {}
        for (u, v, data) in G.edges(data=True):
            edge_labels[(u, v)] = f"{data['flow']:.0f}/{data['capacity']}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        # 添加标题和说明
        plt.title("Road Network Structure with Traffic Flow", fontsize=16, fontweight='bold')
        plt.xlabel("X Coordinate (km)", fontsize=12)
        plt.ylabel("Y Coordinate (km)", fontsize=12)
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color='green', label='Low saturation (<0.5)'),
            mpatches.Patch(color='orange', label='Medium saturation (0.5-0.8)'),
            mpatches.Patch(color='red', label='High saturation (>0.8)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"网络结构图已保存到: {save_path}")
    
    def visualize_algorithm_comparison(self, aon_result, inc_result, fw_result, save_path: str = "./outputs/algorithm_comparison.png"):
        """可视化三种算法的比较结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 算法名称和结果
        algorithms = ['All-or-Nothing', 'Incremental', 'Frank-Wolfe UE']
        travel_times = [aon_result['total_travel_time'], 
                       inc_result['total_travel_time'], 
                       fw_result['total_travel_time']]
        
        # 1. 总出行时间比较柱状图
        ax1 = axes[0, 0]
        bars = ax1.bar(algorithms, travel_times, color=['skyblue', 'lightgreen', 'salmon'])
        ax1.set_title('Total Travel Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Travel Time (veh-hr)', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值
        for bar, time in zip(bars, travel_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{time:.1f}', ha='center', va='bottom', fontsize=11)
        
        # 2. 路段流量对比散点图
        ax2 = axes[0, 1]
        
        # 收集三种算法的路段流量
        links = sorted(self.links.keys())
        aon_flows = [self.links[link]['flow'] for link in links]
        
        # 保存当前流量并计算其他算法的流量
        temp_flows = {link_id: link['flow'] for link_id, link in self.links.items()}
        
        # 计算增量分配的流量
        self.reset_flows()
        inc_result_temp = self.incremental_assignment(increments=4)
        inc_flows = [self.links[link]['flow'] for link in links]
        
        # 计算Frank-Wolfe的流量
        self.reset_flows()
        fw_result_temp = self.frank_wolfe_equilibrium(max_iterations=50, tolerance=1e-3)
        fw_flows = [self.links[link]['flow'] for link in links]
        
        # 恢复原始流量
        for link_id, flow in temp_flows.items():
            self.links[link_id]['flow'] = flow
        
        # 绘制散点图
        capacities = [self.links[link]['capacity'] for link in links]
        
        ax2.scatter(capacities, aon_flows, alpha=0.7, s=100, label='All-or-Nothing', c='skyblue', edgecolors='black')
        ax2.scatter(capacities, inc_flows, alpha=0.7, s=100, label='Incremental', c='lightgreen', edgecolors='black', marker='s')
        ax2.scatter(capacities, fw_flows, alpha=0.7, s=100, label='Frank-Wolfe UE', c='salmon', edgecolors='black', marker='^')
        
        # 添加容量线
        max_capacity = max(capacities)
        max_flow = max(max(aon_flows), max(inc_flows), max(fw_flows))
        ax2.plot([0, max_capacity*1.1], [0, max_capacity*1.1], 'r--', alpha=0.5, label='Capacity limit')
        
        ax2.set_title('Link Flow vs Capacity', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Capacity (veh/hr)', fontsize=12)
        ax2.set_ylabel('Assigned Flow (veh/hr)', fontsize=12)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. 饱和度分布箱线图
        ax3 = axes[1, 0]
        
        # 计算饱和度
        aon_saturations = [flow/cap if cap > 0 else 0 for flow, cap in zip(aon_flows, capacities)]
        inc_saturations = [flow/cap if cap > 0 else 0 for flow, cap in zip(inc_flows, capacities)]
        fw_saturations = [flow/cap if cap > 0 else 0 for flow, cap in zip(fw_flows, capacities)]
        
        data = [aon_saturations, inc_saturations, fw_saturations]
        
        bp = ax3.boxplot(data, labels=algorithms, patch_artist=True)
        
        # 设置箱线图颜色
        colors = ['skyblue', 'lightgreen', 'salmon']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_title('Saturation Distribution by Algorithm', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Saturation (Flow/Capacity)', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 算法性能雷达图
        ax4 = axes[1, 1]
        
        # 计算各项指标（归一化到0-1）
        # 假设：总出行时间越低越好，饱和度标准差越小越好，最高饱和度越小越好，收敛性越好
        max_travel_time = max(travel_times)
        normalized_travel_times = [1 - t/max_travel_time for t in travel_times]
        
        # 计算饱和度标准差
        sat_stds = [np.std(sats) for sats in [aon_saturations, inc_saturations, fw_saturations]]
        max_std = max(sat_stds)
        normalized_stds = [1 - std/max_std for std in sat_stds]
        
        # 计算最高饱和度
        max_sats = [max(sats) for sats in [aon_saturations, inc_saturations, fw_saturations]]
        max_max_sat = max(max_sats)
        normalized_max_sats = [1 - ms/max_max_sat for ms in max_sats]
        
        # 收敛性评分（假设）
        convergence_scores = [0.3, 0.6, 0.9]  # All-or-Nothing不收敛，增量分配部分收敛，Frank-Wolfe收敛
        
        # 雷达图数据
        categories = ['Travel Time', 'Saturation Std', 'Max Saturation', 'Convergence']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # 为每个算法绘制雷达图
        for i, algorithm in enumerate(algorithms):
            values = [normalized_travel_times[i], normalized_stds[i], 
                     normalized_max_sats[i], convergence_scores[i]]
            values += values[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=algorithm)
            ax4.fill(angles, values, alpha=0.25)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Algorithm Performance Radar Chart', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True)
        
        plt.suptitle('Traffic Assignment Algorithm Comparison', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"算法比较图已保存到: {save_path}")
    
    def visualize_link_flow_distribution(self, results: Dict, algorithm_name: str, save_path: str = "./outputs/link_flow_distribution.png"):
        """可视化路段流量分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 准备数据
        links = sorted(self.links.keys())
        flows = [self.links[link]['flow'] for link in links]
        capacities = [self.links[link]['capacity'] for link in links]
        saturations = [flow/cap if cap > 0 else 0 for flow, cap in zip(flows, capacities)]
        
        # 1. 路段流量柱状图
        x = range(len(links))
        width = 0.35
        
        bars1 = ax1.bar(x, flows, width, label='Assigned Flow', color='steelblue', alpha=0.8)
        bars2 = ax1.bar([i + width for i in x], capacities, width, label='Capacity', color='lightcoral', alpha=0.6)
        
        ax1.set_xlabel('Links', fontsize=12)
        ax1.set_ylabel('Flow/Capacity (veh/hr)', fontsize=12)
        ax1.set_title(f'Link Flow Distribution - {algorithm_name}', fontsize=14, fontweight='bold')
        ax1.set_xticks([i + width/2 for i in x])
        ax1.set_xticklabels(links, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加流量数值
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 饱和度热力图
        im = ax2.imshow([saturations], cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1.5)
        
        ax2.set_xticks(range(len(links)))
        ax2.set_xticklabels(links, rotation=45)
        ax2.set_yticks([0])
        ax2.set_yticklabels(['Saturation'])
        ax2.set_title('Link Saturation Heatmap', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2, orientation='vertical')
        cbar.set_label('Saturation (Flow/Capacity)', fontsize=12)
        
        # 在热力图上添加数值
        for i, sat in enumerate(saturations):
            color = 'white' if sat > 0.7 else 'black'
            ax2.text(i, 0, f'{sat:.2f}', ha='center', va='center', 
                    color=color, fontweight='bold')
        
        plt.suptitle(f'Traffic Assignment Results - {algorithm_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"路段流量分布图已保存到: {save_path}")
    
    def visualize_convergence_history(self, convergence_history: List[Dict], save_path: str = "./outputs/convergence_history.png"):
        """可视化Frank-Wolfe算法的收敛历史"""
        if not convergence_history:
            print("没有收敛历史数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        iterations = [h['iteration'] for h in convergence_history]
        relative_gaps = [h['relative_gap'] for h in convergence_history]
        objectives = [h['objective'] for h in convergence_history]
        travel_times = [h['travel_time'] for h in convergence_history]
        
        # 1. 相对对偶间隙收敛图
        ax1.semilogy(iterations, relative_gaps, 'b-o', linewidth=2, markersize=8)
        ax1.axhline(y=1e-3, color='r', linestyle='--', alpha=0.7, label='Convergence threshold (1e-3)')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Relative Gap (log scale)', fontsize=12)
        ax1.set_title('Frank-Wolfe Algorithm Convergence', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标记收敛点
        for i, gap in enumerate(relative_gaps):
            if gap < 1e-3:
                ax1.plot(iterations[i], gap, 'ro', markersize=10, markeredgewidth=2, 
                        markeredgecolor='black', label=f'Converged at iteration {iterations[i]}' if i == 0 else "")
                break
        
        # 2. 目标函数和总出行时间变化图
        ax2.plot(iterations, objectives, 'g-s', linewidth=2, markersize=8, label='Objective Function')
        ax2.plot(iterations, travel_times, 'r-^', linewidth=2, markersize=8, label='Total Travel Time')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('Objective Function and Travel Time Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 添加数值标签
        for i in [0, len(iterations)-1]:
            ax2.text(iterations[i], objectives[i], f'{objectives[i]:.0f}', 
                    ha='center', va='bottom', fontsize=10)
            ax2.text(iterations[i], travel_times[i], f'{travel_times[i]:.0f}', 
                    ha='center', va='top', fontsize=10)
        
        plt.suptitle('Frank-Wolfe Algorithm Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"收敛历史图已保存到: {save_path}")
    
    def visualize_od_demand_matrix(self, save_path: str = "./outputs/od_demand_matrix.png"):
        """可视化OD需求矩阵"""
        # 创建OD矩阵
        nodes = sorted(self.nodes.keys())
        od_matrix = np.zeros((len(nodes), len(nodes)))
        
        # 填充OD矩阵
        node_index = {node: i for i, node in enumerate(nodes)}
        for demand in self.demands:
            i = node_index[demand['from']]
            j = node_index[demand['to']]
            od_matrix[i, j] = demand['amount']
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(od_matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置刻度
        ax.set_xticks(range(len(nodes)))
        ax.set_yticks(range(len(nodes)))
        ax.set_xticklabels(nodes)
        ax.set_yticklabels(nodes)
        
        # 添加标题和标签
        ax.set_title('Origin-Destination Demand Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Destination', fontsize=12)
        ax.set_ylabel('Origin', fontsize=12)
        
        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Demand (veh/hr)', rotation=90, va='bottom')
        
        # 在热力图上添加数值
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if od_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{od_matrix[i, j]:.0f}',
                                  ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"OD需求矩阵图已保存到: {save_path}")
    
    def visualize_path_usage(self, results: Dict, algorithm_name: str, save_path: str = "./outputs/path_usage.png"):
        """可视化路径使用情况"""
        if not results.get('used_paths'):
            print(f"{algorithm_name} 没有找到使用的路径")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 准备数据
        paths = list(results['used_paths'].keys())
        flows = [results['used_paths'][p]['flow'] for p in paths]
        travel_times = [results['used_paths'][p]['travel_time'] for p in paths]
        
        # 1. 路径流量饼图
        ax1.pie(flows, labels=paths, autopct='%1.1f%%', startangle=90, 
                colors=plt.cm.Set3(np.linspace(0, 1, len(paths))))
        ax1.set_title(f'Path Flow Distribution - {algorithm_name}', fontsize=14, fontweight='bold')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # 2. 路径行程时间与流量关系图
        scatter = ax2.scatter(travel_times, flows, s=[f/10 for f in flows], 
                             alpha=0.7, c=travel_times, cmap='viridis', edgecolors='black')
        
        # 添加路径标签
        for i, (path, flow, time) in enumerate(zip(paths, flows, travel_times)):
            # 简化路径标签显示
            simple_path = path.replace('->', '-')
            ax2.annotate(simple_path, (time, flow), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        ax2.set_xlabel('Travel Time (hours)', fontsize=12)
        ax2.set_ylabel('Flow (veh/hr)', fontsize=12)
        ax2.set_title('Path Travel Time vs Flow', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Travel Time (hours)', fontsize=12)
        
        plt.suptitle(f'Path Usage Analysis - {algorithm_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"路径使用情况图已保存到: {save_path}")
    
    def compare_algorithms(self, aon_result: Dict, inc_result: Dict, fw_result: Dict):
        """比较三种算法的结果"""
        print(f"\n{'='*80}")
        print("三种交通分配算法对比分析")
        print(f"{'='*80}")
        
        print(f"\n算法性能对比:")
        print(f"{'算法名称':<25} {'总出行时间(车辆-小时)':<25} {'计算复杂度':<15} {'收敛性':<10}")
        print(f"{'-'*80}")
        
        algorithms = [
            ("全有全无分配", aon_result['total_travel_time'], "低", "不收敛"),
            ("增量分配", inc_result['total_travel_time'], "中", "部分收敛"), 
            ("Frank-Wolfe均衡", fw_result['total_travel_time'], "高", "收敛")
        ]
        
        for name, travel_time, complexity, convergence in algorithms:
            print(f"{name:<25} {travel_time:<25.2f} {complexity:<15} {convergence:<10}")
        
        aon_time = aon_result['total_travel_time']
        inc_time = inc_result['total_travel_time']
        fw_time = fw_result['total_travel_time']
        
        print(f"\n效率提升分析:")
        if aon_time > 0 and inc_time > 0:
            improvement_inc = (aon_time - inc_time) / aon_time * 100
            print(f"增量分配相比全有全无分配提升: {improvement_inc:.2f}%")
        
        if aon_time > 0 and fw_time > 0:
            improvement_fw = (aon_time - fw_time) / aon_time * 100
            print(f"Frank-Wolfe相比全有全无分配提升: {improvement_fw:.2f}%")
        
        if inc_time > 0 and fw_time > 0:
            improvement = (inc_time - fw_time) / inc_time * 100
            print(f"Frank-Wolfe相比增量分配提升: {improvement:.2f}%")
            
            if improvement < -2:  # 允许2%的误差
                print("⚠️ 警告: Frank-Wolfe结果比增量分配差，可能存在收敛问题")
            elif improvement > 0:
                print("✅ Frank-Wolfe达到预期效果，优于增量分配")
            else:
                print("ℹ️ Frank-Wolfe与增量分配结果相近")


# 使用示例
if __name__ == "__main__":
    network = RoadNetwork()
    
    try:
        network.load_network("network.json")
        network.load_demand("demand.json")
        
        network.text_visualization()
        
        # 全有全无分配
        print("\n" + "="*60)
        print("开始执行全有全无分配")
        print("="*60)
        
        # 先执行单OD分配
        network.print_single_od_results("全有全无分配")
        
        # 再执行整体分配
        aon_result = network.all_or_nothing_assignment()
        network.print_assignment_results(aon_result, "全有全无分配")
        
        # 增量分配
        print("\n" + "="*60)
        print("开始执行增量分配")
        print("="*60)
        
        # 先执行单OD分配
        network.print_single_od_results("增量分配")
        
        # 再执行整体分配
        inc_result = network.incremental_assignment(increments=4)
        network.print_assignment_results(inc_result, "增量分配")
        
        # Frank-Wolfe用户均衡分配
        print("\n" + "="*60)
        print("开始执行Frank-Wolfe用户均衡分配")
        print("="*60)
        
        # 先执行单OD分配
        network.print_single_od_results("Frank-Wolfe用户均衡分配")
        
        # 再执行整体分配
        fw_result = network.frank_wolfe_equilibrium(max_iterations=50, tolerance=1e-3)
        network.print_assignment_results(fw_result, "Frank-Wolfe用户均衡分配")
        
        network.compare_algorithms(aon_result, inc_result, fw_result)

        # 可视化Frank-Wolfe分配结果
        network.visualize_link_flow_distribution(fw_result, "Frank-Wolfe User Equilibrium")
        network.visualize_path_usage(fw_result, "Frank-Wolfe User Equilibrium")
        
        # 可视化收敛历史
        if 'convergence_history' in fw_result:
            network.visualize_convergence_history(fw_result['convergence_history'])
        
        # 可视化网络结构
        network.visualize_network_structure()
        
        # 算法比较
        network.compare_algorithms(aon_result, inc_result, fw_result)
        
        # 可视化算法比较
        network.visualize_algorithm_comparison(aon_result, inc_result, fw_result)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()