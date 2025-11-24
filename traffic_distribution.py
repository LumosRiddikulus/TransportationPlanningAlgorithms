import json
import math
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Callable
from collections import defaultdict

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
                    'amount': data['amount'][i]
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
    
    def all_or_nothing_assignment(self) -> Dict:
        """全有全无分配算法"""
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
    
    def incremental_assignment(self, increments: int = 4) -> Dict:
        """增量分配算法"""
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
        """改进的Frank-Wolfe用户均衡分配算法"""
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
    
    def visualize_network(self, results: Dict, algorithm_name: str, save_path: str = None):
        """可视化路网和分配结果"""
        plt.figure(figsize=(14, 10))
        
        # 创建networkx图
        G = nx.DiGraph()
        
        # 添加节点
        for node_id, node_info in self.nodes.items():
            G.add_node(node_id, pos=(node_info['x'], node_info['y']))
        
        # 添加边并设置属性
        edge_flows = []
        edge_times = []
        edge_widths = []
        edge_colors = []
        
        for link_id, link in self.links.items():
            flow = link['flow']
            travel_time = self.calculate_travel_time(link_id, flow)
            saturation = flow / link['capacity'] if link['capacity'] > 0 else 0
            
            G.add_edge(link['from'], link['to'], 
                      flow=flow, 
                      travel_time=travel_time,
                      saturation=saturation)
            
            edge_flows.append(flow)
            edge_times.append(travel_time)
            edge_colors.append(saturation)
            # 线宽基于流量
            edge_widths.append(1 + 3 * saturation)
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 绘制边 - 颜色表示饱和度
        edges = nx.draw_networkx_edges(G, pos, 
                                      edge_color=edge_colors,
                                      edge_cmap=plt.cm.Reds,
                                      edge_vmin=0,
                                      edge_vmax=1,
                                      width=edge_widths,
                                      arrows=True,
                                      arrowsize=20,
                                      connectionstyle="arc3,rad=0.1")
        
        # 修复颜色条问题：创建一个ScalarMappable对象
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array(edge_colors)
        
        # 添加颜色条
        cb = plt.colorbar(sm, label='路段饱和度', shrink=0.8)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.9)
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # 添加边标签（流量和行程时间）
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = f"{data['flow']:.0f}\n{data['travel_time']:.2f}h"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # 设置标题和说明
        plt.title(f"{algorithm_name} - 路网流量分配可视化\n"
                 f"总出行时间: {results['total_travel_time']:.2f} 车辆-小时", 
                 fontsize=14, fontweight='bold')
        
        # 添加图例说明
        plt.figtext(0.02, 0.02, 
                   "说明:\n"
                   "• 线宽表示流量大小\n"
                   "• 颜色深浅表示饱和度(红=高饱和度)\n"
                   "• 边标签: 流量(辆/小时)\\n行程时间(小时)", 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存至: {save_path}")
        
        plt.show()
    
    def plot_convergence(self, convergence_history: List[Dict], algorithm_name: str, save_path: str = None):
        """绘制算法收敛过程"""
        if not convergence_history:
            print("无收敛历史数据可绘制")
            return
        
        iterations = [item['iteration'] for item in convergence_history]
        gaps = [item['relative_gap'] for item in convergence_history]
        objectives = [item['objective'] for item in convergence_history]
        travel_times = [item['travel_time'] for item in convergence_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 绘制相对对偶间隙
        ax1.semilogy(iterations, gaps, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('相对对偶间隙 (对数尺度)')
        ax1.set_title(f'{algorithm_name} - 相对对偶间隙收敛过程')
        ax1.grid(True, alpha=0.3)
        
        # 绘制目标函数和总出行时间
        ax2.plot(iterations, objectives, 'r-o', linewidth=2, markersize=6, label='目标函数值')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(iterations, travel_times, 'g--s', linewidth=2, markersize=4, label='总出行时间')
        
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('目标函数值', color='r')
        ax2_twin.set_ylabel('总出行时间 (车辆-小时)', color='g')
        ax2.set_title(f'{algorithm_name} - 目标函数和总出行时间变化')
        ax2.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛图已保存至: {save_path}")
        
        plt.show()
    
    def plot_flow_comparison(self, aon_flows: Dict, inc_flows: Dict, fw_flows: Dict, save_path: str = None):
        """绘制三种算法流量对比"""
        link_ids = sorted(self.links.keys())
        
        aon_values = [aon_flows.get(link_id, 0) for link_id in link_ids]
        inc_values = [inc_flows.get(link_id, 0) for link_id in link_ids]
        fw_values = [fw_flows.get(link_id, 0) for link_id in link_ids]
        
        capacities = [self.links[link_id]['capacity'] for link_id in link_ids]
        
        x = np.arange(len(link_ids))
        width = 0.25
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 流量对比柱状图
        ax1.bar(x - width, aon_values, width, label='全有全无分配', alpha=0.8)
        ax1.bar(x, inc_values, width, label='增量分配', alpha=0.8)
        ax1.bar(x + width, fw_values, width, label='Frank-Wolfe均衡', alpha=0.8)
        
        # 添加容量线
        ax1.plot(x, capacities, 'r--', linewidth=2, label='路段容量', alpha=0.7)
        
        ax1.set_xlabel('路段')
        ax1.set_ylabel('流量 (辆/小时)')
        ax1.set_title('三种算法路段流量对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(link_ids, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 饱和度对比图
        aon_saturation = [aon_values[i] / capacities[i] if capacities[i] > 0 else 0 for i in range(len(link_ids))]
        inc_saturation = [inc_values[i] / capacities[i] if capacities[i] > 0 else 0 for i in range(len(link_ids))]
        fw_saturation = [fw_values[i] / capacities[i] if capacities[i] > 0 else 0 for i in range(len(link_ids))]
        
        ax2.bar(x - width, aon_saturation, width, label='全有全无分配', alpha=0.8)
        ax2.bar(x, inc_saturation, width, label='增量分配', alpha=0.8)
        ax2.bar(x + width, fw_saturation, width, label='Frank-Wolfe均衡', alpha=0.8)
        
        # 添加饱和线
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='饱和线 (饱和度=1.0)', alpha=0.7)
        
        ax2.set_xlabel('路段')
        ax2.set_ylabel('饱和度')
        ax2.set_title('三种算法路段饱和度对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(link_ids, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"流量对比图已保存至: {save_path}")
        
        plt.show()
    
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


# 使用示例
if __name__ == "__main__":
    network = RoadNetwork()
    
    try:
        # 加载路网和需求数据
        network.load_network("network.json")
        network.load_demand("demand.json")
        
        # 文本可视化路网
        network.text_visualization()
        
        # 执行全有全无分配
        print("\n" + "="*60)
        print("开始执行全有全无分配")
        print("="*60)
        aon_result = network.all_or_nothing_assignment()
        network.print_assignment_results(aon_result, "全有全无分配")
        network.visualize_network(aon_result, "全有全无分配", "aon_assignment.png")
        
        # 执行增量分配
        print("\n" + "="*60)
        print("开始执行增量分配")
        print("="*60)
        inc_result = network.incremental_assignment(increments=4)
        network.print_assignment_results(inc_result, "增量分配")
        network.visualize_network(inc_result, "增量分配", "incremental_assignment.png")
        
        # 执行Frank-Wolfe用户均衡分配
        print("\n" + "="*60)
        print("开始执行Frank-Wolfe用户均衡分配")
        print("="*60)
        fw_result = network.frank_wolfe_equilibrium(max_iterations=50, tolerance=1e-3)
        network.print_assignment_results(fw_result, "Frank-Wolfe用户均衡分配")
        network.visualize_network(fw_result, "Frank-Wolfe用户均衡分配", "frank_wolfe_assignment.png")
        
        # 绘制收敛过程
        if 'convergence_history' in fw_result:
            network.plot_convergence(fw_result['convergence_history'], 
                                   "Frank-Wolfe算法", 
                                   "convergence_plot.png")
        
        # 绘制流量对比
        aon_flows = {link_id: network.links[link_id]['flow'] for link_id in network.links}
        # 重新获取增量分配和Frank-Wolfe的流量
        network.incremental_assignment(increments=4)
        inc_flows = {link_id: network.links[link_id]['flow'] for link_id in network.links}
        network.frank_wolfe_equilibrium(max_iterations=50, tolerance=1e-3)
        fw_flows = {link_id: network.links[link_id]['flow'] for link_id in network.links}
        
        network.plot_flow_comparison(aon_flows, inc_flows, fw_flows, "flow_comparison.png")
        
        # 比较三种算法
        network.compare_algorithms(aon_result, inc_result, fw_result)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()