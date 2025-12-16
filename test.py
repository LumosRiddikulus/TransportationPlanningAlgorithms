import sys
import os
import json
from contextlib import redirect_stdout

def create_test_files():
    """创建所有测试文件"""
    test_cases = [
        {
            "name": "simple_two_node",
            "network": {
                "nodes": {"name": ["A", "B"], "x": [0, 10], "y": [0, 0]},
                "links": {"between": ["AB"], "capacity": [1800], "speedmax": [30]}
            },
            "demand": {
                "from": ["A", "B"],
                "to": ["B", "A"],
                "amount": [1000, 500]
            }
        },
        {
            "name": "cross_road",
            "network": {
                "nodes": {"name": ["A", "B", "C", "D", "E"], 
                         "x": [0, 10, 20, 10, 10], 
                         "y": [10, 0, 10, 10, 20]},
                "links": {
                    "between": ["AB", "BC", "AD", "DE", "BD", "BE", "CE"],
                    "capacity": [1800, 1800, 1800, 1800, 1800, 3600, 1800],
                    "speedmax": [30, 30, 30, 30, 30, 60, 30]
                }
            },
            "demand": {
                "from": ["A", "C", "A", "E", "C", "E"],
                "to": ["C", "A", "E", "A", "C", "E"],
                "amount": [1500, 800, 1000, 600, 1200, 700]
            }
        },
        {
            "name": "grid_network",
            "network": {
                "nodes": {
                    "name": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                    "x": [0, 10, 20, 0, 10, 20, 0, 10, 20],
                    "y": [0, 0, 0, 10, 10, 10, 20, 20, 20]
                },
                "links": {
                    "between": ["AB", "BC", "AD", "BE", "CF", "DG", "EH", "FI", "DE", "EF", "GH", "HI"],
                    "capacity": [1800, 1800, 1800, 3600, 1800, 1800, 1800, 1800, 1800, 3600, 1800, 1800],
                    "speedmax": [30, 30, 30, 60, 30, 30, 30, 30, 30, 60, 30, 30]
                }
            },
            "demand": {
                "from": ["A", "I", "C", "G", "A", "I", "B", "H", "E"],
                "to": ["I", "A", "G", "C", "C", "G", "H", "B", "E"],
                "amount": [2000, 1500, 800, 1200, 1000, 600, 900, 700, 500]
            }
        },
        {
            "name": "asymmetric_demand",
            "network": {
                "nodes": {"name": ["A", "B", "C", "D"], 
                         "x": [0, 10, 20, 10], 
                         "y": [0, 0, 0, 10]},
                "links": {
                    "between": ["AB", "BC", "BD", "CD"],
                    "capacity": [1800, 1800, 3600, 1800],
                    "speedmax": [30, 30, 60, 30]
                }
            },
            "demand": {
                "from": ["A", "C", "B"],
                "to": ["C", "D", "D"],
                "amount": [2000, 1000, 1500]
            }
        },
        {
            "name": "duplicate_road_network",
            "network": {
                "nodes": {
                    "name": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
                    "x": [0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 20, 30],
                    "y": [0, 0, 0, 0, 10, 10, 10, 10, 20, 20, 20, 20]
                },
                "links": {
                    "between": ["AB", "BC", "CD", "AE", "BF", "CG", "DH", "EF", "FG", "GH", "EI", "FJ", "GK", "HL", "IJ", "JK", "KL"],
                    "capacity": [1800, 3600, 1800, 1800, 3600, 1800, 1800, 1800, 3600, 1800, 1800, 3600, 1800, 1800, 1800, 1800, 1800],
                    "speedmax": [30, 60, 30, 30, 60, 30, 30, 30, 60, 30, 30, 60, 30, 30, 30, 30, 30]
                }
            },
            "demand": {
                "from": ["A", "L", "A", "D", "E", "H", "I", "L", "B", "K", "C", "J"],
                "to": ["L", "A", "D", "A", "H", "E", "L", "I", "K", "B", "J", "C"],
                "amount": [2500, 1500, 1200, 800, 1000, 600, 900, 700, 1100, 900, 800, 700]
            }
        },
        {
            "name": "network_with_dead_tie",
            "network": {
                "nodes": {
                    "name": ["A", "B", "C", "D", "E", "F"],
                    "x": [0, 10, 20, 30, 10, 20],
                    "y": [0, 0, 0, 0, 10, 10]
                },
                "links": {
                    "between": ["AB", "BC", "CD", "BE", "CF"],
                    "capacity": [1800, 1800, 1800, 1800, 1800],
                    "speedmax": [30, 30, 30, 30, 30]
                }
            },
            "demand": {
                "from": ["A", "D", "E", "F"],
                "to": ["D", "A", "F", "E"],
                "amount": [2000, 1000, 800, 600]
            }
        },
        {
            "name": "single_direction_network",
            "network": {
                "nodes": {
                    "name": ["A", "B", "C", "D"],
                    "x": [0, 10, 10, 0],
                    "y": [0, 0, 10, 10]
                },
                "links": {
                    "between": ["AB", "BC", "CD", "DA"],
                    "capacity": [1800, 1800, 1800, 1800],
                    "speedmax": [30, 30, 30, 30]
                }
            },
            "demand": {
                "from": ["A", "B", "C", "D"],
                "to": ["C", "D", "A", "B"],
                "amount": [1500, 1200, 900, 600]
            }
        }
    ]
    
    # 创建测试文件
    for i, test_case in enumerate(test_cases, 1):
        network_file = f"network{i}.json"
        demand_file = f"demand{i}.json"
        
        with open(network_file, 'w') as f:
            json.dump(test_case["network"], f, indent=2)
        
        with open(demand_file, 'w') as f:
            json.dump(test_case["demand"], f, indent=2)
        
        print(f"创建测试用例 {i}: {test_case['name']}")
        print(f"  路网文件: {network_file}")
        print(f"  需求文件: {demand_file}")
        print(f"  节点数: {len(test_case['network']['nodes']['name'])}")
        print(f"  路段数: {len(test_case['network']['links']['between'])}")
        print(f"  OD对数量: {len(test_case['demand']['from'])}")
        print()


def run_all_tests_with_output():
    """运行所有测试并将输出保存到文件"""
    from traffic_distribution import RoadNetwork  # 假设您的代码保存为road_network_code.py
    
    # 设置输出文件
    output_file = "test_results.txt"
    
    # 使用redirect_stdout将标准输出重定向到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            for i in range(1, 8):  # 运行前4个测试用例
                network_file = f"network{i}.json"
                demand_file = f"demand{i}.json"
                
                if not os.path.exists(network_file) or not os.path.exists(demand_file):
                    print(f"测试用例 {i} 的文件不存在，跳过...")
                    continue
                
                print(f"\n{'='*80}")
                print(f"开始运行测试用例 {i}")
                print(f"{'='*80}")
                
                try:
                    network = RoadNetwork()
                    network.load_network(network_file)
                    network.load_demand(demand_file)
                    
                    # 执行全有全无分配
                    aon_result = network.all_or_nothing_assignment()
                    network.print_assignment_results(aon_result, f"Test {i} - All-or-Nothing")
                    
                    # 执行增量分配
                    inc_result = network.incremental_assignment(increments=4)
                    network.print_assignment_results(inc_result, f"Test {i} - Incremental")
                    
                    # 执行Frank-Wolfe用户均衡分配
                    fw_result = network.frank_wolfe_equilibrium(max_iterations=50, tolerance=1e-3)
                    network.print_assignment_results(fw_result, f"Test {i} - Frank-Wolfe")
                    
                    # 比较算法
                    network.compare_algorithms(aon_result, inc_result, fw_result)
                    
                except Exception as e:
                    print(f"测试用例 {i} 执行失败: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"所有测试结果已保存到: {output_file}")

if __name__ == "__main__":
    print("创建测试用例文件...")
    create_test_files()
    
    print("\n运行所有测试用例...")
    run_all_tests_with_output()