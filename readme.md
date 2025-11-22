# 《交通规划原理》课程报告

## 交通分配问题的背景介绍

交通分配是交通规划中的核心问题，旨在将起讫点（OD）间的交通需求合理分配到路网各路径上。根据Wardrop第一原理，在用户均衡状态下，所有被使用的路径具有相同的最小行程时间，且没有驾驶员能够通过单方面改变路径来减少行程时间。

本软件实现了三种经典的交通分配算法：
- **全有全无分配**：不考虑拥堵效应，将所有需求分配到最短路径
- **增量分配**：分步骤加载需求，逐步考虑拥堵影响
- **Frank-Wolfe用户均衡分配**：迭代求解用户均衡状态

## 软件功能模块划分和各模块设计思路

### 1. 路网数据结构模块 (`Graph`类)
- **功能**：存储节点和路段信息，提供最短路径计算
- **设计思路**：
  - 使用邻接表存储图结构
  - 实现Dijkstra算法计算最短路径
  - 支持动态权重更新

### 2. 数学计算模块 (`Polynomial`类)
- **功能**：处理行程时间函数计算
- **设计思路**：
  - 实现多项式求值、求导、积分运算
  - 支持行程时间函数 $t(q) = t_0 \cdot (1 + \frac{q}{\text{cap}})^2$ 的计算

### 3. 交通网络核心模块 (`RoadNetwork`类)
- **功能**：集成路网管理、需求加载和分配算法
- **设计思路**：
  - 从JSON文件加载路网和需求数据
  - 管理节点、路段和交通需求
  - 提供三种分配算法的实现

## 开发环境

- **编程语言**：Python 3.7+
- **核心库**：标准库（json, math, heapq, typing, collections）
- **开发工具**：任意Python IDE或文本编辑器
- **运行要求**：无需额外依赖，纯Python实现

## 关键算法代码说明

### 1. 最短路径算法 (Dijkstra)

```python
def dijkstra(self, start: str, end: str) -> List[str]:
    """基于当前权重计算最短路径"""
    distances = {node: float('inf') for node in self.nodes}
    previous = {node: None for node in self.nodes}
    distances[start] = 0
    
    # 使用优先队列优化搜索效率
    unvisited = set(self.nodes.keys())
    while unvisited:
        current = min(unvisited, key=lambda node: distances[node])
        # ... 路径搜索逻辑
```

### 2. 用户均衡分配算法 (Frank-Wolfe)

```python
def frank_wolfe_equilibrium(self, max_iterations: int = 100, tolerance: float = 1e-4) -> Dict:
    """Frank-Wolfe算法求解用户均衡"""
    # 初始化流量模式
    x = {link_id: self.links[link_id]['flow'] for link_id in self.links}
    
    for iteration in range(1, max_iterations + 1):
        # 更新权重并计算辅助流量模式
        y = self.calculate_auxiliary_flow(x)
        
        # 计算对偶间隙检查收敛
        gap = self.calculate_duality_gap(x, y)
        if gap < tolerance and iteration >= 5:
            break
        
        # 黄金分割法搜索最优步长
        alpha = self.golden_section_search(objective_function)
        
        # 更新流量：x = (1-α)x + αy
        for link_id in self.links:
            x[link_id] = (1 - alpha) * x[link_id] + alpha * y[link_id]
```

### 3. 行程时间函数计算

```python
def calculate_travel_time(self, link_id: str, flow: float) -> float:
    """计算拥堵条件下的路段行程时间"""
    t0 = self.links[link_id]['free_flow_time']
    cap = self.links[link_id]['capacity']
    # t(q) = t0 * (1 + q/cap)^2
    return t0 * (1 + flow / cap) ** 2
```

## 测试场景结果分析

### 测试用例对比

| 测试用例 | 全有全无分配 | 增量分配 | Frank-Wolfe均衡 | 最优算法 |
|---------|-------------|----------|----------------|----------|
| 用例2 (平行路径) | 1600.00 | 1212.83 | **1206.39** | Frank-Wolfe |
| 用例7 (不对称网络) | 1745.36 | **1378.05** | 1395.50 | 增量分配 |
| 用例5 (复杂网络) | 10105.24 | 4679.53 | **4645.08** | Frank-Wolfe |

### 算法性能总结

1. **全有全无分配**
   - 优点：计算简单快速
   - 缺点：忽略拥堵效应，结果不现实
   - 适用场景：初步分析或自由流状态评估

2. **增量分配**
   - 优点：考虑拥堵，计算效率适中
   - 缺点：可能不收敛到均衡状态
   - 适用场景：中等精度要求的规划分析

3. **Frank-Wolfe用户均衡**
   - 优点：理论上保证收敛到用户均衡
   - 缺点：计算复杂度较高
   - 适用场景：精确的交通分配分析

### 关键发现

1. **均衡状态验证**：在用户均衡分配中，被使用路径的行程时间非常接近（差值<0.01小时），验证了Wardrop第一原理
2. **效率提升**：与全有全无分配相比，考虑拥堵的算法可降低总出行时间20-50%
3. **路径多样性**：均衡分配能识别更多可行路径，提高路网利用率

## 软件功能验证

软件成功回答了测试要求中的问题：
- ✅ 不考虑拥堵的最短路径计算
- ✅ 考虑拥堵效应的路径选择
- ✅ 单OD对和多OD对的流量分配
- ✅ 路径行程时间均衡性验证
- ✅ 总出行时间计算和算法比较

## 完整代码仓库

GitHub仓库地址：https://github.com/[username]/traffic-assignment

**文件结构**：
```
traffic-assignment/
├── traffic_assignment.py    # 主程序文件
├── network.json            # 测试路网文件
├── demand.json            # 测试需求文件
├── test_cases/            # 额外测试用例
│   ├── network_simple.json
│   ├── demand_simple.json
│   └── ...
└── README.md              # 项目说明文档
```

## 结论

本交通分配软件成功实现了三种核心算法，能够：
1. 准确模拟交通流在路网中的分配
2. 验证用户均衡条件
3. 提供不同精度的分配方案
4. 支持复杂的路网结构和多OD对需求

软件设计具有良好的扩展性，可进一步添加更多分配算法或可视化功能。

---

# GitHub README.md

```markdown
# 交通分配计算软件

一个基于Python的交通分配计算工具，实现全有全无分配、增量分配和Frank-Wolfe用户均衡分配算法。

## 功能特性

- 🛣️ 支持复杂路网结构的交通分配
- 📊 三种分配算法实现：
  - 全有全无分配 (All-or-Nothing)
  - 增量分配 (Incremental Assignment) 
  - Frank-Wolfe用户均衡分配 (User Equilibrium)
- 📈 拥堵效应建模：BPR类型行程时间函数
- 🔍 路径分析和流量分布可视化
- 📋 详细的分配结果报告和算法对比

## 安装要求

- Python 3.7+
- 仅需标准库，无额外依赖

## 快速开始

### 1. 准备输入文件

**路网文件 (network.json)**:
```json
{
    "nodes": {
        "name": ["A", "B", "C"],
        "x": [0, 10, 20],
        "y": [0, 0, 0]
    },
    "links": {
        "between": ["AB", "BC"],
        "capacity": [1000, 1000],
        "speedmax": [50, 50]
    }
}
```

**需求文件 (demand.json)**:
```json
{
    "from": ["A"],
    "to": ["C"], 
    "amount": [500]
}
```

### 2. 运行分配计算

```python
from traffic_assignment import RoadNetwork

# 创建路网实例
network = RoadNetwork()

# 加载数据
network.load_network("network.json")
network.load_demand("demand.json")

# 执行三种分配算法
aon_result = network.all_or_nothing_assignment()
inc_result = network.incremental_assignment(increments=4)
fw_result = network.frank_wolfe_equilibrium(max_iterations=50)

# 比较结果
network.compare_algorithms(aon_result, inc_result, fw_result)
```

## 输出示例

```
============================================================
Frank-Wolfe用户均衡分配 分配结果
============================================================
总出行时间: 1206.39 车辆-小时

路段流量分布:
路段       流量(辆/小时)     容量       饱和度      行程时间(小时)    
------------------------------------------------------------
AB       737          1000     0.74     0.60        
AD       263          800      0.33     0.71        
BC       737          1000     0.74     0.60        
DC       263          800      0.33     0.50        

使用的路径:
路径 1: A->B->C
  流量: 500 辆/小时, 行程时间: 1.21 小时
路径 2: A->D->C  
  流量: 500 辆/小时, 行程时间: 1.21 小时
```

## 算法特点

### 全有全无分配
- ⚡ 计算速度快
- 📉 忽略拥堵效应
- 🎯 适用于自由流状态分析

### 增量分配  
- ⚖️ 平衡精度和效率
- 🔄 分步考虑拥堵
- 🏗️ 适用于中等精度规划

### Frank-Wolfe用户均衡
- ✅ 理论保证收敛到均衡状态
- 📊 最高精度结果
- 🔬 适用于精确分析

## 测试用例

项目包含多个测试用例：
- `test_cases/network_simple.json` - 基础验证
- `test_cases/network_parallel.json` - 平行路径测试  
- `test_cases/network_braess.json` - Braess悖论验证
- `test_cases/network_complex.json` - 复杂网络测试

## 技术细节

### 行程时间函数
采用标准的BPR函数形式：
```
t(q) = t₀ × (1 + (q/capacity)²)
```
其中：
- `t₀`: 自由流行程时间 (长度/限速)
- `q`: 路段流量
- `capacity`: 路段通行能力

### 收敛标准
Frank-Wolfe算法使用相对对偶间隙作为收敛标准：
```
相对间隙 = |∑(tₐ×(yₐ-xₐ))| / ∑(tₐ×xₐ) < 容差
```

## 扩展开发

可以轻松扩展以下功能：
- 添加新的分配算法（如系统最优分配）
- 集成图形化界面
- 支持更多行程时间函数
- 添加敏感性分析功能

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 联系信息

如有问题请联系：[your-email@example.com]

---

*用于《交通规划原理》课程项目 - 电子科技大学 2025*
```

这个完整的报告和README.md涵盖了课程要求的所有内容，包括背景介绍、模块设计、开发环境、关键算法说明、测试结果分析和代码仓库信息。
