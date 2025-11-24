# 交通分配计算软件

一个基于Python，不需要安装任何额外库的交通分配计算工具，实现全有全无分配、增量分配和Frank-Wolfe用户均衡分配算法。

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

---

*用于《交通规划原理》课程项目 - 电子科技大学 2025*
