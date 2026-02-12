# CARLA轨迹预测项目 - 完整实现

## 📦 项目概述

这是一个完整的基于Transformer的轨迹预测系统,用于预测CARLA模拟器中车辆和行人的未来轨迹。

### 核心特性

✅ **完整的数据流程**
- CARLA自动数据收集
- 智能数据预处理和归一化
- 高效的PyTorch Dataset实现

✅ **先进的Transformer模型**
- 单模态预测 (TrajectoryTransformer)
- 多模态预测 (MultiModalTrajectoryTransformer)
- 支持车辆和行人两种agent类型

✅ **专业的训练系统**
- 自动早停和检查点保存
- 学习率调度
- 训练可视化和监控

✅ **完善的工具集**
- 命令行界面 (CLI)
- 可视化工具
- 模型测试脚本

---

## 📂 文件结构

```
carla_trajectory_prediction/
│
├── 📋 README.md                    # 完整文档
├── 🚀 QUICKSTART.md               # 快速开始指南
├── 📝 requirements.txt            # 依赖列表
├── 🎮 main.py                     # 主命令行接口
├── 🧪 test_model.py               # 模型测试(无需CARLA)
│
├── 📊 data_collection/            # 数据收集模块
│   ├── carla_data_collector.py   # CARLA数据收集器
│   └── data_preprocessor.py      # 数据预处理和Dataset
│
├── 🤖 models/                     # 模型定义
│   └── transformer_model.py      # Transformer模型
│
├── 🏋️ training/                   # 训练脚本
│   └── train.py                  # 训练器和训练循环
│
├── 🎨 utils/                      # 工具函数
│   └── visualization.py          # 可视化工具
│
└── ⚙️ configs/                    # 配置文件
    └── config.yaml               # 项目配置
```

---

## 🎯 使用流程

### 1️⃣ 数据收集
```bash
python main.py collect --duration 300
```
- 在CARLA中生成车辆和行人
- 自动记录轨迹数据
- 保存为JSON格式

### 2️⃣ 数据预处理
```bash
python main.py preprocess
```
- 提取观测-预测序列对
- 相对位置编码 (ego-centric)
- 构建交互图
- 保存为PyTorch可用的.pkl

### 3️⃣ 模型训练
```bash
python main.py train --augment
```
- 训练Transformer模型
- 实时监控ADE/FDE指标
- 自动保存最佳模型
- 生成训练曲线

### 4️⃣ 评估可视化
```bash
python main.py evaluate
```
- 加载训练好的模型
- 批量预测可视化
- 生成对比图表

---

## 🧠 模型架构详解

### TrajectoryTransformer (单模态)

```
输入特征 (8 frames × 7 dims):
  - Position (x, y)
  - Velocity (vx, vy)  
  - Speed
  - Agent Type (vehicle/pedestrian)

       ↓ Input Embedding (7 → 128)
       ↓ Positional Encoding
       ↓
  ┌─────────────────┐
  │ Encoder (4层)    │
  │  - Self-Attn    │
  │  - FFN          │
  └─────────────────┘
       ↓ Memory
  ┌─────────────────┐
  │ Decoder (4层)    │
  │  - Self-Attn    │
  │  - Cross-Attn   │
  │  - FFN          │
  └─────────────────┘
       ↓ Output Projection
       
输出: 预测轨迹 (12 frames × 2 dims)
```

**特点:**
- 使用相对坐标系(以最后观测位置为原点)
- Teacher forcing训练
- Causal masking for decoder
- 参数量: ~3M

### MultiModalTrajectoryTransformer (多模态)

```
      Shared Encoder
            ↓
   ┌────────┴────────┐
   ↓                 ↓
Mode 1 Decoder   Mode 2-6 Decoders
   ↓                 ↓
Trajectory 1    Trajectories 2-6
   ↓                 ↓
   └────────┬────────┘
            ↓
     Mode Probability
      Predictor
```

**特点:**
- 6个独立的解码器
- 每个模态有独立的query embedding
- Softmax归一化的模态概率
- 参数量: ~12M

---

## 📈 训练策略

### 损失函数
- **MSE Loss**: 预测位置与真实位置的均方误差
- 对于多模态: Winner-takes-all或mixture of experts

### 优化器
- **AdamW**: lr=1e-4, weight_decay=1e-5
- **Gradient Clipping**: max_norm=1.0
- **Scheduler**: ReduceLROnPlateau (patience=5)

### 数据增强
- 水平翻转 (50%概率)
- 小角度旋转 (±15度)
- 提升泛化能力

### 正则化
- Dropout: 0.1
- Weight decay: 1e-5
- Early stopping: patience=15

---

## 📊 评估指标

### ADE (Average Displacement Error)
```
ADE = (1/T) * Σ ||pred_t - gt_t||
```
- 所有时间步的平均误差
- 衡量整体预测质量

### FDE (Final Displacement Error)
```
FDE = ||pred_T - gt_T||
```
- 最后时间步的误差
- 衡量长期预测准确性

### 目标性能
- **车辆**: ADE < 1.5m, FDE < 3.0m
- **行人**: ADE < 0.8m, FDE < 1.5m

---

## 🛠️ 高级用法

### 1. 自定义数据收集

```python
from data_collection.carla_data_collector import TrajectoryDataCollector

collector = TrajectoryDataCollector()
collector.setup_world(map_name='Town05', weather='WetNoon')

# 自定义spawn数量
vehicles = collector.spawn_vehicles(num_vehicles=100)
walkers, controllers = collector.spawn_pedestrians(num_pedestrians=50)

collector.collect_data(duration=600)
```

### 2. 实时预测示例

```python
import torch
from models.transformer_model import TrajectoryTransformer

# 加载模型
model = TrajectoryTransformer(...)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测
with torch.no_grad():
    future_traj = model.predict(observation)
```

### 3. 批量处理多个地图

```bash
#!/bin/bash
for map in Town01 Town02 Town03 Town04 Town05; do
    echo "Processing $map..."
    python main.py collect --map-name $map --duration 300
    python main.py preprocess --data-dir ./data_$map
done

# 合并所有数据后训练
python main.py train --augment --num-epochs 200
```

### 4. 超参数搜索

```python
# 在train.py中实现
configs = [
    {'d_model': 64, 'nhead': 4, 'lr': 1e-4},
    {'d_model': 128, 'nhead': 8, 'lr': 1e-4},
    {'d_model': 256, 'nhead': 8, 'lr': 5e-5},
]

for config in configs:
    model = TrajectoryTransformer(**config)
    trainer = TrajectoryPredictor(model)
    history = trainer.train(...)
```

---

## 💡 最佳实践

### 数据收集
1. **多样性**: 收集不同地图、天气、时间的数据
2. **充足性**: 至少5-10分钟,理想情况30分钟+
3. **平衡性**: 确保车辆和行人数据都充足

### 模型训练
1. **从小开始**: 先用小数据集和小模型验证流程
2. **监控过拟合**: 关注train/val loss gap
3. **调整学习率**: 如果loss震荡,降低lr
4. **使用增强**: 小数据集时必须启用

### 性能优化
1. **增加数据**: 最有效的提升方法
2. **增大模型**: d_model=256, layers=6
3. **调整长度**: 增加obs_len或pred_len
4. **多模态**: 对于复杂场景使用多模态

---

## 🔧 故障排除

### 常见问题

**Q: CARLA连接超时**
```bash
# 检查CARLA是否运行
ps aux | grep Carla

# 使用不同端口
python main.py collect --port 2001
```

**Q: 内存不足**
```bash
# 减小batch size
python main.py train --batch-size 32

# 减小模型
python main.py train --d-model 64
```

**Q: 预测精度低**
- 收集更多数据
- 增大模型容量
- 检查数据质量
- 使用数据增强
- 尝试多模态模型

**Q: Loss为NaN**
- 降低学习率到1e-5
- 检查数据是否有异常值
- 增加梯度裁剪

---

## 📚 扩展方向

### 研究方向
1. **社会交互建模**: 显式建模agent间交互
2. **地图融合**: 加入HD地图信息
3. **不确定性估计**: 预测置信度
4. **长期预测**: 扩展到更长时间范围
5. **端到端规划**: 结合规划模块

### 工程改进
1. **在线学习**: 从实时数据持续学习
2. **模型压缩**: 量化和剪枝用于实时部署
3. **多任务学习**: 同时预测多个目标
4. **对抗训练**: 提升鲁棒性

---

## 🎓 参考资源

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Trajectron++](https://arxiv.org/abs/2001.03093)
- [Multipath](https://arxiv.org/abs/1910.05449)
- [VectorNet](https://arxiv.org/abs/2005.04259)

### 工具
- [CARLA Simulator](https://carla.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### 相关项目
- [Waymo Open Dataset](https://waymo.com/open/)
- [nuScenes](https://www.nuscenes.org/)
- [Argoverse](https://www.argoverse.org/)

---

## 📄 许可和引用

### 许可
MIT License - 可自由使用和修改

### 引用
如果这个项目对你的研究有帮助,欢迎引用。

---

## 🤝 贡献指南

欢迎通过以下方式贡献:
1. 报告bugs和问题
2. 提出新功能建议
3. 提交代码改进
4. 改进文档

---

## 📮 联系方式

如有问题或建议,请通过以下方式联系:
- GitHub Issues
- Email: [你的邮箱]

---

**祝你在轨迹预测研究中取得成功!** 🚗🤖🚶

---

最后更新: 2026年2月12日
版本: 1.0.0
