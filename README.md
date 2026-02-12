# CARLA轨迹预测项目

基于Transformer的车辆和行人轨迹预测系统,使用CARLA模拟器收集训练数据。

## 📋 项目结构

```
carla_trajectory_prediction/
├── data_collection/          # 数据收集模块
│   ├── carla_data_collector.py    # CARLA数据收集器
│   └── data_preprocessor.py       # 数据预处理
├── models/                   # 模型定义
│   └── transformer_model.py       # Transformer模型
├── training/                 # 训练脚本
│   └── train.py                   # 训练主程序
├── utils/                    # 工具函数
│   └── visualization.py           # 可视化工具
├── configs/                  # 配置文件
│   └── config.yaml
├── requirements.txt          # 依赖列表
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保CARLA服务器已启动
# 下载CARLA: https://github.com/carla-simulator/carla/releases
# 运行: ./CarlaUE4.sh (Linux) 或 CarlaUE4.exe (Windows)
```

### 2. 数据收集

```bash
# 从CARLA收集轨迹数据
cd data_collection
python carla_data_collector.py

# 这将在CARLA中生成车辆和行人,并收集5分钟的轨迹数据
# 数据保存在 ./carla_trajectory_data/
```

**收集的数据包括:**
- 车辆和行人的位置 (x, y, z)
- 速度向量
- 朝向角度
- Agent类型

### 3. 数据预处理

```bash
# 预处理原始数据
python data_preprocessor.py

# 这将:
# - 提取轨迹序列 (观测8帧 + 预测12帧)
# - 归一化坐标(相对位置编码)
# - 创建交互图(邻居关系)
# - 保存为 .pkl 文件
```

**预处理输出:**
- `processed_data/train_data.pkl` - 训练数据

### 4. 训练模型

```bash
# 开始训练
cd ../training
python train.py

# 训练配置可在脚本中修改:
# - batch_size: 64
# - learning_rate: 1e-4
# - num_epochs: 100
# - 自动早停(patience=15)
```

**训练输出:**
- `checkpoints/best_model.pth` - 最佳模型
- `checkpoints/training_history.json` - 训练历史
- `checkpoints/training_curves.png` - 训练曲线

### 5. 可视化结果

```bash
# 可视化预测结果
cd ../utils
python visualization.py

# 这将生成:
# - visualization_single.png - 单个预测详图
# - visualization_batch.png - 批量预测对比
```

## 📊 模型架构

### Transformer轨迹预测模型

```
输入: 观测轨迹 (8帧历史数据)
  ↓
[Input Embedding] → [Positional Encoding]
  ↓
[Transformer Encoder] (4层)
  - Multi-head Self-Attention (8 heads)
  - Feed-Forward Network
  ↓
[Transformer Decoder] (4层)
  - Masked Self-Attention
  - Cross-Attention with encoder
  - Feed-Forward Network
  ↓
[Output Projection]
  ↓
输出: 预测轨迹 (12帧未来数据)
```

**模型特点:**
- ✅ 使用相对位置编码(ego-centric坐标系)
- ✅ 支持车辆和行人两种agent类型
- ✅ Teacher forcing训练策略
- ✅ 可选的多模态预测(6个可能的未来轨迹)

## 📈 评估指标

训练过程监控以下指标:

1. **ADE (Average Displacement Error)**
   - 所有预测时间步的平均位移误差
   - 单位: 米

2. **FDE (Final Displacement Error)**
   - 最后一个预测时间步的位移误差
   - 单位: 米

3. **Loss (MSE)**
   - 均方误差损失

## 🎯 使用建议

### 数据收集优化

```python
# 在 carla_data_collector.py 中调整:

# 1. 增加多样性 - 使用不同地图
for map_name in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']:
    collector.setup_world(map_name=map_name)
    collector.collect_data(duration=180)

# 2. 不同天气条件
weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'MidRainyNoon']
for weather in weathers:
    collector.setup_world(weather=weather)
    collector.collect_data(duration=120)

# 3. 增加agent数量以获得更多交互
collector.spawn_vehicles(num_vehicles=80)
collector.spawn_pedestrians(num_pedestrians=50)
```

### 模型调优

```python
# 在 train.py 中调整超参数:

config = {
    # 增加模型容量
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    
    # 调整学习率
    'learning_rate': 5e-5,
    
    # 增加batch size(如果GPU内存足够)
    'batch_size': 128,
    
    # 数据增强
    'augment': True  # 在TrajectoryDataset中启用
}
```

### 多模态预测

```python
# 使用MultiModalTrajectoryTransformer
from models.transformer_model import MultiModalTrajectoryTransformer

model = MultiModalTrajectoryTransformer(
    input_dim=7,
    d_model=128,
    num_modes=6,  # 预测6个可能的未来轨迹
    obs_len=8,
    pred_len=12
)

# 推理
trajectories, mode_probs = model(obs_features)
# trajectories: (batch, 6, 12, 2)
# mode_probs: (batch, 6) - 每个模态的概率
```

## 🔧 高级功能

### 1. 实时预测

```python
# 创建实时预测器
from models.transformer_model import TrajectoryTransformer
import torch

class RealtimePredictor:
    def __init__(self, model_path):
        self.model = TrajectoryTransformer(...)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict(self, obs_trajectory):
        """
        实时预测
        obs_trajectory: 最近8帧的观测数据
        """
        with torch.no_grad():
            pred = self.model.predict(obs_trajectory)
        return pred
```

### 2. 集成到CARLA

```python
# 在CARLA中使用训练好的模型
import carla

predictor = RealtimePredictor('checkpoints/best_model.pth')

while True:
    # 获取车辆状态
    vehicle = world.get_actors().filter('vehicle.*')[0]
    
    # 收集历史轨迹(8帧)
    obs_data = collect_history(vehicle, frames=8)
    
    # 预测未来轨迹
    future_traj = predictor.predict(obs_data)
    
    # 可视化或用于决策
    visualize_prediction(future_traj)
```

## 📝 配置说明

编辑 `configs/config.yaml` 来自定义:

```yaml
# 观测和预测长度
preprocessing:
  obs_len: 8   # 历史观测帧数
  pred_len: 12 # 未来预测帧数

# 模型尺寸
model:
  d_model: 128        # 隐藏层维度
  nhead: 8            # 注意力头数
  num_encoder_layers: 4
  num_decoder_layers: 4

# 训练参数
training:
  batch_size: 64
  learning_rate: 0.0001
  num_epochs: 100
```

## 🐛 常见问题

**Q1: CARLA连接失败**
```bash
# 确保CARLA服务器正在运行
./CarlaUE4.sh -carla-server -benchmark -fps=20

# 检查端口是否被占用
netstat -an | grep 2000
```

**Q2: GPU内存不足**
```python
# 减小batch size
config['batch_size'] = 32

# 或减小模型尺寸
config['d_model'] = 64
config['dim_feedforward'] = 256
```

**Q3: 预测精度不够**
- 收集更多数据(增加duration)
- 增加模型容量(d_model, num_layers)
- 调整学习率
- 使用多模态预测

## 📚 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原理
- [Trajectron++](https://arxiv.org/abs/2001.03093) - 多模态轨迹预测
- [CARLA Simulator](https://carla.org/) - 自动驾驶模拟器

## 🤝 贡献

欢迎提交Issue和Pull Request!

## 📄 许可证

MIT License

---

**祝你训练顺利! 🚗🚶**
