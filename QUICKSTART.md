# 快速开始指南

## 🎯 完整工作流程

### 第一步: 环境准备

```bash
# 1. 克隆/下载项目
cd carla_trajectory_prediction

# 2. 安装依赖
pip install -r requirements.txt

# 3. 确保CARLA正在运行
# 下载CARLA: https://github.com/carla-simulator/carla/releases
# 启动CARLA服务器:
./CarlaUE4.sh -carla-server -benchmark -fps=20
```

### 第二步: 收集数据

```bash
# 使用默认配置收集数据 (Town01, 5分钟)
python main.py collect

# 或自定义参数
python main.py collect \
    --map-name Town03 \
    --weather MidRainyNoon \
    --duration 600 \
    --output-dir ./my_data
```

**输出文件:**
- `carla_trajectory_data/trajectories_TIMESTAMP.json` - 轨迹数据
- `carla_trajectory_data/frames_TIMESTAMP.json` - 帧数据
- `carla_trajectory_data/statistics_TIMESTAMP.json` - 统计信息

### 第三步: 预处理数据

```bash
# 自动查找最新的数据文件并预处理
python main.py preprocess

# 或指定文件
python main.py preprocess \
    --trajectory-file ./carla_trajectory_data/trajectories_20240101_120000.json \
    --frame-file ./carla_trajectory_data/frames_20240101_120000.json \
    --output-file ./processed_data/train_data.pkl
```

**输出文件:**
- `processed_data/train_data.pkl` - 处理后的训练数据

### 第四步: 训练模型

```bash
# 基础训练 (使用默认Transformer模型)
python main.py train --data-file ./processed_data/train_data.pkl

# 使用数据增强
python main.py train --data-file ./processed_data/train_data.pkl --augment

# 训练多模态模型
python main.py train \
    --data-file ./processed_data/train_data.pkl \
    --model-type multimodal \
    --num-modes 6 \
    --augment

# 自定义超参数
python main.py train \
    --data-file ./processed_data/train_data.pkl \
    --batch-size 128 \
    --learning-rate 5e-5 \
    --d-model 256 \
    --num-encoder-layers 6 \
    --num-decoder-layers 6 \
    --num-epochs 150
```

**训练监控:**
训练过程会实时显示:
- Train Loss, ADE, FDE
- Validation Loss, ADE, FDE
- 学习率变化
- 最佳模型自动保存

**输出文件:**
- `checkpoints/best_model.pth` - 最佳模型
- `checkpoints/checkpoint_epoch_XX.pth` - 定期检查点
- `checkpoints/training_history.json` - 训练历史
- `checkpoints/training_curves.png` - 训练曲线图

### 第五步: 评估和可视化

```bash
# 评估最佳模型
python main.py evaluate \
    --model-path ./checkpoints/best_model.pth \
    --data-path ./processed_data/train_data.pkl

# 使用CPU (如果没有GPU)
python main.py evaluate \
    --model-path ./checkpoints/best_model.pth \
    --data-path ./processed_data/train_data.pkl \
    --device cpu
```

**输出文件:**
- `visualization_single.png` - 单个预测详细图
- `visualization_batch.png` - 批量预测对比图

---

## 📊 命令参考

### collect命令参数

```bash
python main.py collect [OPTIONS]

--host              CARLA服务器地址 (默认: localhost)
--port              CARLA端口 (默认: 2000)
--map-name          地图名称 (默认: Town01)
                    可选: Town01-Town10
--weather           天气预设 (默认: ClearNoon)
                    可选: ClearNoon, CloudyNoon, WetNoon, MidRainyNoon
--duration          收集时长/秒 (默认: 300)
--save-interval     保存间隔/帧 (默认: 100)
--output-dir        输出目录 (默认: ./carla_trajectory_data)
```

### preprocess命令参数

```bash
python main.py preprocess [OPTIONS]

--data-dir          数据目录 (默认: ./carla_trajectory_data)
--trajectory-file   轨迹文件路径 (默认: None, 自动查找最新)
--frame-file        帧文件路径 (默认: None, 自动查找最新)
--output-file       输出文件路径 (默认: ./processed_data/train_data.pkl)
--obs-len           观测长度/帧 (默认: 8)
--pred-len          预测长度/帧 (默认: 12)
--min-trajectory-len 最小轨迹长度 (默认: 20)
```

### train命令参数

```bash
python main.py train [OPTIONS]

# 数据和训练
--data-file         处理后的数据文件 (默认: ./processed_data/train_data.pkl)
--model-type        模型类型 (默认: transformer)
                    可选: transformer, multimodal
--batch-size        批次大小 (默认: 64)
--num-epochs        训练轮数 (默认: 100)
--learning-rate     学习率 (默认: 1e-4)
--weight-decay      权重衰减 (默认: 1e-5)
--train-split       训练集比例 (默认: 0.8)
--num-workers       数据加载器工作进程 (默认: 4)
--checkpoint-dir    检查点目录 (默认: ./checkpoints)
--patience          早停耐心值 (默认: 15)
--augment           启用数据增强 (标志)

# 模型架构
--input-dim         输入维度 (默认: 7)
--d-model           模型维度 (默认: 128)
--nhead             注意力头数 (默认: 8)
--num-encoder-layers Encoder层数 (默认: 4)
--num-decoder-layers Decoder层数 (默认: 4)
--dim-feedforward   前馈网络维度 (默认: 512)
--dropout           Dropout率 (默认: 0.1)
--obs-len           观测长度 (默认: 8)
--pred-len          预测长度 (默认: 12)
--num-modes         模态数量 (默认: 6, 仅多模态)
```

### evaluate命令参数

```bash
python main.py evaluate [OPTIONS]

--model-path        模型检查点路径 (默认: ./checkpoints/best_model.pth)
--data-path         数据文件路径 (默认: ./processed_data/train_data.pkl)
--device            设备 (默认: cuda)
                    可选: cuda, cpu
```

---

## 🎓 示例场景

### 场景1: 小规模快速测试

```bash
# 1. 收集少量数据 (1分钟)
python main.py collect --duration 60

# 2. 预处理
python main.py preprocess

# 3. 快速训练 (小模型, 少轮数)
python main.py train \
    --num-epochs 20 \
    --d-model 64 \
    --batch-size 32

# 4. 评估
python main.py evaluate
```

### 场景2: 完整训练流程

```bash
# 1. 收集多样化数据
for map in Town01 Town02 Town03; do
    python main.py collect \
        --map-name $map \
        --duration 300 \
        --output-dir ./data_${map}
done

# 2. 分别预处理(或合并)
python main.py preprocess --data-dir ./data_Town01
python main.py preprocess --data-dir ./data_Town02
python main.py preprocess --data-dir ./data_Town03

# 3. 大模型训练
python main.py train \
    --d-model 256 \
    --num-encoder-layers 6 \
    --num-decoder-layers 6 \
    --batch-size 128 \
    --num-epochs 150 \
    --augment

# 4. 评估
python main.py evaluate
```

### 场景3: 多模态预测

```bash
# 1. 收集数据
python main.py collect --duration 600

# 2. 预处理
python main.py preprocess

# 3. 训练多模态模型
python main.py train \
    --model-type multimodal \
    --num-modes 6 \
    --d-model 128 \
    --batch-size 64 \
    --num-epochs 100 \
    --augment

# 4. 评估
python main.py evaluate
```

---

## 💡 提示和技巧

### 数据收集技巧

1. **多样化场景**: 在不同地图和天气条件下收集数据
2. **足够时长**: 至少收集5-10分钟数据(6000-12000帧)
3. **检查数据**: 查看statistics文件确认收集了足够的agents

### 训练技巧

1. **从小开始**: 先用小模型和少量数据测试流程
2. **监控指标**: 关注ADE和FDE,而不仅仅是loss
3. **调整学习率**: 如果loss不下降,尝试降低学习率
4. **数据增强**: 对于小数据集,启用--augment很重要
5. **早停**: 如果验证loss不再下降,早停会自动触发

### GPU内存管理

如果遇到GPU内存不足:
```bash
# 减小batch size
--batch-size 32

# 减小模型尺寸
--d-model 64 --dim-feedforward 256

# 减少worker数量
--num-workers 2
```

### 提升性能

如果预测精度不够:
```bash
# 1. 收集更多数据
--duration 1800  # 30分钟

# 2. 增大模型
--d-model 256 --num-encoder-layers 6 --num-decoder-layers 6

# 3. 更多训练轮数
--num-epochs 200

# 4. 使用多模态
--model-type multimodal --num-modes 6
```

---

## 📈 预期结果

### 良好的训练指标 (Town01, 5分钟数据)

- **ADE**: < 1.5 米 (车辆), < 0.8 米 (行人)
- **FDE**: < 3.0 米 (车辆), < 1.5 米 (行人)
- **训练时间**: ~1-2小时 (单GPU, 100 epochs)

### 可视化示例

训练成功后,你应该看到:
- 训练曲线平滑下降
- 验证loss稳定在低水平
- 预测轨迹与真实轨迹基本重合

---

## ❓ 疑难解答

**问题**: CARLA连接失败
```bash
# 解决方案
# 1. 确保CARLA正在运行
ps aux | grep Carla

# 2. 检查端口
netstat -an | grep 2000

# 3. 尝试不同端口
python main.py collect --port 2001
```

**问题**: 预处理时没有找到文件
```bash
# 解决方案
# 手动指定文件路径
python main.py preprocess \
    --trajectory-file path/to/trajectories_xxx.json \
    --frame-file path/to/frames_xxx.json
```

**问题**: 训练过程中loss为NaN
```bash
# 解决方案
# 1. 降低学习率
--learning-rate 1e-5

# 2. 增加梯度裁剪(已默认启用)
# 3. 检查数据是否正常
```

---

**祝你训练成功! 如有问题,请查看README.md获取更多信息。** 🚀
