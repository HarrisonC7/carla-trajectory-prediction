"""
模型测试脚本 - 不需要CARLA,使用合成数据测试模型
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.transformer_model import TrajectoryTransformer, MultiModalTrajectoryTransformer


def generate_synthetic_trajectory(num_samples=100, obs_len=8, pred_len=12):
    """
    生成合成轨迹数据用于测试
    
    包括:
    - 直线行驶
    - 左转/右转
    - 加速/减速
    - 停止
    """
    trajectories = []
    
    for _ in range(num_samples):
        # 随机选择轨迹类型
        traj_type = np.random.choice(['straight', 'left_turn', 'right_turn', 'stop'])
        
        total_len = obs_len + pred_len
        t = np.linspace(0, 1, total_len)
        
        if traj_type == 'straight':
            # 直线行驶
            x = t * 20  # 20米
            y = np.zeros_like(t)
            speed = np.ones(total_len) * 10  # 10 m/s
            
        elif traj_type == 'left_turn':
            # 左转
            angle = t * np.pi / 2  # 90度转弯
            radius = 10
            x = radius * np.sin(angle)
            y = radius * (1 - np.cos(angle))
            speed = np.ones(total_len) * 8
            
        elif traj_type == 'right_turn':
            # 右转
            angle = t * np.pi / 2
            radius = 10
            x = radius * np.sin(angle)
            y = -radius * (1 - np.cos(angle))
            speed = np.ones(total_len) * 8
            
        else:  # stop
            # 逐渐停止
            x = t * 10 * (1 - t)
            y = np.zeros_like(t)
            speed = np.maximum(0, 10 * (1 - t * 2))
        
        # 添加噪声
        x += np.random.randn(total_len) * 0.5
        y += np.random.randn(total_len) * 0.5
        
        # 计算速度向量
        vx = np.gradient(x) / 0.05  # 假设dt=0.05s
        vy = np.gradient(y) / 0.05
        
        # 创建特征
        positions = np.stack([x, y], axis=-1)
        velocities = np.stack([vx, vy], axis=-1)
        
        # Agent类型 (随机选择车辆或行人)
        agent_type = np.random.choice([0, 1])  # 0: vehicle, 1: pedestrian
        agent_onehot = np.zeros(2)
        agent_onehot[agent_type] = 1.0
        
        # 组合特征 [pos(2), vel(2), speed(1), type(2)]
        features = np.concatenate([
            positions[:obs_len],
            velocities[:obs_len],
            speed[:obs_len, np.newaxis],
            np.tile(agent_onehot, (obs_len, 1))
        ], axis=-1)
        
        trajectory = {
            'obs_features': features,
            'obs_pos': positions[:obs_len],
            'pred_pos': positions[obs_len:],
            'type': traj_type,
            'agent_type': agent_type
        }
        
        trajectories.append(trajectory)
    
    return trajectories


def test_transformer_model():
    """测试Transformer模型"""
    print("="*60)
    print("Testing TrajectoryTransformer")
    print("="*60)
    
    # 创建模型
    model = TrajectoryTransformer(
        input_dim=7,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        obs_len=8,
        pred_len=12
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成测试数据
    print("\nGenerating synthetic data...")
    trajectories = generate_synthetic_trajectory(num_samples=32)
    
    # 准备batch
    obs_features = torch.FloatTensor(np.stack([t['obs_features'] for t in trajectories]))
    target_pos = torch.FloatTensor(np.stack([t['pred_pos'] for t in trajectories]))
    
    print(f"Batch shape: obs_features={obs_features.shape}, target={target_pos.shape}")
    
    # 前向传播测试
    print("\n--- Forward Pass Test ---")
    model.eval()
    with torch.no_grad():
        pred_pos = model.predict(obs_features)
    
    print(f"Prediction shape: {pred_pos.shape}")
    print(f"✓ Forward pass successful!")
    
    # 简单训练测试
    print("\n--- Training Test (10 iterations) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(10):
        optimizer.zero_grad()
        pred = model(obs_features, target_pos)
        loss = criterion(pred, target_pos)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print("✓ Training test successful!")
    
    # 可视化一些预测
    print("\n--- Visualization ---")
    model.eval()
    with torch.no_grad():
        pred_pos = model.predict(obs_features)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(6):
        ax = axes[i]
        
        obs_pos = trajectories[i]['obs_pos']
        true_pos = trajectories[i]['pred_pos']
        pred = pred_pos[i].numpy()
        traj_type = trajectories[i]['type']
        
        # 绘制轨迹
        ax.plot(obs_pos[:, 0], obs_pos[:, 1], 'o-', 
               color='blue', linewidth=2, markersize=6, label='Observation')
        ax.plot(true_pos[:, 0], true_pos[:, 1], '^-',
               color='green', linewidth=2, markersize=6, label='Ground Truth')
        ax.plot(pred[:, 0], pred[:, 1], 's-',
               color='red', linewidth=2, markersize=6, label='Prediction')
        
        # 计算误差
        ade = np.mean(np.sqrt(np.sum((pred - true_pos) ** 2, axis=1)))
        fde = np.sqrt(np.sum((pred[-1] - true_pos[-1]) ** 2))
        
        ax.set_title(f'{traj_type.replace("_", " ").title()}\n'
                    f'ADE: {ade:.2f}m, FDE: {fde:.2f}m', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('./test_transformer_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: ./test_transformer_predictions.png")
    
    return model


def test_multimodal_model():
    """测试多模态模型"""
    print("\n" + "="*60)
    print("Testing MultiModalTrajectoryTransformer")
    print("="*60)
    
    # 创建模型
    model = MultiModalTrajectoryTransformer(
        input_dim=7,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        obs_len=8,
        pred_len=12,
        num_modes=6
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成测试数据
    print("\nGenerating synthetic data...")
    trajectories = generate_synthetic_trajectory(num_samples=16)
    
    obs_features = torch.FloatTensor(np.stack([t['obs_features'] for t in trajectories]))
    target_pos = torch.FloatTensor(np.stack([t['pred_pos'] for t in trajectories]))
    
    # 前向传播测试
    print("\n--- Forward Pass Test ---")
    model.eval()
    with torch.no_grad():
        pred_trajectories, mode_probs = model(obs_features)
    
    print(f"Prediction shape: {pred_trajectories.shape}")
    print(f"Mode probabilities shape: {mode_probs.shape}")
    print(f"Mode prob sum (first sample): {mode_probs[0].sum():.4f}")
    print(f"✓ Forward pass successful!")
    
    # 可视化多模态预测
    print("\n--- Visualization ---")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, 6))
    
    for i in range(4):
        ax = axes[i]
        
        obs_pos = trajectories[i]['obs_pos']
        true_pos = trajectories[i]['pred_pos']
        pred_modes = pred_trajectories[i].numpy()
        probs = mode_probs[i].numpy()
        traj_type = trajectories[i]['type']
        
        # 观测和真实
        ax.plot(obs_pos[:, 0], obs_pos[:, 1], 'o-',
               color='blue', linewidth=3, markersize=8, label='Observation', zorder=10)
        ax.plot(true_pos[:, 0], true_pos[:, 1], '^-',
               color='green', linewidth=3, markersize=8, label='Ground Truth', zorder=9)
        
        # 所有模态
        for mode_idx, (traj, prob) in enumerate(zip(pred_modes, probs)):
            alpha = 0.3 + 0.7 * prob
            ax.plot(traj[:, 0], traj[:, 1], 's-',
                   color=colors[mode_idx], linewidth=2, markersize=4,
                   alpha=alpha, label=f'Mode {mode_idx+1} ({prob:.2f})', zorder=5-mode_idx)
        
        # 计算最佳模态误差
        best_idx = np.argmax(probs)
        best_pred = pred_modes[best_idx]
        ade = np.mean(np.sqrt(np.sum((best_pred - true_pos) ** 2, axis=1)))
        fde = np.sqrt(np.sum((best_pred[-1] - true_pos[-1]) ** 2))
        
        ax.set_title(f'{traj_type.replace("_", " ").title()}\n'
                    f'Best Mode ADE: {ade:.2f}m, FDE: {fde:.2f}m', fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('./test_multimodal_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: ./test_multimodal_predictions.png")
    
    return model


def main():
    """主测试函数"""
    print("\n" + "🚀 "*20)
    print("CARLA Trajectory Prediction - Model Testing")
    print("🚀 "*20 + "\n")
    
    print("This script tests the models without requiring CARLA.")
    print("It generates synthetic trajectory data and tests:")
    print("  1. TrajectoryTransformer (single-modal)")
    print("  2. MultiModalTrajectoryTransformer (multi-modal)")
    print()
    
    # 测试单模态模型
    transformer_model = test_transformer_model()
    
    # 测试多模态模型
    multimodal_model = test_multimodal_model()
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - test_transformer_predictions.png")
    print("  - test_multimodal_predictions.png")
    print("\nYou can now proceed to collect real CARLA data and train!")
    print()


if __name__ == "__main__":
    main()
