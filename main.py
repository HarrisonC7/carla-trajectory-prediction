#!/usr/bin/env python3
"""
CARLA轨迹预测项目 - 主启动脚本
提供命令行接口用于数据收集、训练和评估
"""
import argparse
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def collect_data(args):
    """数据收集"""
    from data_collection.carla_data_collector import TrajectoryDataCollector
    
    print("="*60)
    print("Starting CARLA Data Collection")
    print("="*60)
    
    collector = TrajectoryDataCollector(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir
    )
    
    collector.setup_world(
        map_name=args.map_name,
        weather=args.weather
    )
    
    collector.collect_data(
        duration=args.duration,
        save_interval=args.save_interval
    )
    
    print("\n✓ Data collection completed!")


def preprocess_data(args):
    """数据预处理"""
    from data_collection.data_preprocessor import TrajectoryPreprocessor
    
    print("="*60)
    print("Starting Data Preprocessing")
    print("="*60)
    
    preprocessor = TrajectoryPreprocessor(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        min_trajectory_len=args.min_trajectory_len
    )
    
    # 查找最新的轨迹文件
    if args.trajectory_file is None:
        data_dir = Path(args.data_dir)
        trajectory_files = sorted(data_dir.glob('trajectories_*.json'))
        frame_files = sorted(data_dir.glob('frames_*.json'))
        
        if not trajectory_files or not frame_files:
            print("Error: No trajectory or frame data files found!")
            return
        
        args.trajectory_file = str(trajectory_files[-1])
        args.frame_file = str(frame_files[-1])
        print(f"Using trajectory file: {args.trajectory_file}")
        print(f"Using frame file: {args.frame_file}")
    
    preprocessor.process_carla_data(
        trajectory_file=args.trajectory_file,
        frame_data_file=args.frame_file,
        output_file=args.output_file
    )
    
    print("\n✓ Data preprocessing completed!")


def train_model(args):
    """训练模型"""
    import torch
    from torch.utils.data import DataLoader, random_split
    from training.train import TrajectoryPredictor
    from models.transformer_model import TrajectoryTransformer, MultiModalTrajectoryTransformer
    from data_collection.data_preprocessor import TrajectoryDataset
    
    print("="*60)
    print("Starting Model Training")
    print("="*60)
    
    # 加载数据
    print("Loading dataset...")
    full_dataset = TrajectoryDataset(args.data_file, augment=args.augment)
    
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    print("\nInitializing model...")
    if args.model_type == 'transformer':
        model = TrajectoryTransformer(
            input_dim=args.input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            obs_len=args.obs_len,
            pred_len=args.pred_len
        )
    else:  # multimodal
        model = MultiModalTrajectoryTransformer(
            input_dim=args.input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            num_modes=args.num_modes
        )
    
    # 训练
    trainer = TrajectoryPredictor(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.patience
    )
    
    print("\n✓ Training completed!")


def evaluate_model(args):
    """评估模型"""
    from utils.visualization import evaluate_and_visualize
    
    print("="*60)
    print("Starting Model Evaluation")
    print("="*60)
    
    evaluate_and_visualize(
        model_path=args.model_path,
        data_path=args.data_path,
        device=args.device
    )
    
    print("\n✓ Evaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='CARLA Trajectory Prediction')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # ===== 数据收集命令 =====
    collect_parser = subparsers.add_parser('collect', help='Collect data from CARLA')
    collect_parser.add_argument('--host', default='localhost', help='CARLA server host')
    collect_parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    collect_parser.add_argument('--map-name', default='Town01', help='CARLA map name')
    collect_parser.add_argument('--weather', default='ClearNoon', help='Weather preset')
    collect_parser.add_argument('--duration', type=int, default=300, help='Collection duration (seconds)')
    collect_parser.add_argument('--save-interval', type=int, default=100, help='Save interval (frames)')
    collect_parser.add_argument('--output-dir', default='./carla_trajectory_data', help='Output directory')
    
    # ===== 数据预处理命令 =====
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess collected data')
    preprocess_parser.add_argument('--data-dir', default='./carla_trajectory_data', help='Data directory')
    preprocess_parser.add_argument('--trajectory-file', default=None, help='Trajectory file path')
    preprocess_parser.add_argument('--frame-file', default=None, help='Frame file path')
    preprocess_parser.add_argument('--output-file', default='./processed_data/train_data.pkl', 
                                   help='Output file path')
    preprocess_parser.add_argument('--obs-len', type=int, default=8, help='Observation length')
    preprocess_parser.add_argument('--pred-len', type=int, default=12, help='Prediction length')
    preprocess_parser.add_argument('--min-trajectory-len', type=int, default=20, 
                                   help='Minimum trajectory length')
    
    # ===== 训练命令 =====
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-file', default='./processed_data/train_data.pkl', 
                             help='Processed data file')
    train_parser.add_argument('--model-type', default='transformer', choices=['transformer', 'multimodal'],
                             help='Model type')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    train_parser.add_argument('--train-split', type=float, default=0.8, help='Train split ratio')
    train_parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    train_parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    train_parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    
    # Model architecture
    train_parser.add_argument('--input-dim', type=int, default=7, help='Input dimension')
    train_parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    train_parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    train_parser.add_argument('--num-encoder-layers', type=int, default=4, help='Number of encoder layers')
    train_parser.add_argument('--num-decoder-layers', type=int, default=4, help='Number of decoder layers')
    train_parser.add_argument('--dim-feedforward', type=int, default=512, help='FFN dimension')
    train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    train_parser.add_argument('--obs-len', type=int, default=8, help='Observation length')
    train_parser.add_argument('--pred-len', type=int, default=12, help='Prediction length')
    train_parser.add_argument('--num-modes', type=int, default=6, help='Number of modes (multimodal only)')
    
    # ===== 评估命令 =====
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model-path', default='./checkpoints/best_model.pth', 
                            help='Model checkpoint path')
    eval_parser.add_argument('--data-path', default='./processed_data/train_data.pkl', 
                            help='Data file path')
    eval_parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.command == 'collect':
        collect_data(args)
    elif args.command == 'preprocess':
        preprocess_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
