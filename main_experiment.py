#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our modules
from breast_cancer_model import (
    BreastCancerDataset, 
    MultiModalBreastCancerModel, 
    train_model
)
from visualization_utils import (
    plot_training_history,
    evaluate_model,
    plot_attention_weights
)


def setup_argparse():
    """Setup argument parser for command line options"""
    parser = argparse.ArgumentParser(description='Train breast cancer ODX score prediction model')
    
    # Data params
    parser.add_argument('--dataset', type=str, default='TCGA_BRCA',
                       help='Dataset name (e.g., TCGA_BRCA, OSU)')
    parser.add_argument('--exp_name', type=str, default='default',
                       help='Experiment name for saving results')
    parser.add_argument('--splits_dir', type=str, default='splits',
                       help='Directory containing data splits')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    # Model params
    parser.add_argument('--backbone', type=str, default='ctp',
                       help='Feature backbone (ctp, resnet50, quilt1m, clip)')
    parser.add_argument('--slide_feature_dim', type=int, default=1024,
                       help='Dimension of slide features from backbone')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate for regularization')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--reg_weight', type=float, default=0.7,
                       help='Weight for regression loss')
    parser.add_argument('--cls_weight', type=float, default=0.3,
                       help='Weight for classification loss')
    
    # Split configuration
    parser.add_argument('--use_fixed_splits', action='store_true',
                       help='Use fixed train/val splits instead of k-fold cross-validation')
    parser.add_argument('--split_num', type=int, default=0,
                       help='Specific split number to use (e.g., 0 for train0.csv/val0.csv)')
    parser.add_argument('--n_splits', type=int, default=3,
                       help='Number of pre-defined splits (0 to n_splits-1)')
    
    # Cross-validation params (used if not using fixed splits)
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def run_experiment(args):
    """Run model training with fixed splits or cross-validation"""
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, 
        f"{args.dataset}_{args.exp_name}_{args.backbone}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args for reproducibility
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Tracking metrics across splits/folds
    all_metrics = {
        'mae': [],
        'mse': [],
        'r2': [],
        'hl_accuracy': []
    }
    
    # Device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Fixed test set path for this dataset
    test_csv_path = os.path.join(args.splits_dir, args.dataset, 'test.csv')
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test set not found at {test_csv_path}")
    
    print(f"Fixed test set will be used from: {test_csv_path}")
    
    # Determine if using fixed splits or cross-validation
    if args.use_fixed_splits:
        # Using pre-defined splits (train0.csv, val0.csv, etc.)
        splits_to_run = [args.split_num] if args.split_num >= 0 else range(args.n_splits)
        
        for split_idx in splits_to_run:
            print(f"\n{'='*50}")
            print(f"Split {split_idx}")
            print(f"{'='*50}")
            
            # Create split directory
            split_dir = os.path.join(output_dir, f"split_{split_idx}")
            os.makedirs(split_dir, exist_ok=True)
            
            # Load train and validation splits
            train_csv_path = os.path.join(args.splits_dir, args.dataset, f"train{split_idx}.csv")
            val_csv_path = os.path.join(args.splits_dir, args.dataset, f"val{split_idx}.csv")
            
            if not os.path.exists(train_csv_path):
                raise FileNotFoundError(f"Train split not found at {train_csv_path}")
            if not os.path.exists(val_csv_path):
                raise FileNotFoundError(f"Validation split not found at {val_csv_path}")
            
            # Run training for this split
            metrics = train_and_evaluate_split(
                train_csv_path=train_csv_path,
                val_csv_path=val_csv_path,
                test_csv_path=test_csv_path,
                output_dir=split_dir,
                args=args,
                device=device
            )
            
            # Store metrics
            for metric, value in metrics.items():
                all_metrics[metric].append(value)
    else:
        # Using cross-validation on a combined dataset
        print("Running cross-validation...")
        
        # Load all training data (combining all train splits)
        all_train_dfs = []
        for split_idx in range(args.n_splits):
            train_csv_path = os.path.join(args.splits_dir, args.dataset, f"train{split_idx}.csv")
            if os.path.exists(train_csv_path):
                df = pd.read_csv(train_csv_path)
                all_train_dfs.append(df)
        
        if not all_train_dfs:
            raise ValueError(f"No training splits found in {os.path.join(args.splits_dir, args.dataset)}")
        
        # Combine all training data
        combined_df = pd.concat(all_train_dfs, ignore_index=True)
        print(f"Combined dataset has {len(combined_df)} samples")
        
        # Define cross-validation
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        
        # Run cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_df)):
            print(f"\n{'='*50}")
            print(f"Fold {fold+1}/{args.n_folds}")
            print(f"{'='*50}")
            
            # Create fold directory
            fold_dir = os.path.join(output_dir, f"fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Split data
            train_df = combined_df.iloc[train_idx].reset_index(drop=True)
            val_df = combined_df.iloc[val_idx].reset_index(drop=True)
            
            # Save splits for reference
            train_csv_path = os.path.join(fold_dir, 'train.csv')
            val_csv_path = os.path.join(fold_dir, 'val.csv')
            
            train_df.to_csv(train_csv_path, index=False)
            val_df.to_csv(val_csv_path, index=False)
            
            # Run training for this fold
            metrics = train_and_evaluate_split(
                train_csv_path=train_csv_path,
                val_csv_path=val_csv_path,
                test_csv_path=test_csv_path,
                output_dir=fold_dir,
                args=args,
                device=device
            )
            
            # Store metrics
            for metric, value in metrics.items():
                all_metrics[metric].append(value)
    
    # Calculate and summarize overall performance
    print("\n" + "="*50)
    print("Overall Performance Summary")
    print("="*50)
    
    summary = {}
    for metric, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric] = {'mean': mean_val, 'std': std_val}
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save summary to file
    with open(os.path.join(output_dir, 'performance_summary.txt'), 'w') as f:
        for metric, stats in summary.items():
            f.write(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
    
    # Plot overall performance
    plt.figure(figsize=(12, 8))
    metrics_to_plot = list(all_metrics.keys())
    n_runs = len(all_metrics[metrics_to_plot[0]])
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        plt.bar(range(1, n_runs+1), all_metrics[metric])
        plt.xlabel('Run' if args.use_fixed_splits else 'Fold')
        plt.ylabel(metric)
        plt.title(f'{metric} across {"splits" if args.use_fixed_splits else "folds"}')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=summary[metric]['mean'], color='r', linestyle='-', label='Mean')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_performance.png'), dpi=300)
    
    print(f"\nExperiment completed. Results saved to {output_dir}")


def train_and_evaluate_split(train_csv_path, val_csv_path, test_csv_path, output_dir, args, device):
    """Train and evaluate model on a specific data split"""
    print(f"Training with data: {train_csv_path}")
    print(f"Validation with data: {val_csv_path}")
    print(f"Testing with data: {test_csv_path}")
    
    # Load datasets
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")
    
    # Import necessary modules
    import sys
    import os
    
    # Add the current directory to sys.path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Import explicitly from the module file
    from breast_cancer_model import BreastCancerDataset, MultiModalBreastCancerModel, custom_collate_fn
    
    # Create datasets with the specified backbone
    train_dataset = BreastCancerDataset(csv_path=train_csv_path, backbone=args.backbone)
    val_dataset = BreastCancerDataset(csv_path=val_csv_path, backbone=args.backbone)
    test_dataset = BreastCancerDataset(csv_path=test_csv_path, backbone=args.backbone)
    
    # Set batch size - use 1 to avoid dimension issues, can try larger batches once it works
    batch_size = 1
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues during debugging
        collate_fn=custom_collate_fn,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=custom_collate_fn,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=custom_collate_fn,
        pin_memory=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1
    )
    
    # Determine feature dimension based on backbone
    if args.backbone == 'ctp':  # CTransPath
        feature_dim = 1024
    elif args.backbone == 'resnet50':
        feature_dim = 2048
    elif args.backbone == 'quilt1m':
        feature_dim = 1024
    elif args.backbone == 'clip':
        feature_dim = 512
    else:
        feature_dim = args.slide_feature_dim
    
    # Initialize model
    model = MultiModalBreastCancerModel(
        slide_feature_dim=feature_dim,
        clinical_feature_dim=4,  # Length + 3 one-hot grade features
        dropout_rate=args.dropout_rate
    )
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        regression_weight=args.reg_weight,
        classification_weight=args.cls_weight
    )
    
    # Save model
    torch.save(trained_model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    plot_training_history(history)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(trained_model, test_loader, device)
    
    # Save predictions for further analysis
    test_predictions = pd.DataFrame({
        'true_odx': results['odx_true'],
        'pred_odx': results['odx_pred'],
        'true_hl': results['hl_true'],
        'pred_hl': results['hl_pred'],
        'grade': results['grades']
    })
    test_predictions.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    # Visualize attention weights
    print("Visualizing attention weights...")
    attention_dir = os.path.join(output_dir, 'attention')
    os.makedirs(attention_dir, exist_ok=True)
    plot_attention_weights(
        model=trained_model,
        dataloader=test_loader,
        device=device,
        num_samples=5
    )
    
    return results['metrics']


if __name__ == "__main__":
    args = setup_argparse()
    run_experiment(args)