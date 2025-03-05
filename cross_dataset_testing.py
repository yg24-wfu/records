#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Dataset Testing Script

This script allows testing a model trained on one dataset (e.g., TCGA_BRCA)
on another dataset (e.g., OSU) to evaluate generalization performance.
"""

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import our modules
from breast_cancer_model import MultiModalBreastCancerModel, BreastCancerDataset
from visualization_utils import evaluate_model


def setup_argparse():
    """Setup argument parser for command line options"""
    parser = argparse.ArgumentParser(description='Cross-dataset testing for breast cancer model')
    
    # Model params
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--model_backbone', type=str, default='ctp',
                       help='Backbone used for the trained model')
    parser.add_argument('--slide_feature_dim', type=int, default=1024,
                       help='Dimension of slide features from backbone')
    
    # Source dataset (training dataset)
    parser.add_argument('--source_dataset', type=str, default='TCGA_BRCA',
                       help='Source dataset name (e.g., TCGA_BRCA)')
    
    # Target dataset (testing dataset)
    parser.add_argument('--target_dataset', type=str, required=True,
                       help='Target dataset name for testing (e.g., OSU)')
    parser.add_argument('--target_split', type=str, default='test',
                       help='Split to use from target dataset (e.g., test, val0)')
    
    # Data params
    parser.add_argument('--splits_dir', type=str, default='splits',
                       help='Directory containing data splits')
    parser.add_argument('--output_dir', type=str, default='cross_dataset_results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    
    return parser.parse_args()


def cross_dataset_test(args):
    """Test a trained model on a different dataset"""
    # Device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(
        args.output_dir, 
        f"{args.source_dataset}_to_{args.target_dataset}_{args.model_backbone}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args for reproducibility
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Determine target CSV path
    if args.target_split == 'test':
        target_csv_path = os.path.join(args.splits_dir, args.target_dataset, 'test.csv')
    else:
        target_csv_path = os.path.join(args.splits_dir, args.target_dataset, f"{args.target_split}.csv")
    
    if not os.path.exists(target_csv_path):
        raise FileNotFoundError(f"Target dataset split not found at {target_csv_path}")
    
    print(f"Testing on data: {target_csv_path}")
    
    # Load the dataset
    test_dataset = BreastCancerDataset(
        csv_path=target_csv_path, 
        backbone=args.model_backbone
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Load the model
    model = MultiModalBreastCancerModel(
        slide_feature_dim=args.slide_feature_dim,
        clinical_feature_dim=4
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate on target dataset
    print("\nEvaluating on target dataset...")
    results = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nPerformance on target dataset:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Save predictions for further analysis
    test_predictions = pd.DataFrame({
        'true_odx': results['odx_true'],
        'pred_odx': results['odx_pred'],
        'true_hl': results['hl_true'],
        'pred_hl': results['hl_pred'],
        'grade': results['grades']
    })
    test_predictions.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    # Compare predictions by grade
    plt.figure(figsize=(12, 8))
    
    for grade in sorted(set(results['grades'])):
        grade_mask = results['grades'] == grade
        plt.scatter(
            results['odx_true'][grade_mask], 
            results['odx_pred'][grade_mask],
            label=f'Grade {grade}',
            alpha=0.7
        )
    
    # Add perfect prediction line
    min_val = min(min(results['odx_true']), min(results['odx_pred']))
    max_val = max(max(results['odx_true']), max(results['odx_pred']))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('True ODX Score')
    plt.ylabel('Predicted ODX Score')
    plt.title(f'Cross-Dataset Performance: {args.source_dataset} → {args.target_dataset}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'cross_dataset_performance.png'), dpi=300)
    
    print(f"\nCross-dataset testing completed. Results saved to {output_dir}")


if __name__ == "__main__":
    args = setup_argparse()
    cross_dataset_test(args)