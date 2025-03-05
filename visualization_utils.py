import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader

def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training metrics
    """
    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training and validation loss
    axes[0].plot(history['train_loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot validation MAE
    axes[1].plot(history['val_mae'], color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Validation MAE')
    axes[1].grid(True)
    
    # Plot validation R²
    axes[2].plot(history['val_r2'], color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('Validation R²')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()


def plot_prediction_scatter(y_true, y_pred, odx_hl=None):
    """
    Plot scatter plot of true vs predicted ODX scores.
    
    Args:
        y_true: Array of true ODX scores
        y_pred: Array of predicted ODX scores
        odx_hl: Array of binary ODX high/low values for color coding
    """
    plt.figure(figsize=(10, 8))
    
    if odx_hl is not None:
        # Color points by high/low classification
        colors = ['blue' if hl == 0 else 'red' for hl in odx_hl]
        plt.scatter(y_true, y_pred, c=colors, alpha=0.6)
        
        # Add legend
        plt.scatter([], [], c='blue', label='Low Risk')
        plt.scatter([], [], c='red', label='High Risk')
        plt.legend()
    else:
        plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('True ODX Score')
    plt.ylabel('Predicted ODX Score')
    plt.title('True vs Predicted ODX Scores')
    plt.grid(True)
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    plt.annotate(f'Correlation: {correlation:.3f}', xy=(0.05, 0.95), 
                 xycoords='axes fraction', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('prediction_scatter.png', dpi=300)
    plt.show()


def plot_error_by_grade(y_true, y_pred, grades):
    """
    Plot error distribution by grade.
    
    Args:
        y_true: Array of true ODX scores
        y_pred: Array of predicted ODX scores
        grades: Array of grade values
    """
    # Calculate absolute errors
    errors = np.abs(y_pred - y_true)
    
    # Create a DataFrame for easy plotting
    error_df = pd.DataFrame({
        'Grade': grades,
        'Absolute Error': errors
    })
    
    plt.figure(figsize=(10, 6))
    
    # Box plot of errors by grade
    sns.boxplot(x='Grade', y='Absolute Error', data=error_df)
    sns.stripplot(x='Grade', y='Absolute Error', data=error_df, 
                  color='black', alpha=0.3, jitter=True)
    
    plt.title('Distribution of Prediction Errors by Grade')
    plt.xlabel('Tumor Grade')
    plt.ylabel('Absolute Error')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('error_by_grade.png', dpi=300)
    plt.show()


def plot_attention_weights(model, dataloader, device, num_samples=5):
    """
    Visualize attention weights for a few samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing samples
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    model.eval()
    samples_plotted = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_plotted >= num_samples:
                break
                
            slide_features = batch['slide_features'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            
            # Get model outputs including attention weights
            outputs = model(slide_features, clinical_features)
            attention_weights = outputs['attention_weights']
            
            if attention_weights is None:
                print("No attention weights available for visualization.")
                break
                
            # Plot attention weights for batch samples
            for i in range(min(len(slide_features), num_samples - samples_plotted)):
                sample_attention = attention_weights[i].cpu().numpy()
                
                if len(sample_attention.shape) > 1:  # For 2D attention maps
                    plt.figure(figsize=(10, 4))
                    plt.imshow(sample_attention, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Attention Weight')
                    plt.title(f'Attention Weights for Sample {samples_plotted + 1}')
                    plt.xlabel('Feature Dimension')
                    plt.ylabel('Sequence Position')
                    plt.tight_layout()
                    plt.savefig(f'attention_sample_{samples_plotted + 1}.png', dpi=300)
                    plt.show()
                else:  # For 1D attention weights
                    plt.figure(figsize=(12, 4))
                    plt.bar(range(len(sample_attention)), sample_attention)
                    plt.title(f'Attention Weights for Sample {samples_plotted + 1}')
                    plt.xlabel('Patch Index')
                    plt.ylabel('Attention Weight')
                    plt.tight_layout()
                    plt.savefig(f'attention_sample_{samples_plotted + 1}.png', dpi=300)
                    plt.show()
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    break


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data and return metrics and predictions.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
    
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    model.eval()
    
    all_odx_preds = []
    all_odx_true = []
    all_hl_preds = []
    all_hl_true = []
    all_grades = []
    
    with torch.no_grad():
        for batch in test_loader:
            clinical_features = batch['clinical_features'].to(device)
            odx_score = batch['odx_score'].cpu().numpy()
            odx_hl = batch['odx_hl'].cpu().numpy()
            
            # Get grade from one-hot encoding
            grades = np.argmax(clinical_features[:, 1:4].cpu().numpy(), axis=1) + 1
            
            # Check if batch requires individual processing
            requires_individual = batch.get('batch_requires_individual_processing', False)
            
            if requires_individual:
                # Process each sample individually
                slide_features_list = batch['slide_features']  # List of tensors
                
                for i in range(len(slide_features_list)):
                    individual_slide_features = slide_features_list[i].unsqueeze(0).to(device)
                    individual_clinical = clinical_features[i].unsqueeze(0)
                    
                    # Forward pass
                    outputs = model(individual_slide_features, individual_clinical)
                    
                    # Regression predictions
                    odx_pred = outputs['odx_score_pred'].cpu().numpy()
                    
                    # Classification predictions
                    hl_logits = outputs['odx_hl_logits'].cpu().numpy()
                    hl_pred = np.argmax(hl_logits, axis=1)
                    
                    # Store results
                    all_odx_preds.extend(odx_pred)
                    all_odx_true.append(odx_score[i])
                    all_hl_preds.extend(hl_pred)
                    all_hl_true.append(odx_hl[i])
                    all_grades.append(grades[i])
            else:
                # Standard batch processing
                slide_features = batch['slide_features'].to(device)
                
                # Forward pass
                outputs = model(slide_features, clinical_features)
                
                # Regression predictions
                odx_pred = outputs['odx_score_pred'].cpu().numpy()
                
                # Classification predictions
                hl_logits = outputs['odx_hl_logits'].cpu().numpy()
                hl_pred = np.argmax(hl_logits, axis=1)
                
                # Store results
                all_odx_preds.extend(odx_pred)
                all_odx_true.extend(odx_score)
                all_hl_preds.extend(hl_pred)
                all_hl_true.extend(odx_hl)
                all_grades.extend(grades)
    
    # Convert to numpy arrays
    all_odx_preds = np.array(all_odx_preds)
    all_odx_true = np.array(all_odx_true)
    all_hl_preds = np.array(all_hl_preds)
    all_hl_true = np.array(all_hl_true)
    all_grades = np.array(all_grades)
    
    # Calculate regression metrics
    mae = np.mean(np.abs(all_odx_preds - all_odx_true))
    mse = np.mean((all_odx_preds - all_odx_true) ** 2)
    r2 = 1 - (np.sum((all_odx_true - all_odx_preds) ** 2) / 
              np.sum((all_odx_true - np.mean(all_odx_true)) ** 2))
    
    # Calculate classification metrics
    hl_accuracy = np.mean(all_hl_preds == all_hl_true)
    cm = confusion_matrix(all_hl_true, all_hl_preds)
    
    # Print results
    print("Regression Metrics:")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print("\nClassification Metrics:")
    print(f"  Accuracy: {hl_accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_hl_true, all_hl_preds, 
                               target_names=['Low Risk', 'High Risk']))
    
    # Create visualizations
    plot_prediction_scatter(all_odx_true, all_odx_preds, all_hl_true)
    plot_error_by_grade(all_odx_true, all_odx_preds, all_grades)
    
    return {
        'odx_true': all_odx_true,
        'odx_pred': all_odx_preds,
        'hl_true': all_hl_true,
        'hl_pred': all_hl_preds,
        'grades': all_grades,
        'metrics': {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'hl_accuracy': hl_accuracy
        }
    }