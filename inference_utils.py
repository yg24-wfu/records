import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Import model architecture
from breast_cancer_model import MultiModalBreastCancerModel

class InferenceDataset(Dataset):
    """Dataset for inference with slide features and clinical data"""
    
    def __init__(self, slide_paths, clinical_features, scaler=None):
        """
        Args:
            slide_paths: List of paths to slide feature tensors
            clinical_features: DataFrame with clinical features (grade, length)
            scaler: Optional pre-trained scaler for length normalization
        """
        self.slide_paths = slide_paths
        self.clinical_features = clinical_features
        
        # Convert grade to one-hot encoding
        self.grades_onehot = pd.get_dummies(self.clinical_features['grade'], prefix='grade')
        
        # Normalize length feature
        if scaler is None:
            self.length_scaler = StandardScaler()
            self.clinical_features['length_scaled'] = self.length_scaler.fit_transform(
                self.clinical_features['length'].values.reshape(-1, 1)
            )
        else:
            self.length_scaler = scaler
            self.clinical_features['length_scaled'] = self.length_scaler.transform(
                self.clinical_features['length'].values.reshape(-1, 1)
            )
    
    def __len__(self):
        return len(self.slide_paths)
    
    def __getitem__(self, idx):
        # Load pre-extracted slide features
        slide_features = torch.load(self.slide_paths[idx])
        
        # Get clinical features
        row = self.clinical_features.iloc[idx]
        grade = int(row['grade'])
        
        # Create one-hot encoding for grade
        grade_onehot = [0, 0, 0]
        grade_onehot[grade-1] = 1
        
        # Clinical features: grade (one-hot) and normalized length
        clinical_features = torch.tensor([
            row['length_scaled'],
            grade_onehot[0],
            grade_onehot[1],
            grade_onehot[2]
        ], dtype=torch.float32)
        
        return {
            'slide_features': slide_features,
            'clinical_features': clinical_features,
            'slide_path': self.slide_paths[idx],
            'grade': grade
        }


class ODXPredictor:
    """Class for making ODX score predictions using a trained model"""
    
    def __init__(self, model_path, slide_feature_dim=1024, device=None):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model weights
            slide_feature_dim: Dimension of slide features
            device: Device to run inference on (cpu or cuda)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load model
        self.model = MultiModalBreastCancerModel(
            slide_feature_dim=slide_feature_dim,
            clinical_feature_dim=4
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path} to {self.device}")
    
    def predict_single(self, slide_path, grade, length):
        """
        Make prediction for a single sample
        
        Args:
            slide_path: Path to the slide feature tensor
            grade: Tumor grade (1, 2, or 3)
            length: Length value
        
        Returns:
            Dictionary with prediction results
        """
        # Load slide features
        slide_features = torch.load(slide_path)
        slide_features = slide_features.to(self.device)
        
        # Prepare clinical features
        # Note: This is simplified and does not use the scaler from training
        # In a production environment, you should use the same scaler
        grade_onehot = [0, 0, 0]
        grade_onehot[grade-1] = 1
        
        clinical_features = torch.tensor([
            length, grade_onehot[0], grade_onehot[1], grade_onehot[2]
        ], dtype=torch.float32).to(self.device)
        
        # Add batch dimension if needed
        if len(slide_features.shape) == 2:
            slide_features = slide_features.unsqueeze(0)
        clinical_features = clinical_features.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(slide_features, clinical_features)
        
        # Get predictions
        odx_score = outputs['odx_score_pred'].cpu().numpy()[0]
        odx_hl_probs = torch.softmax(outputs['odx_hl_logits'], dim=1).cpu().numpy()[0]
        odx_hl = np.argmax(odx_hl_probs)
        
        return {
            'odx_score': float(odx_score),
            'odx_hl': int(odx_hl),
            'odx_hl_prob': float(odx_hl_probs[odx_hl]),
            'low_risk_prob': float(odx_hl_probs[0]),
            'high_risk_prob': float(odx_hl_probs[1])
        }
    
    def predict_batch(self, slide_paths, clinical_data, batch_size=16):
        """
        Make predictions for a batch of samples
        
        Args:
            slide_paths: List of paths to slide feature tensors
            clinical_data: DataFrame with clinical features (must have 'grade' and 'length' columns)
            batch_size: Batch size for inference
        
        Returns:
            DataFrame with prediction results
        """
        # Create dataset and dataloader
        dataset = InferenceDataset(slide_paths, clinical_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # Make predictions
        all_results = []
        
        with torch.no_grad():
            for batch in dataloader:
                slide_features = batch['slide_features'].to(self.device)
                clinical_features = batch['clinical_features'].to(self.device)
                
                outputs = self.model(slide_features, clinical_features)
                
                # Get predictions
                odx_scores = outputs['odx_score_pred'].cpu().numpy()
                odx_hl_probs = torch.softmax(outputs['odx_hl_logits'], dim=1).cpu().numpy()
                odx_hl = np.argmax(odx_hl_probs, axis=1)
                
                # Store results
                for i in range(len(slide_features)):
                    result = {
                        'slide_path': batch['slide_path'][i],
                        'grade': int(batch['grade'][i]),
                        'odx_score': float(odx_scores[i]),
                        'odx_hl': int(odx_hl[i]),
                        'low_risk_prob': float(odx_hl_probs[i, 0]),
                        'high_risk_prob': float(odx_hl_probs[i, 1])
                    }
                    all_results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        return results_df


def load_ensemble_models(model_dir, slide_feature_dim=1024, device=None):
    """
    Load ensemble of models from a directory
    
    Args:
        model_dir: Directory containing model files
        slide_feature_dim: Dimension of slide features
        device: Device to run inference on
    
    Returns:
        List of loaded models
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    # Load models
    models = []
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model = MultiModalBreastCancerModel(
            slide_feature_dim=slide_feature_dim,
            clinical_feature_dim=4
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    
    print(f"Loaded {len(models)} models from {model_dir}")
    return models


def ensemble_predict(models, slide_features, clinical_features, device):
    """
    Make prediction using an ensemble of models
    
    Args:
        models: List of models
        slide_features: Slide features tensor
        clinical_features: Clinical features tensor
        device: Device to run inference on
    
    Returns:
        Dictionary with ensemble prediction results
    """
    # Ensure tensors are on the correct device
    slide_features = slide_features.to(device)
    clinical_features = clinical_features.to(device)
    
    # Add batch dimension if needed
    if len(slide_features.shape) == 2:
        slide_features = slide_features.unsqueeze(0)
    if len(clinical_features.shape) == 1:
        clinical_features = clinical_features.unsqueeze(0)
    
    # Make predictions with each model
    all_odx_scores = []
    all_odx_hl_probs = []
    
    with torch.no_grad():
        for model in models:
            outputs = model(slide_features, clinical_features)
            
            odx_scores = outputs['odx_score_pred'].cpu().numpy()
            odx_hl_probs = torch.softmax(outputs['odx_hl_logits'], dim=1).cpu().numpy()
            
            all_odx_scores.append(odx_scores)
            all_odx_hl_probs.append(odx_hl_probs)
    
    # Average predictions
    mean_odx_scores = np.mean(all_odx_scores, axis=0)
    mean_odx_hl_probs = np.mean(all_odx_hl_probs, axis=0)
    pred_odx_hl = np.argmax(mean_odx_hl_probs, axis=1)
    
    # Calculate uncertainty (standard deviation)
    std_odx_scores = np.std(all_odx_scores, axis=0)
    
    # Prepare results
    results = []
    for i in range(len(mean_odx_scores)):
        result = {
            'odx_score': float(mean_odx_scores[i]),
            'odx_score_uncertainty': float(std_odx_scores[i]),
            'odx_hl': int(pred_odx_hl[i]),
            'low_risk_prob': float(mean_odx_hl_probs[i, 0]),
            'high_risk_prob': float(mean_odx_hl_probs[i, 1])
        }
        results.append(result)
    
    return results


def save_predictions_to_json(predictions, output_file):
    """
    Save prediction results to a JSON file
    
    Args:
        predictions: List of prediction results
        output_file: Path to save the JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)


def main():
    """Example usage of the predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make ODX score predictions')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--data_csv', type=str, required=True,
                        help='Path to CSV file with slide paths and clinical data')
    parser.add_argument('--output_file', type=str, default='predictions.csv',
                        help='Path to save prediction results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    args = parser.parse_args()
    
    # Load data
    data = pd.read_csv(args.data_csv)
    slide_paths = data['path'].tolist()
    
    # Check required columns
    required_cols = ['grade', 'length']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in the data")
    
    # Create predictor
    predictor = ODXPredictor(args.model_path)
    
    # Make predictions
    results = predictor.predict_batch(slide_paths, data[required_cols], args.batch_size)
    
    # Save results
    results.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()