def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized slide features
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Collated batch with correctly handled tensors
    """
    # Extract slide features, clinical features, and targets
    slide_features = [item['slide_features'] for item in batch]
    clinical_features = torch.stack([item['clinical_features'] for item in batch])
    odx_scores = torch.stack([item['odx_score'] for item in batch])
    odx_hl = torch.stack([item['odx_hl'] for item in batch])
    slide_paths = [item['slide_path'] for item in batch]
    
    # Check if slide_features are all tensors with the same shape
    if len(slide_features) > 1:
        shapes = [feat.shape for feat in slide_features]
        # Different shapes means we can't stack
        if any(shape != shapes[0] for shape in shapes):
            return {
                'slide_features': slide_features,  # Keep as list of individual tensors
                'clinical_features': clinical_features,
                'odx_score': odx_scores,
                'odx_hl': odx_hl,
                'slide_path': slide_paths,
                'batch_requires_individual_processing': True
            }
    
    # If we get here, either we have only one sample or all samples have the same shape
    try:
        # Try to stack slide features
        slide_features = torch.stack(slide_features)
        return {
            'slide_features': slide_features,
            'clinical_features': clinical_features,
            'odx_score': odx_scores,
            'odx_hl': odx_hl,
            'slide_path': slide_paths,
            'batch_requires_individual_processing': False
        }
    except RuntimeError:
        # If stacking fails for any reason, use individual processing
        print("Warning: Could not stack slide features, switching to individual processing")
        return {
            'slide_features': slide_features,
            'clinical_features': clinical_features,
            'odx_score': odx_scores,
            'odx_hl': odx_hl,
            'slide_path': slide_paths,
            'batch_requires_individual_processing': True
        }
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


class BreastCancerDataset(Dataset):
    """Dataset for loading breast cancer slide features and clinical data"""
    
    def __init__(self, csv_path, transform=None, backbone='ctp'):
        """
        Args:
            csv_path: Path to the CSV file with slide paths and clinical data
            transform: Optional transform to be applied on features
            backbone: Feature extraction backbone ('ctp', 'resnet50', 'quilt1m', 'clip')
        """
        self.data_df = pd.read_csv(csv_path)
        self.transform = transform
        self.backbone = backbone
        
        # Handle different path formats based on backbone
        if 'path' in self.data_df.columns:
            # Path column exists, keep as is
            pass
        elif f'{backbone}_path' in self.data_df.columns:
            # Specific backbone path column exists
            self.data_df['path'] = self.data_df[f'{backbone}_path']
        else:
            # Try to construct path from slide_id if available
            if 'slide_id' in self.data_df.columns:
                # Construct path based on backbone and slide_id
                # This assumes a standard directory structure
                base_dir = os.path.dirname(os.path.dirname(csv_path))
                features_dir = os.path.join(base_dir, 'features', backbone)
                self.data_df['path'] = self.data_df['slide_id'].apply(
                    lambda x: os.path.join(features_dir, f"{x}.pt")
                )
            else:
                raise ValueError(f"No path column found for backbone {backbone} in {csv_path}")
        
        # Verify that all paths exist
        missing_paths = [path for path in self.data_df['path'] if not os.path.exists(path)]
        if missing_paths:
            print(f"WARNING: {len(missing_paths)} feature files not found. First few: {missing_paths[:5]}")
        
        # Get feature dimensions from first valid file
        self.feature_dim = None
        for path in self.data_df['path']:
            if os.path.exists(path):
                try:
                    features = torch.load(path, map_location='cpu')
                    if isinstance(features, dict) and 'features' in features:
                        features = features['features']
                    
                    # Check feature shape and format
                    if len(features.shape) == 1:  # 1D vector
                        self.feature_dim = features.shape[0]
                        self.feature_type = "vector"
                    elif len(features.shape) == 2:  # 2D tensor (patches x features)
                        self.feature_dim = features.shape[1]
                        self.feature_type = "patches"
                    else:
                        self.feature_dim = features.shape[-1]
                        self.feature_type = "multi_dim"
                    
                    print(f"Feature type: {self.feature_type}, dimension: {self.feature_dim}")
                    break
                except Exception as e:
                    continue
        
        if self.feature_dim is None:
            # Fallback dimensions based on backbone
            if backbone == 'resnet50':
                self.feature_dim = 2048
            elif backbone == 'clip':
                self.feature_dim = 512
            else:
                self.feature_dim = 1024
            self.feature_type = "unknown"
            print(f"WARNING: Could not determine feature dimensions. Using fallback: {self.feature_dim}")
            
        # Convert grade to one-hot encoding
        self.grades_onehot = pd.get_dummies(self.data_df['grade'], prefix='grade')
        self.data_df = pd.concat([self.data_df, self.grades_onehot], axis=1)
        
        # Normalize length feature
        self.length_scaler = StandardScaler()
        self.data_df['length_scaled'] = self.length_scaler.fit_transform(
            self.data_df['length'].values.reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Load pre-extracted slide features (saved as PyTorch tensors)
        try:
            slide_path = row['path']
            slide_features = torch.load(slide_path, map_location='cpu')
            
            # Handle different tensor formats
            if isinstance(slide_features, dict) and 'features' in slide_features:
                # Some extraction methods store features in a dict
                slide_features = slide_features['features']
            
            # Ensure slide_features is a tensor
            if not isinstance(slide_features, torch.Tensor):
                slide_features = torch.tensor(slide_features, dtype=torch.float32)
            
            # Handle 3D+ tensors (squeezing extra dimensions)
            if len(slide_features.shape) > 2:
                print(f"Warning: Found {len(slide_features.shape)}D tensor, reshaping to 2D")
                # If more than 2D, reshape to 2D (patches x features)
                original_shape = slide_features.shape
                slide_features = slide_features.reshape(-1, original_shape[-1])
            
            # For 1D tensors, expand to 2D (1 x features)
            if len(slide_features.shape) == 1:
                slide_features = slide_features.unsqueeze(0)
                
        except Exception as e:
            print(f"Error loading features from {row['path']}: {str(e)}")
            # Provide dummy features as fallback with 1 patch and feature_dim features
            slide_features = torch.zeros((1, self.feature_dim), dtype=torch.float32)
        
        # Clinical features: grade (one-hot) and normalized length
        clinical_features = torch.tensor([
            row['length_scaled'],
            row['grade_1'] if 'grade_1' in row else 0,
            row['grade_2'] if 'grade_2' in row else 0,
            row['grade_3'] if 'grade_3' in row else 0
        ], dtype=torch.float32)
        
        # Target: ODX score
        odx_score = torch.tensor(row['odx_score'], dtype=torch.float32)
        
        # Secondary target: ODX high/low
        odx_hl = torch.tensor(row['odx_HL'], dtype=torch.long)
        
        if self.transform:
            slide_features = self.transform(slide_features)
            
        return {
            'slide_features': slide_features,
            'clinical_features': clinical_features,
            'odx_score': odx_score,
            'odx_hl': odx_hl,
            'slide_path': row['path']
        }


class AttentionPool(nn.Module):
    """Attention pooling module for slide features that adapts to input dimensions"""
    
    def __init__(self, expected_dim):
        super(AttentionPool, self).__init__()
        self.expected_dim = expected_dim
        # We'll initialize the attention layers dynamically during the forward pass
        # to adapt to the actual input dimension
        self.attention_layers = None
        
    def _initialize_layers(self, actual_dim):
        """Dynamically initialize attention layers based on actual input dimension"""
        self.attention_layers = nn.Sequential(
            nn.Linear(actual_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(next(self.parameters()).device)
        print(f"Initialized attention layers for feature dim: {actual_dim}")
        
    def forward(self, x):
        """
        Adaptive attention pooling that handles various input formats
        
        Args:
            x: Input features, can be:
               - [batch_size, num_patches, feature_dim] for patch-based features
               - [batch_size, feature_dim] for global features
               
        Returns:
            Pooled features and attention weights
        """
        # Handle global features case
        if len(x.shape) == 2:
            return x, None
        
        # For patch-based features
        batch_size, num_patches, feature_dim = x.shape
        
        # Initialize attention layers based on actual feature dimension
        if self.attention_layers is None or self.attention_layers[0].in_features != feature_dim:
            self._initialize_layers(feature_dim)
        
        # Apply attention mechanism
        try:
            a = self.attention_layers(x)  # [batch_size, num_patches, 1]
            a = torch.softmax(a, dim=1)   # Attention weights
            v = torch.sum(a * x, dim=1)   # [batch_size, feature_dim]
            return v, a
        except RuntimeError as e:
            print(f"Error in attention pooling: {e}")
            print(f"Input shape: {x.shape}, Expected feature dim: {self.expected_dim}")
            # Fallback: use mean pooling if attention fails
            v = torch.mean(x, dim=1)
            return v, None


class CrossAttention(nn.Module):
    """Cross-attention module for feature fusion with dynamic feature dimensions"""
    
    def __init__(self, img_dim, clinical_dim):
        super(CrossAttention, self).__init__()
        self.img_dim = img_dim
        self.clinical_dim = clinical_dim
        
        # These will be initialized dynamically during the first forward pass
        self.query_proj = None
        self.key_proj = None
        self.value_proj = None
        
    def _initialize_projections(self, actual_img_dim):
        """Dynamically initialize projection layers based on actual image feature dimension"""
        self.query_proj = nn.Linear(self.clinical_dim, actual_img_dim).to(self.device)
        self.key_proj = nn.Linear(actual_img_dim, actual_img_dim).to(self.device)
        self.value_proj = nn.Linear(actual_img_dim, actual_img_dim).to(self.device)
        print(f"Initialized cross-attention projections for feature dim: {actual_img_dim}")
    
    @property
    def device(self):
        # Get the device of the module parameters
        params = list(self.parameters())
        if params:
            return params[0].device
        return torch.device("cpu")
        
    def forward(self, img_features, clinical_features):
        # img_features: [batch_size, actual_img_dim]
        # clinical_features: [batch_size, clinical_dim]
        
        actual_img_dim = img_features.shape[1]
        
        # Initialize projection layers if needed
        if self.query_proj is None or self.query_proj.out_features != actual_img_dim:
            self._initialize_projections(actual_img_dim)
        
        try:
            # Project query from clinical features
            q = self.query_proj(clinical_features)  # [batch_size, actual_img_dim]
            
            # Project key and value from image features
            k = self.key_proj(img_features)  # [batch_size, actual_img_dim]
            v = self.value_proj(img_features)  # [batch_size, actual_img_dim]
            
            # Reshape for attention computation
            q = q.unsqueeze(1)  # [batch_size, 1, actual_img_dim]
            k = k.unsqueeze(2)  # [batch_size, actual_img_dim, 1]
            
            # Compute attention scores
            scale_factor = torch.sqrt(torch.tensor(actual_img_dim, dtype=torch.float32, device=self.device))
            attention = torch.bmm(q, k) / scale_factor  # [batch_size, 1, 1]
            attention = torch.softmax(attention, dim=2)
            
            # Apply attention to values
            context = attention * v  # [batch_size, actual_img_dim]
            
            return context
        except RuntimeError as e:
            print(f"Error in cross-attention: {e}")
            print(f"Image features shape: {img_features.shape}, Clinical features shape: {clinical_features.shape}")
            # Fallback: return original image features if cross-attention fails
            return img_features


class MultiModalBreastCancerModel(nn.Module):
    """Multi-modal model for ODX score prediction that adapts to input dimensions"""
    
    def __init__(self, slide_feature_dim=1024, clinical_feature_dim=4, dropout_rate=0.3):
        super(MultiModalBreastCancerModel, self).__init__()
        
        # Define expected dimensions
        self.slide_feature_dim = slide_feature_dim
        self.clinical_feature_dim = clinical_feature_dim
        self.dropout_rate = dropout_rate
        
        # Attention pooling for slide features
        self.attention_pool = AttentionPool(slide_feature_dim)
        
        # Clinical features processing
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # Cross-attention fusion
        self.cross_attention = CrossAttention(slide_feature_dim, 32)
        
        # Dynamic fusion layer will be initialized in forward pass
        self.fusion = None
        
        # Regression and classification heads will be initialized in forward pass
        self.regression_head = None
        self.classification_head = None
        
    def _initialize_fusion_layers(self, actual_dim):
        """Initialize fusion layers based on actual feature dimension"""
        device = next(self.parameters()).device
        
        self.fusion = nn.Sequential(
            nn.Linear(actual_dim + 32, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.dropout_rate)
        ).to(device)
        
        self.regression_head = nn.Linear(128, 1).to(device)
        self.classification_head = nn.Linear(128, 2).to(device)
        
        print(f"Initialized fusion layers for feature dim: {actual_dim}")
    
    def forward(self, slide_features, clinical_features):
        try:
            # Apply attention pooling to handle different feature formats
            pooled_slide_features, attention_weights = self.attention_pool(slide_features)
            
            # Get actual feature dimension
            actual_dim = pooled_slide_features.shape[1]
            
            # Process clinical features
            processed_clinical = self.clinical_branch(clinical_features)
            
            # Cross-attention between modalities
            attended_features = self.cross_attention(pooled_slide_features, processed_clinical)
            
            # Initialize fusion layers if needed
            if self.fusion is None or self.fusion[0].in_features != (actual_dim + 32):
                self._initialize_fusion_layers(actual_dim)
            
            # Concatenate attended features with clinical
            combined = torch.cat([attended_features, processed_clinical], dim=1)
            
            # Common representation
            common = self.fusion(combined)
            
            # Regression output
            odx_score_pred = self.regression_head(common).squeeze(1)
            
            # Classification output
            odx_hl_logits = self.classification_head(common)
            
            return {
                'odx_score_pred': odx_score_pred,
                'odx_hl_logits': odx_hl_logits,
                'attention_weights': attention_weights
            }
            
        except RuntimeError as e:
            print(f"Error in model forward pass: {e}")
            print(f"Slide features shape: {slide_features.shape}, Clinical features shape: {clinical_features.shape}")
            # Re-raise the error
            raise


class HuberLoss(nn.Module):
    """Huber loss for robust regression"""
    
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss.mean()


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, 
               regression_weight=0.7, classification_weight=0.3):
    """
    Train the multi-modal model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        regression_weight: Weight for regression loss
        classification_weight: Weight for classification loss
    
    Returns:
        Trained model and training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device} for training")
    
    # Define losses
    regression_criterion = HuberLoss(delta=1.0)
    classification_criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_r2': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Check if batch requires individual processing
            requires_individual = batch.get('batch_requires_individual_processing', False)
            
            if requires_individual:
                # Process each sample individually
                slide_features_list = batch['slide_features']  # List of tensors
                clinical_features = batch['clinical_features'].to(device)
                odx_score = batch['odx_score'].to(device)
                odx_hl = batch['odx_hl'].to(device)
                
                batch_loss = 0.0
                n_samples = len(slide_features_list)
                
                for i in range(n_samples):
                    try:
                        # Process one sample at a time
                        slide_features = slide_features_list[i].to(device)
                        
                        # If batch_size > 1, take the corresponding slice
                        if clinical_features.size(0) > 1:
                            clinical_feat = clinical_features[i:i+1]
                            odx_score_i = odx_score[i:i+1]
                            odx_hl_i = odx_hl[i:i+1]
                        else:
                            clinical_feat = clinical_features
                            odx_score_i = odx_score
                            odx_hl_i = odx_hl
                        
                        # Forward pass
                        outputs = model(slide_features.unsqueeze(0), clinical_feat)
                        
                        # Calculate losses
                        reg_loss = regression_criterion(outputs['odx_score_pred'], odx_score_i)
                        cls_loss = classification_criterion(outputs['odx_hl_logits'], odx_hl_i)
                        
                        # Combined loss
                        sample_loss = regression_weight * reg_loss + classification_weight * cls_loss
                        
                        # Backward pass for this sample
                        sample_loss.backward()
                        batch_loss += sample_loss.item()
                        
                    except Exception as e:
                        print(f"Error processing sample {i} in batch {batch_idx}: {str(e)}")
                        print(f"Slide feature shape: {slide_features_list[i].shape}")
                        continue
                
                # Optimizer step after processing all samples
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update train loss
                avg_batch_loss = batch_loss / n_samples
                train_loss += avg_batch_loss
                batch_count += 1
                
            else:
                # Standard batch processing
                try:
                    slide_features = batch['slide_features'].to(device)
                    clinical_features = batch['clinical_features'].to(device)
                    odx_score = batch['odx_score'].to(device)
                    odx_hl = batch['odx_hl'].to(device)
                    
                    # Forward pass
                    outputs = model(slide_features, clinical_features)
                    
                    # Calculate losses
                    reg_loss = regression_criterion(outputs['odx_score_pred'], odx_score)
                    cls_loss = classification_criterion(outputs['odx_hl_logits'], odx_hl)
                    
                    # Combined loss
                    loss = regression_weight * reg_loss + classification_weight * cls_loss
                    
                    # Backward and optimize
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {train_loss/batch_count:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Skip validation if we have no valid batches
        if batch_count == 0:
            print(f"Epoch {epoch+1}/{epochs} - No valid batches. Skipping validation.")
            continue
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Check if batch requires individual processing
                requires_individual = batch.get('batch_requires_individual_processing', False)
                
                if requires_individual:
                    # Process each sample individually
                    slide_features_list = batch['slide_features']  # List of tensors
                    clinical_features = batch['clinical_features'].to(device)
                    odx_score = batch['odx_score'].to(device)
                    odx_hl = batch['odx_hl'].to(device)
                    
                    batch_loss = 0.0
                    n_samples = len(slide_features_list)
                    
                    for i in range(n_samples):
                        try:
                            # Process one sample at a time
                            slide_features = slide_features_list[i].to(device)
                            
                            # If batch_size > 1, take the corresponding slice
                            if clinical_features.size(0) > 1:
                                clinical_feat = clinical_features[i:i+1]
                                odx_score_i = odx_score[i:i+1]
                                odx_hl_i = odx_hl[i:i+1]
                            else:
                                clinical_feat = clinical_features
                                odx_score_i = odx_score
                                odx_hl_i = odx_hl
                            
                            # Forward pass
                            outputs = model(slide_features.unsqueeze(0), clinical_feat)
                            
                            # Calculate losses
                            reg_loss = regression_criterion(outputs['odx_score_pred'], odx_score_i)
                            cls_loss = classification_criterion(outputs['odx_hl_logits'], odx_hl_i)
                            
                            # Combined loss
                            sample_loss = regression_weight * reg_loss + classification_weight * cls_loss
                            batch_loss += sample_loss.item()
                            
                            # Store predictions for metrics
                            all_preds.extend(outputs['odx_score_pred'].cpu().numpy())
                            all_targets.extend(odx_score_i.cpu().numpy())
                            
                        except Exception as e:
                            print(f"Error processing validation sample {i} in batch {batch_idx}: {str(e)}")
                            continue
                    
                    # Update validation loss
                    if n_samples > 0:
                        avg_batch_loss = batch_loss / n_samples
                        val_loss += avg_batch_loss
                        val_batch_count += 1
                    
                else:
                    # Standard batch processing
                    try:
                        slide_features = batch['slide_features'].to(device)
                        clinical_features = batch['clinical_features'].to(device)
                        odx_score = batch['odx_score'].to(device)
                        odx_hl = batch['odx_hl'].to(device)
                        
                        # Forward pass
                        outputs = model(slide_features, clinical_features)
                        
                        # Calculate losses
                        reg_loss = regression_criterion(outputs['odx_score_pred'], odx_score)
                        cls_loss = classification_criterion(outputs['odx_hl_logits'], odx_hl)
                        
                        # Combined loss
                        loss = regression_weight * reg_loss + classification_weight * cls_loss
                        
                        val_loss += loss.item()
                        val_batch_count += 1
                        
                        # Store predictions for metrics
                        all_preds.extend(outputs['odx_score_pred'].cpu().numpy())
                        all_targets.extend(odx_score.cpu().numpy())
                        
                    except Exception as e:
                        print(f"Error processing validation batch {batch_idx}: {str(e)}")
                        continue
        
        # Skip validation if we have no valid validation batches
        if val_batch_count == 0:
            print(f"Epoch {epoch+1}/{epochs} - No valid validation batches.")
            continue
        
        # Calculate validation metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        val_mae = mean_absolute_error(all_targets, all_preds)
        val_r2 = r2_score(all_targets, all_preds)
        
        # Update history
        history['train_loss'].append(train_loss / batch_count)
        history['val_loss'].append(val_loss / val_batch_count)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss/batch_count:.4f} - '
              f'Val Loss: {val_loss/val_batch_count:.4f} - '
              f'Val MAE: {val_mae:.4f} - '
              f'Val R²: {val_r2:.4f}')
    
    return model, history


def main():
    # Configuration
    data_path = 'train0.csv'
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    slide_feature_dim = 1024  # CTransPath feature dimension (adjust if needed)
    
    # Split data
    df = pd.read_csv(data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['odx_HL'])
    
    # Create temporary CSV files for train and validation sets
    train_df.to_csv('train_temp.csv', index=False)
    val_df.to_csv('val_temp.csv', index=False)
    
    # Create datasets and dataloaders
    train_dataset = BreastCancerDataset(csv_path='train_temp.csv')
    val_dataset = BreastCancerDataset(csv_path='val_temp.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = MultiModalBreastCancerModel(
        slide_feature_dim=slide_feature_dim,
        clinical_feature_dim=4,  # Length + 3 one-hot grade features
        dropout_rate=0.3
    )
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate
    )
    
    # Save model
    torch.save(trained_model.state_dict(), 'breast_cancer_odx_model.pt')
    
    # Clean up temporary files
    os.remove('train_temp.csv')
    os.remove('val_temp.csv')
    
    print("Training completed and model saved.")


if __name__ == "__main__":
    main()