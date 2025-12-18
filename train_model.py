# -*- coding: utf-8 -*-
""" ProInX: State-of-the-Art Machine Learning Model Training
==========================================================

This script trains advanced ML models to predict local Kd values from 
physicochemical features of protein patches and small molecules.

Models included:
- Traditional ML: Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
- Advanced ML: XGBoost, LightGBM, CatBoost
- Deep Learning: Multi-layer Perceptron with attention-like mechanisms
- Ensemble Methods: Stacking, Voting regressors
- Bayesian Optimization for hyperparameter tuning

IMPORTANT: This is a proof-of-concept model trained on SYNTHETIC data.
Real experimental validation is required before any application.

Author: ProInX Project/Jonas
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    KFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    StackingRegressor, VotingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import matplotlib.pyplot as plt
import json
import os
import pickle
import warnings
from typing import Dict, Tuple, List, Optional, Any
from scipy.stats import uniform, randint, loguniform

# Try to import advanced libraries (optional dependencies)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    warnings.warn("CatBoost not installed. Install with: pip install catboost")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not installed. Install with: pip install torch")

# Seed for reproducibility
np.random.seed(42)
if HAS_TORCH:
    torch.manual_seed(42)

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data(filepath: str = "data/synthetic_local_kd_data.csv") -> pd.DataFrame:
    """Load the synthetic dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Please run generate_synthetic_data.py first."
        )
    return pd.read_csv(filepath)


# =============================================================================
# DATA BALANCING AND AUGMENTATION (Inspired by recent protein ML papers)
# =============================================================================

def balance_dataset_by_kd_bins(df: pd.DataFrame, n_bins: int = 10, 
                                strategy: str = 'oversample') -> pd.DataFrame:
    """
    Balance dataset by Kd value bins to prevent bias towards common Kd ranges.
    
    Inspired by: "Deep Dive into Machine Learning Models for Protein Engineering"
    and data balancing strategies from protein-ligand binding papers.
    
    Args:
        df: Input dataframe
        n_bins: Number of Kd bins
        strategy: 'oversample', 'undersample', or 'smote'
    """
    df = df.copy()
    
    # Create Kd bins (log scale)
    df['kd_log'] = np.log10(df['kd_mM'])
    df['kd_bin'] = pd.cut(df['kd_log'], bins=n_bins, labels=False)
    
    bin_counts = df['kd_bin'].value_counts()
    print(f"\n   Original bin distribution:\n{bin_counts.sort_index()}")
    
    if strategy == 'oversample':
        # Oversample minority bins to match majority
        max_count = bin_counts.max()
        balanced_dfs = []
        
        for bin_id in range(n_bins):
            bin_df = df[df['kd_bin'] == bin_id]
            if len(bin_df) > 0:
                # Sample with replacement to reach max_count
                oversampled = bin_df.sample(n=max_count, replace=True, random_state=42)
                balanced_dfs.append(oversampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif strategy == 'undersample':
        # Undersample majority bins to match minority
        min_count = bin_counts[bin_counts > 0].min()
        balanced_dfs = []
        
        for bin_id in range(n_bins):
            bin_df = df[df['kd_bin'] == bin_id]
            if len(bin_df) > 0:
                undersampled = bin_df.sample(n=min(len(bin_df), min_count), 
                                             replace=False, random_state=42)
                balanced_dfs.append(undersampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif strategy == 'hybrid':
        # Hybrid: undersample majority, oversample minority to median
        median_count = int(bin_counts.median())
        balanced_dfs = []
        
        for bin_id in range(n_bins):
            bin_df = df[df['kd_bin'] == bin_id]
            if len(bin_df) > 0:
                if len(bin_df) > median_count:
                    sampled = bin_df.sample(n=median_count, replace=False, random_state=42)
                else:
                    sampled = bin_df.sample(n=median_count, replace=True, random_state=42)
                balanced_dfs.append(sampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    else:
        balanced_df = df
    
    # Remove helper columns
    balanced_df = balanced_df.drop(columns=['kd_log', 'kd_bin'])
    
    print(f"   Balanced dataset size: {len(balanced_df)} (from {len(df)})")
    return balanced_df


def augment_data_with_noise(X: np.ndarray, y: np.ndarray, 
                            noise_factor: float = 0.05,
                            n_augmentations: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Data augmentation by adding Gaussian noise to features.
    
    Inspired by: Augmentation strategies in "End-to-end learning on protein surfaces"
    and noise injection techniques from ESM protein models.
    
    Args:
        X: Feature matrix
        y: Target values
        noise_factor: Standard deviation of noise as fraction of feature std
        n_augmentations: Number of augmented copies to create
    """
    augmented_X = [X]
    augmented_y = [y]
    
    for i in range(n_augmentations):
        # Add Gaussian noise scaled by feature standard deviation
        noise = np.random.normal(0, noise_factor, X.shape) * np.std(X, axis=0)
        X_noisy = X + noise
        augmented_X.append(X_noisy)
        augmented_y.append(y)
    
    return np.vstack(augmented_X), np.hstack(augmented_y)


def create_stratified_kfold_by_kd(X: np.ndarray, y: np.ndarray, 
                                   n_splits: int = 5) -> List[Tuple]:
    """
    Create stratified K-fold splits based on Kd value bins.
    Ensures each fold has similar Kd distribution.
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Create bins for stratification
    y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    splits = []
    for train_idx, test_idx in skf.split(X, y_bins):
        splits.append((train_idx, test_idx))
    
    return splits


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix X and target vector y.
    
    Features used:
    - Patch physicochemical properties (hydrophobicity, charge, aromatic_count, sasa)
    - Small molecule encoding (label encoded)
    - Protein encoding (label encoded)
    """
    # Encode categorical variables
    le_protein = LabelEncoder()
    le_sm = LabelEncoder()
    
    df = df.copy()
    df['protein_encoded'] = le_protein.fit_transform(df['protein'])
    df['sm_encoded'] = le_sm.fit_transform(df['small_molecule'])
    
    # Feature columns
    feature_cols = [
        'hydrophobicity',
        'charge', 
        'aromatic_count',
        'sasa',
        'protein_encoded',
        'sm_encoded',
    ]
    
    X = df[feature_cols].values
    y = df['kd_mM'].values
    
    # Return log-transformed Kd for better model performance
    y_log = np.log10(y)
    
    return X, y_log, feature_cols


def prepare_features_onehot(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features with one-hot encoding for categorical variables.
    """
    # One-hot encode protein and small molecule
    df_encoded = pd.get_dummies(df, columns=['protein', 'small_molecule'], prefix=['prot', 'sm'])
    
    # Numeric feature columns
    numeric_cols = ['hydrophobicity', 'charge', 'aromatic_count', 'sasa']
    
    # Get one-hot columns
    onehot_cols = [c for c in df_encoded.columns if c.startswith('prot_') or c.startswith('sm_')]
    
    feature_cols = numeric_cols + onehot_cols
    
    X = df_encoded[feature_cols].values
    y = df['kd_mM'].values
    y_log = np.log10(y)
    
    return X, y_log, feature_cols


def prepare_features_advanced(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Advanced feature engineering with interaction terms and polynomial features.
    """
    df = df.copy()
    
    # Basic features
    numeric_cols = ['hydrophobicity', 'charge', 'aromatic_count', 'sasa']
    
    # Create interaction features
    df['hydro_charge'] = df['hydrophobicity'] * df['charge']
    df['hydro_sasa'] = df['hydrophobicity'] * df['sasa']
    df['charge_sasa'] = df['charge'] * df['sasa']
    df['aromatic_sasa'] = df['aromatic_count'] * df['sasa']
    
    # Polynomial features
    df['hydro_sq'] = df['hydrophobicity'] ** 2
    df['sasa_sq'] = df['sasa'] ** 2
    
    # Normalized features
    df['hydro_norm'] = (df['hydrophobicity'] - df['hydrophobicity'].mean()) / df['hydrophobicity'].std()
    
    # One-hot encode categorical
    df_encoded = pd.get_dummies(df, columns=['protein', 'small_molecule'], prefix=['prot', 'sm'])
    
    # All feature columns
    interaction_cols = ['hydro_charge', 'hydro_sasa', 'charge_sasa', 'aromatic_sasa', 
                        'hydro_sq', 'sasa_sq', 'hydro_norm']
    onehot_cols = [c for c in df_encoded.columns if c.startswith('prot_') or c.startswith('sm_')]
    
    feature_cols = numeric_cols + interaction_cols + onehot_cols
    
    X = df_encoded[feature_cols].values
    y = df['kd_mM'].values
    y_log = np.log10(y)
    
    return X, y_log, feature_cols


# =============================================================================
# REGULARIZATION TECHNIQUES (State-of-the-art methods)
# =============================================================================

class RegularizedFeatureSelector:
    """
    Feature selection using multiple regularization techniques.
    
    Inspired by: Feature importance methods from AlphaFold2 and 
    protein engineering ML papers.
    """
    def __init__(self, method: str = 'elastic_net', threshold: float = 0.01):
        self.method = method
        self.threshold = threshold
        self.selected_features = None
        self.feature_importances = None
        self.selector_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Fit the feature selector."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if self.method == 'elastic_net':
            # ElasticNet for sparse feature selection
            from sklearn.linear_model import ElasticNetCV
            self.selector_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                                               cv=5, random_state=42)
            self.selector_model.fit(X_scaled, y)
            self.feature_importances = np.abs(self.selector_model.coef_)
            
        elif self.method == 'lasso':
            from sklearn.linear_model import LassoCV
            self.selector_model = LassoCV(cv=5, random_state=42)
            self.selector_model.fit(X_scaled, y)
            self.feature_importances = np.abs(self.selector_model.coef_)
            
        elif self.method == 'rf_importance':
            from sklearn.ensemble import RandomForestRegressor
            self.selector_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.selector_model.fit(X_scaled, y)
            self.feature_importances = self.selector_model.feature_importances_
            
        elif self.method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            self.feature_importances = mutual_info_regression(X_scaled, y, random_state=42)
        
        # Select features above threshold
        max_importance = self.feature_importances.max()
        self.selected_features = self.feature_importances >= (self.threshold * max_importance)
        
        if feature_names is not None:
            selected_names = [name for name, selected in zip(feature_names, self.selected_features) if selected]
            print(f"   Selected {len(selected_names)} features out of {len(feature_names)}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to selected features only."""
        return X[:, self.selected_features]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names)
        return self.transform(X)


def add_l1_l2_regularization_cv(X_train: np.ndarray, y_train: np.ndarray,
                                 model_type: str = 'ridge') -> Tuple[Any, Dict]:
    """
    Find optimal L1/L2 regularization strength via cross-validation.
    
    Returns the best model and optimal parameters.
    """
    from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    if model_type == 'ridge':
        alphas = np.logspace(-4, 4, 50)
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X_scaled, y_train)
        best_params = {'alpha': model.alpha_}
        
    elif model_type == 'lasso':
        model = LassoCV(cv=5, random_state=42)
        model.fit(X_scaled, y_train)
        best_params = {'alpha': model.alpha_}
        
    elif model_type == 'elastic_net':
        model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, random_state=42)
        model.fit(X_scaled, y_train)
        best_params = {'alpha': model.alpha_, 'l1_ratio': model.l1_ratio_}
    
    print(f"   Best regularization params for {model_type}: {best_params}")
    return model, best_params


class DropoutRegularization:
    """
    Apply dropout-style regularization during training by randomly masking features.
    
    Inspired by: Dropout techniques used in transformer models for protein structure.
    """
    def __init__(self, dropout_rate: float = 0.1):
        self.dropout_rate = dropout_rate
    
    def apply(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, X.shape)
            # Scale up to maintain expected value
            return X * mask / (1 - self.dropout_rate)
        return X


def mixup_augmentation(X: np.ndarray, y: np.ndarray, 
                       alpha: float = 0.2, n_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    MixUp data augmentation for regression.
    
    Inspired by: MixUp techniques from computer vision adapted for molecular property prediction.
    Reference: "Deep Dive into Machine Learning Models for Protein Engineering"
    
    Creates new training examples by linear interpolation between existing examples.
    """
    if n_samples is None:
        n_samples = len(X)
    
    # Sample mixing coefficients from Beta distribution
    lam = np.random.beta(alpha, alpha, n_samples)
    
    # Random indices for mixing
    idx1 = np.random.randint(0, len(X), n_samples)
    idx2 = np.random.randint(0, len(X), n_samples)
    
    # Mix features
    lam_expanded = lam.reshape(-1, 1)
    X_mixed = lam_expanded * X[idx1] + (1 - lam_expanded) * X[idx2]
    
    # Mix targets
    y_mixed = lam * y[idx1] + (1 - lam) * y[idx2]
    
    return X_mixed, y_mixed


def uncertainty_quantification_ensemble(models: List, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction uncertainty using ensemble of models.
    
    Inspired by: Deep ensemble uncertainty quantification from 
    "Accurate uncertainties for deep learning using calibrated regression"
    
    Returns mean prediction and uncertainty (std).
    """
    predictions = np.array([model.predict(X) for model in models])
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    return mean_pred, std_pred


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Inspired by: Bayesian deep learning techniques used in protein property prediction.
    """
    def __init__(self, base_model, n_iterations: int = 50, dropout_rate: float = 0.1):
        self.base_model = base_model
        self.n_iterations = n_iterations
        self.dropout = DropoutRegularization(dropout_rate)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        
        for _ in range(self.n_iterations):
            X_dropout = self.dropout.apply(X, training=True)
            pred = self.base_model.predict(X_dropout)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_baseline_models() -> Dict[str, Any]:
    """
    Get dictionary of baseline sklearn models.
    """
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'BayesianRidge': BayesianRidge(),
        'SVR_RBF': SVR(kernel='rbf', C=10, gamma='scale'),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', 
                           max_iter=500, random_state=42, early_stopping=True),
    }
    return models


def get_advanced_models() -> Dict[str, Any]:
    """
    Get dictionary of advanced gradient boosting models.
    """
    models = {}
    
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
    
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0
        )
    
    return models


def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Train and evaluate multiple baseline models.
    """
    models = get_baseline_models()
    
    results = {}
    
    for name, model in models.items():
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Evaluate (on log scale)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Also evaluate on original Kd scale
        y_test_orig = 10 ** y_test
        y_pred_test_orig = 10 ** y_pred_test
        test_mae_mM = mean_absolute_error(y_test_orig, y_pred_test_orig)
        
        results[name] = {
            'model': pipeline,
            'train_rmse_log': train_rmse,
            'test_rmse_log': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae_mM': test_mae_mM,
        }
        
        print(f"\n{name}:")
        print(f"  Train RMSE (log): {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"  Test  RMSE (log): {test_rmse:.4f}, R2: {test_r2:.4f}")
        print(f"  Test  MAE (mM):   {test_mae_mM:.2f}")
    
    return results


def train_advanced_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Train and evaluate advanced gradient boosting models.
    """
    models = get_advanced_models()
    
    if not models:
        print("No advanced models available. Install xgboost, lightgbm, or catboost.")
        return {}
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        y_test_orig = 10 ** y_test
        y_pred_test_orig = 10 ** y_pred_test
        test_mae_mM = mean_absolute_error(y_test_orig, y_pred_test_orig)
        
        results[name] = {
            'model': pipeline,
            'train_rmse_log': train_rmse,
            'test_rmse_log': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae_mM': test_mae_mM,
        }
        
        print(f"  Train RMSE (log): {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"  Test  RMSE (log): {test_rmse:.4f}, R2: {test_r2:.4f}")
        print(f"  Test  MAE (mM):   {test_mae_mM:.2f}")
    
    return results


# =============================================================================
# DEEP LEARNING MODEL (PyTorch) with Advanced Regularization
# =============================================================================

if HAS_TORCH:
    class AttentionBlock(nn.Module):
        """Self-attention mechanism for feature importance weighting."""
        def __init__(self, embed_dim: int):
            super().__init__()
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)
            self.scale = embed_dim ** 0.5
        
        def forward(self, x):
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            
            attn_weights = torch.softmax(q * k / self.scale, dim=-1)
            return attn_weights * v

    class MultiHeadSelfAttention(nn.Module):
        """
        Multi-head self-attention for capturing different aspects of feature interactions.
        
        Inspired by: Transformer architectures in ESM-2 and AlphaFold2.
        """
        def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = self.head_dim ** -0.5
        
        def forward(self, x):
            batch_size = x.size(0)
            
            # Project and reshape for multi-head attention
            q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention scores
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention to values
            out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, -1)
            return self.out_proj(out)

    class SpectralNormLinear(nn.Module):
        """
        Linear layer with spectral normalization for Lipschitz constraint.
        
        Inspired by: Spectral normalization techniques for stable training.
        Reference: "Spectral Normalization for Generative Adversarial Networks"
        """
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.utils.spectral_norm(nn.Linear(in_features, out_features))
        
        def forward(self, x):
            return self.linear(x)

    class LabelSmoothing(nn.Module):
        """
        Label smoothing for regression to prevent overconfident predictions.
        
        Adapted from classification label smoothing for regression tasks.
        """
        def __init__(self, smoothing: float = 0.1):
            super().__init__()
            self.smoothing = smoothing
        
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # Add noise to targets for smoothing effect
            noise = torch.randn_like(target) * self.smoothing
            smoothed_target = target + noise
            return nn.MSELoss()(pred, smoothed_target)

    class FocalMSELoss(nn.Module):
        """
        Focal MSE Loss - emphasizes hard examples.
        
        Inspired by: Focal loss adapted for regression to focus on samples
        with high prediction error.
        Reference: "Focal Loss for Dense Object Detection" adapted for regression.
        """
        def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
            super().__init__()
            self.gamma = gamma
            self.reduction = reduction
        
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            mse = (pred - target) ** 2
            # Weight by error magnitude (higher weight for harder examples)
            weight = (mse.detach() + 1e-8) ** (self.gamma / 2)
            focal_mse = weight * mse
            
            if self.reduction == 'mean':
                return focal_mse.mean()
            elif self.reduction == 'sum':
                return focal_mse.sum()
            return focal_mse

    class KdPredictorNet(nn.Module):
        """
        Deep neural network for Kd prediction with advanced regularization.
        
        Incorporates:
        - Multi-head self-attention (inspired by ESM-2/AlphaFold2)
        - Spectral normalization (Lipschitz constraint)
        - Residual connections with layer normalization
        - Adaptive dropout (increased for deeper layers)
        - GELU activation (modern activation function)
        """
        def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64, 32],
                     dropout: float = 0.2, use_attention: bool = True,
                     use_spectral_norm: bool = True, num_attention_heads: int = 4):
            super().__init__()
            
            self.use_attention = use_attention
            self.use_spectral_norm = use_spectral_norm
            
            # Input projection with optional spectral norm
            if use_spectral_norm:
                self.input_proj = SpectralNormLinear(input_dim, hidden_dims[0])
            else:
                self.input_proj = nn.Linear(input_dim, hidden_dims[0])
            
            self.input_ln = nn.LayerNorm(hidden_dims[0])
            
            # Multi-head attention block (optional)
            if use_attention:
                self.attention = MultiHeadSelfAttention(
                    hidden_dims[0], num_heads=num_attention_heads, dropout=dropout
                )
                self.attention_ln = nn.LayerNorm(hidden_dims[0])
            
            # Hidden layers with residual connections
            self.hidden_layers = nn.ModuleList()
            self.layer_norms = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            
            for i in range(len(hidden_dims) - 1):
                if use_spectral_norm:
                    self.hidden_layers.append(SpectralNormLinear(hidden_dims[i], hidden_dims[i+1]))
                else:
                    self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                self.layer_norms.append(nn.LayerNorm(hidden_dims[i+1]))
                # Adaptive dropout: increase for deeper layers
                adaptive_dropout = dropout * (1 + 0.1 * i)
                self.dropouts.append(nn.Dropout(min(adaptive_dropout, 0.5)))
            
            # Output layer
            self.output = nn.Linear(hidden_dims[-1], 1)
            
            # Activation
            self.activation = nn.GELU()
            
            # Skip connections for residual learning
            self.skip_projections = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                if hidden_dims[i] != hidden_dims[i+1]:
                    self.skip_projections.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                else:
                    self.skip_projections.append(nn.Identity())
        
        def forward(self, x):
            # Input projection
            x = self.input_proj(x)
            x = self.input_ln(x)
            x = self.activation(x)
            
            # Multi-head attention with residual
            if self.use_attention:
                # Reshape for attention (add sequence dimension)
                x_attn = x.unsqueeze(1)
                attn_out = self.attention(x_attn)
                x = x + attn_out.squeeze(1) if attn_out.dim() > 2 else x + attn_out
                x = self.attention_ln(x)
            
            # Hidden layers with residual connections
            for i, (linear, ln, dropout) in enumerate(zip(self.hidden_layers, self.layer_norms, self.dropouts)):
                identity = self.skip_projections[i](x)
                x = linear(x)
                x = ln(x)
                x = self.activation(x)
                x = dropout(x)
                x = x + identity  # Residual connection
            
            # Output
            return self.output(x).squeeze(-1)

    class DeepKdPredictor:
        """
        Wrapper class for PyTorch deep learning model with sklearn-like interface.
        
        Incorporates:
        - Early stopping with patience
        - Learning rate scheduling (cosine annealing with warm restarts)
        - Gradient clipping for stability
        - MixUp augmentation during training
        - Multiple loss functions (MSE, Focal MSE, Huber)
        - Model checkpointing
        """
        def __init__(self, input_dim: int = None, hidden_dims: List[int] = [256, 128, 64, 32],
                     dropout: float = 0.2, learning_rate: float = 0.001,
                     batch_size: int = 32, epochs: int = 100, 
                     early_stopping_patience: int = 10, use_attention: bool = True,
                     use_spectral_norm: bool = True, use_mixup: bool = True,
                     mixup_alpha: float = 0.2, loss_type: str = 'mse',
                     weight_decay: float = 0.01, use_focal_loss: bool = False):
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.early_stopping_patience = early_stopping_patience
            self.use_attention = use_attention
            self.use_spectral_norm = use_spectral_norm
            self.use_mixup = use_mixup
            self.mixup_alpha = mixup_alpha
            self.loss_type = loss_type
            self.weight_decay = weight_decay
            self.use_focal_loss = use_focal_loss
            self.model = None
            self.scaler = StandardScaler()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.train_losses = []
            self.val_losses = []
            self.best_model_state = None
        
        def _get_loss_function(self):
            """Get the appropriate loss function."""
            if self.use_focal_loss:
                return FocalMSELoss(gamma=2.0)
            elif self.loss_type == 'huber':
                return nn.HuberLoss(delta=1.0)
            elif self.loss_type == 'smooth_l1':
                return nn.SmoothL1Loss()
            else:
                return nn.MSELoss()
        
        def _apply_mixup(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Apply MixUp augmentation."""
            if not self.use_mixup:
                return X, y
            
            batch_size = X.size(0)
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # Random permutation for mixing
            idx = torch.randperm(batch_size).to(self.device)
            
            X_mixed = lam * X + (1 - lam) * X[idx]
            y_mixed = lam * y + (1 - lam) * y[idx]
            
            return X_mixed, y_mixed
        
        def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
            # Set input dimension
            if self.input_dim is None:
                self.input_dim = X.shape[1]
            
            # Scale data
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Validation data
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Initialize model
            self.model = KdPredictorNet(
                self.input_dim, self.hidden_dims, self.dropout, 
                self.use_attention, self.use_spectral_norm
            ).to(self.device)
            
            # Optimizer with weight decay (L2 regularization)
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
            
            # Cosine annealing scheduler with warm restarts
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            
            # Loss function
            criterion = self._get_loss_function()
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0.0
                
                for batch_X, batch_y in loader:
                    # Apply MixUp augmentation
                    batch_X, batch_y = self._apply_mixup(batch_X, batch_y)
                    
                    optimizer.zero_grad()
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                
                scheduler.step()
                
                avg_train_loss = epoch_loss / len(loader)
                self.train_losses.append(avg_train_loss)
                
                # Validation
                if X_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred = self.model(X_val_tensor)
                        val_loss = criterion(val_pred, y_val_tensor).item()
                    self.val_losses.append(val_loss)
                    
                    # Early stopping with model checkpointing
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            self.model.load_state_dict(self.best_model_state)
                            print(f"Early stopping at epoch {epoch + 1}")
                            break
                
                if (epoch + 1) % 20 == 0:
                    val_str = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                    print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}{val_str}")
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()
            
            return predictions
        
        def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
            """
            Predict with uncertainty using Monte Carlo Dropout.
            """
            self.model.train()  # Enable dropout
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            predictions = []
            for _ in range(n_samples):
                with torch.no_grad():
                    pred = self.model(X_tensor).cpu().numpy()
                    predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            self.model.eval()  # Disable dropout
            return mean_pred, std_pred


def train_deep_learning_model(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
    """
    Train deep learning model for Kd prediction with advanced regularization.
    
    Includes:
    - MixUp data augmentation
    - Spectral normalization
    - Multi-head attention
    - Focal loss option
    - Monte Carlo dropout for uncertainty
    """
    if not HAS_TORCH:
        print("PyTorch not available. Skipping deep learning model.")
        return {}
    
    print("\nTraining Deep Learning Model (PyTorch with Advanced Regularization)...")
    
    # Use test set as validation if no validation set provided
    if X_val is None:
        X_val, y_val = X_test, y_test
    
    model = DeepKdPredictor(
        input_dim=X_train.shape[1],
        hidden_dims=[256, 128, 64, 32],
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=200,
        early_stopping_patience=15,
        use_attention=True,
        use_spectral_norm=True,
        use_mixup=True,
        mixup_alpha=0.2,
        weight_decay=0.01,
        use_focal_loss=False  # Can enable for hard example mining
    )
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    y_test_orig = 10 ** y_test
    y_pred_test_orig = 10 ** y_pred_test
    test_mae_mM = mean_absolute_error(y_test_orig, y_pred_test_orig)
    
    # Get uncertainty estimates
    mean_pred, uncertainty = model.predict_with_uncertainty(X_test, n_samples=30)
    avg_uncertainty = uncertainty.mean()
    
    print(f"  Train RMSE (log): {train_rmse:.4f}, R2: {train_r2:.4f}")
    print(f"  Test  RMSE (log): {test_rmse:.4f}, R2: {test_r2:.4f}")
    print(f"  Test  MAE (mM):   {test_mae_mM:.2f}")
    print(f"  Avg Uncertainty:  {avg_uncertainty:.4f}")
    
    return {
        'DeepLearning': {
            'model': model,
            'train_rmse_log': train_rmse,
            'test_rmse_log': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae_mM': test_mae_mM,
            'avg_uncertainty': avg_uncertainty,
        }
    }


# =============================================================================
# ENSEMBLE WITH UNCERTAINTY QUANTIFICATION
# =============================================================================

def build_deep_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        n_models: int = 5) -> Dict:
    """
    Build deep ensemble for uncertainty quantification.
    
    Inspired by: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
    Reference: Lakshminarayanan et al., NeurIPS 2017
    """
    print(f"\nBuilding Deep Ensemble ({n_models} models)...")
    
    ensemble_models = []
    all_predictions = []
    
    for i in range(n_models):
        print(f"  Training model {i+1}/{n_models}...")
        
        # Bootstrap sampling for diversity
        n_samples = len(X_train)
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[bootstrap_idx]
        y_boot = y_train[bootstrap_idx]
        
        # Different random initialization for each model
        np.random.seed(42 + i)
        
        # Train model with slight hyperparameter variation
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100 + i * 20,
                max_depth=5 + (i % 3),
                learning_rate=0.1 - i * 0.01,
                random_state=42 + i
            ))
        ])
        
        pipeline.fit(X_boot, y_boot)
        ensemble_models.append(pipeline)
        
        pred = pipeline.predict(X_test)
        all_predictions.append(pred)
    
    # Ensemble predictions
    all_predictions = np.array(all_predictions)
    mean_pred = all_predictions.mean(axis=0)
    std_pred = all_predictions.std(axis=0)
    
    # Evaluate
    train_preds = np.array([m.predict(X_train) for m in ensemble_models]).mean(axis=0)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, mean_pred)
    
    y_test_orig = 10 ** y_test
    mean_pred_orig = 10 ** mean_pred
    test_mae_mM = mean_absolute_error(y_test_orig, mean_pred_orig)
    
    print(f"  Train RMSE (log): {train_rmse:.4f}, R2: {train_r2:.4f}")
    print(f"  Test  RMSE (log): {test_rmse:.4f}, R2: {test_r2:.4f}")
    print(f"  Test  MAE (mM):   {test_mae_mM:.2f}")
    print(f"  Avg Uncertainty:  {std_pred.mean():.4f}")
    
    return {
        'DeepEnsemble': {
            'model': ensemble_models,
            'train_rmse_log': train_rmse,
            'test_rmse_log': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae_mM': test_mae_mM,
            'uncertainty': std_pred,
        }
    }


def train_with_cross_validation_averaging(X: np.ndarray, y: np.ndarray,
                                          X_test: np.ndarray, y_test: np.ndarray,
                                          n_folds: int = 5) -> Dict:
    """
    Train models with K-fold cross-validation and average predictions.
    
    Reduces overfitting by averaging predictions from models trained on different folds.
    """
    print(f"\nTraining with {n_folds}-Fold Cross-Validation Averaging...")
    
    kfold_splits = create_stratified_kfold_by_kd(X, y, n_splits=n_folds)
    
    fold_models = []
    oof_predictions = np.zeros(len(X))
    test_predictions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold_splits):
        print(f"  Fold {fold_idx + 1}/{n_folds}...")
        
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Train model
        if HAS_XGBOOST:
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, random_state=42
            )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X_fold_train, y_fold_train)
        fold_models.append(pipeline)
        
        # Out-of-fold predictions
        oof_predictions[val_idx] = pipeline.predict(X_fold_val)
        
        # Test predictions
        test_predictions.append(pipeline.predict(X_test))
    
    # Average test predictions
    avg_test_pred = np.array(test_predictions).mean(axis=0)
    
    # Evaluate
    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, avg_test_pred))
    oof_r2 = r2_score(y, oof_predictions)
    test_r2 = r2_score(y_test, avg_test_pred)
    
    y_test_orig = 10 ** y_test
    avg_test_pred_orig = 10 ** avg_test_pred
    test_mae_mM = mean_absolute_error(y_test_orig, avg_test_pred_orig)
    
    print(f"  OOF   RMSE (log): {oof_rmse:.4f}, R2: {oof_r2:.4f}")
    print(f"  Test  RMSE (log): {test_rmse:.4f}, R2: {test_r2:.4f}")
    print(f"  Test  MAE (mM):   {test_mae_mM:.2f}")
    
    return {
        'CV_Averaged': {
            'model': fold_models,
            'train_rmse_log': oof_rmse,
            'test_rmse_log': test_rmse,
            'train_r2': oof_r2,
            'test_r2': test_r2,
            'test_mae_mM': test_mae_mM,
        }
    }


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================

def build_stacking_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Build stacking ensemble with diverse base learners.
    """
    print("\nBuilding Stacking Ensemble...")
    
    # Base estimators
    estimators = [
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
    ]
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        estimators.append(('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)))
    
    # Stacking regressor with Ridge as final estimator
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.5),
        cv=5,
        n_jobs=-1
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('stacking', stacking)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    y_test_orig = 10 ** y_test
    y_pred_test_orig = 10 ** y_pred_test
    test_mae_mM = mean_absolute_error(y_test_orig, y_pred_test_orig)
    
    print(f"  Train RMSE (log): {train_rmse:.4f}, R2: {train_r2:.4f}")
    print(f"  Test  RMSE (log): {test_rmse:.4f}, R2: {test_r2:.4f}")
    print(f"  Test  MAE (mM):   {test_mae_mM:.2f}")
    
    return {
        'StackingEnsemble': {
            'model': pipeline,
            'train_rmse_log': train_rmse,
            'test_rmse_log': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae_mM': test_mae_mM,
        }
    }


# =============================================================================
# BAYESIAN OPTIMIZATION (Optuna)
# =============================================================================

def optuna_optimize_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            n_trials: int = 50) -> Dict:
    """
    Use Optuna for Bayesian optimization of XGBoost hyperparameters.
    """
    if not HAS_OPTUNA or not HAS_XGBOOST:
        print("Optuna or XGBoost not available. Skipping Bayesian optimization.")
        return {}
    
    print(f"\nRunning Bayesian Optimization with Optuna ({n_trials} trials)...")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'verbosity': 0
        }
        
        model = xgb.XGBRegressor(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                    scoring='neg_mean_squared_error', n_jobs=-1)
        return -cv_scores.mean()
    
    # Optimize
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest trial: RMSE = {np.sqrt(study.best_trial.value):.4f}")
    print(f"Best parameters: {study.best_trial.params}")
    
    # Train final model with best parameters
    best_params = study.best_trial.params
    best_params['random_state'] = 42
    best_params['verbosity'] = 0
    
    best_model = xgb.XGBRegressor(**best_params)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    y_test_orig = 10 ** y_test
    y_pred_test_orig = 10 ** y_pred_test
    test_mae_mM = mean_absolute_error(y_test_orig, y_pred_test_orig)
    
    print(f"\nOptimized XGBoost:")
    print(f"  Train RMSE (log): {train_rmse:.4f}, R2: {train_r2:.4f}")
    print(f"  Test  RMSE (log): {test_rmse:.4f}, R2: {test_r2:.4f}")
    print(f"  Test  MAE (mM):   {test_mae_mM:.2f}")
    
    return {
        'XGBoost_Optimized': {
            'model': pipeline,
            'train_rmse_log': train_rmse,
            'test_rmse_log': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae_mM': test_mae_mM,
            'best_params': best_params,
            'optuna_study': study,
        }
    }


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Perform hyperparameter tuning for Random Forest model.
    """
    print("\nPerforming hyperparameter tuning for Random Forest...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_split': [2, 5, 10],
    }
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_


def cross_validate_model(model: Pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
    """
    Perform cross-validation and return metrics.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    
    return {
        'cv_rmse_mean': rmse_scores.mean(),
        'cv_rmse_std': rmse_scores.std(),
        'cv_rmse_scores': rmse_scores.tolist(),
    }


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(model: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract and analyze feature importance from tree-based models.
    """
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str = "Predicted vs Actual Kd",
                     save_path: str = None):
    """
    Plot predicted vs actual values.
    """
    # Convert from log scale to original
    y_true_orig = 10 ** y_true
    y_pred_orig = 10 ** y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Log scale plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual log10(Kd)')
    axes[0].set_ylabel('Predicted log10(Kd)')
    axes[0].set_title(f'{title} (log scale)')
    
    r2 = r2_score(y_true, y_pred)
    axes[0].text(0.05, 0.95, f'R2 = {r2:.3f}', transform=axes[0].transAxes,
                 fontsize=12, verticalalignment='top')
    
    # Original scale plot
    axes[1].scatter(y_true_orig, y_pred_orig, alpha=0.5, s=20)
    axes[1].plot([y_true_orig.min(), y_true_orig.max()], 
                 [y_true_orig.min(), y_true_orig.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Kd (mM)')
    axes[1].set_ylabel('Predicted Kd (mM)')
    axes[1].set_title(f'{title} (mM scale)')
    
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    axes[1].text(0.05, 0.95, f'MAE = {mae:.1f} mM', transform=axes[1].transAxes,
                 fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, 
                            top_n: int = 15,
                            save_path: str = None):
    """
    Plot feature importance bar chart.
    """
    if importance_df is None:
        return
    
    df_plot = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(df_plot)), df_plot['importance'].values, color='steelblue')
    plt.yticks(range(len(df_plot)), df_plot['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances for Kd Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot residuals analysis.
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted log10(Kd)')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    
    # Residuals histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# =============================================================================
# MODEL SAVING AND LOADING
# =============================================================================

def save_model(model: Pipeline, filepath: str, metadata: Dict = None):
    """
    Save trained model and metadata.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_dict = {
        'model': model,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Tuple[Pipeline, Dict]:
    """
    Load trained model and metadata.
    """
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    
    return save_dict['model'], save_dict.get('metadata', {})


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main training pipeline with state-of-the-art models and advanced regularization.
    
    Incorporates methods from recent protein ML papers:
    - Data balancing by Kd bins
    - MixUp augmentation
    - Feature selection with regularization
    - Spectral normalization
    - Focal loss for hard example mining
    - Deep ensembles for uncertainty quantification
    - Cross-validation averaging
    """
    print("=" * 70)
    print("PatchKd State-of-the-Art Model Training")
    print("With Advanced Regularization & Data Balancing")
    print("=" * 70)
    
    # Print available libraries
    print("\nAvailable libraries:")
    print(f"  XGBoost:   {'Yes' if HAS_XGBOOST else 'No'}")
    print(f"  LightGBM:  {'Yes' if HAS_LIGHTGBM else 'No'}")
    print(f"  CatBoost:  {'Yes' if HAS_CATBOOST else 'No'}")
    print(f"  PyTorch:   {'Yes' if HAS_TORCH else 'No'}")
    print(f"  Optuna:    {'Yes' if HAS_OPTUNA else 'No'}")
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Load data
    print("\n" + "=" * 70)
    print("1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} data points")
    
    # Balance dataset by Kd bins
    print("\n2. Balancing dataset by Kd bins...")
    df_balanced = balance_dataset_by_kd_bins(df, n_bins=10, strategy='hybrid')
    
    # Prepare features with advanced engineering
    print("\n3. Preparing features (with advanced engineering)...")
    X, y, feature_names = prepare_features_advanced(df_balanced)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of features: {len(feature_names)}")
    
    # Feature selection with regularization
    print("\n4. Feature selection with regularization...")
    feature_selector = RegularizedFeatureSelector(method='elastic_net', threshold=0.05)
    X_selected = feature_selector.fit_transform(X, y, feature_names)
    selected_feature_names = [name for name, selected in zip(feature_names, feature_selector.selected_features) if selected]
    print(f"   Selected features: {len(selected_feature_names)}")
    
    # Use selected features for training
    X = X_selected
    feature_names = selected_feature_names
    
    # Split data
    print("\n5. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split training for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"   Train set:      {len(X_train_final)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    print(f"   Test set:       {len(X_test)} samples")
    
    # Data augmentation with MixUp
    print("\n6. Applying MixUp data augmentation...")
    X_train_augmented, y_train_augmented = mixup_augmentation(
        X_train, y_train, alpha=0.2, n_samples=len(X_train) // 2
    )
    X_train_full = np.vstack([X_train, X_train_augmented])
    y_train_full = np.hstack([y_train, y_train_augmented])
    print(f"   Augmented training set: {len(X_train_full)} samples")
    
    # Collect all results
    all_results = {}
    
    # Train baseline models
    print("\n" + "=" * 70)
    print("7. Training BASELINE models...")
    print("-" * 70)
    baseline_results = train_baseline_models(X_train_full, y_train_full, X_test, y_test)
    all_results.update(baseline_results)
    
    # Train advanced gradient boosting models
    print("\n" + "=" * 70)
    print("8. Training ADVANCED gradient boosting models...")
    print("-" * 70)
    advanced_results = train_advanced_models(X_train_full, y_train_full, X_test, y_test)
    all_results.update(advanced_results)
    
    # Train deep learning model with advanced regularization
    print("\n" + "=" * 70)
    print("9. Training DEEP LEARNING model (with advanced regularization)...")
    print("-" * 70)
    dl_results = train_deep_learning_model(X_train_final, y_train_final, X_test, y_test, X_val, y_val)
    all_results.update(dl_results)
    
    # Build stacking ensemble
    print("\n" + "=" * 70)
    print("10. Building STACKING ENSEMBLE...")
    print("-" * 70)
    ensemble_results = build_stacking_ensemble(X_train_full, y_train_full, X_test, y_test)
    all_results.update(ensemble_results)
    
    # Build deep ensemble for uncertainty quantification
    print("\n" + "=" * 70)
    print("11. Building DEEP ENSEMBLE (Uncertainty Quantification)...")
    print("-" * 70)
    deep_ensemble_results = build_deep_ensemble(X_train, y_train, X_test, y_test, n_models=5)
    all_results.update(deep_ensemble_results)
    
    # Cross-validation averaged model
    print("\n" + "=" * 70)
    print("12. Training with CROSS-VALIDATION AVERAGING...")
    print("-" * 70)
    cv_avg_results = train_with_cross_validation_averaging(X_train, y_train, X_test, y_test, n_folds=5)
    all_results.update(cv_avg_results)
    
    # Bayesian optimization
    print("\n" + "=" * 70)
    print("13. BAYESIAN OPTIMIZATION with Optuna...")
    print("-" * 70)
    optuna_results = optuna_optimize_xgboost(X_train_full, y_train_full, X_test, y_test, n_trials=30)
    all_results.update(optuna_results)
    
    # Find best model
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    # Create comparison dataframe
    comparison_data = []
    for name, res in all_results.items():
        comparison_data.append({
            'Model': name,
            'Train_RMSE_log': res['train_rmse_log'],
            'Test_RMSE_log': res['test_rmse_log'],
            'Train_R2': res['train_r2'],
            'Test_R2': res['test_r2'],
            'Test_MAE_mM': res['test_mae_mM'],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_RMSE_log')
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = all_results[best_model_name]['model']
    print(f"\n*** BEST MODEL: {best_model_name} ***")
    
    # Overfitting analysis
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS")
    print("=" * 70)
    comparison_df['Overfit_Gap'] = comparison_df['Train_RMSE_log'] - comparison_df['Test_RMSE_log']
    comparison_df['Overfit_Score'] = np.abs(comparison_df['Overfit_Gap'])
    print("\nModels ranked by generalization (lowest overfit gap):")
    overfit_df = comparison_df.sort_values('Overfit_Score')[['Model', 'Train_RMSE_log', 'Test_RMSE_log', 'Overfit_Gap']]
    print(overfit_df.to_string(index=False))
    
    # Cross-validation on best model (if sklearn-compatible)
    print("\n" + "=" * 70)
    print("14. Cross-validation on best model...")
    try:
        cv_results = cross_validate_model(best_model, X, y, cv=5)
        print(f"   CV RMSE: {cv_results['cv_rmse_mean']:.4f} +/- {cv_results['cv_rmse_std']:.4f}")
    except Exception as e:
        print(f"   Cross-validation not available for this model type: {e}")
        cv_results = {'cv_rmse_mean': float('nan'), 'cv_rmse_std': float('nan')}
    
    # Feature importance analysis (for tree-based models)
    print("\n15. Feature importance analysis...")
    importance_df = analyze_feature_importance(best_model, feature_names)
    if importance_df is not None:
        print("\n   Top 15 important features:")
        print(importance_df.head(15).to_string(index=False))
    
    # Generate plots
    print("\n" + "=" * 70)
    print("16. Generating plots...")
    
    # Predictions plot for best model
    if hasattr(best_model, 'predict'):
        y_pred_final = best_model.predict(X_test)
    elif isinstance(best_model, list):
        # Ensemble model
        y_pred_final = np.array([m.predict(X_test) for m in best_model]).mean(axis=0)
    else:
        y_pred_final = all_results[best_model_name]['model'].predict(X_test)
    
    plot_predictions(y_test, y_pred_final, 
                     title=f"{best_model_name}: Predicted vs Actual Kd",
                     save_path="plots/predictions_best.png")
    
    if importance_df is not None:
        plot_feature_importance(importance_df, 
                                save_path="plots/feature_importance.png")
    
    plot_residuals(y_test, y_pred_final, 
                   save_path="plots/residuals.png")
    
    # Model comparison plot
    plot_model_comparison(comparison_df, save_path="plots/model_comparison.png")
    
    # Plot overfitting analysis
    plot_overfitting_analysis(comparison_df, save_path="plots/overfitting_analysis.png")
    
    # Save the best model
    print("\n17. Saving models...")
    
    # Prepare metadata
    best_metrics = all_results[best_model_name]
    metadata = {
        'model_type': best_model_name,
        'features': feature_names,
        'test_rmse_log': float(best_metrics['test_rmse_log']),
        'test_r2': float(best_metrics['test_r2']),
        'test_mae_mM': float(best_metrics['test_mae_mM']),
        'cv_rmse_mean': float(cv_results['cv_rmse_mean']) if not np.isnan(cv_results['cv_rmse_mean']) else None,
        'cv_rmse_std': float(cv_results['cv_rmse_std']) if not np.isnan(cv_results['cv_rmse_std']) else None,
        'training_samples': len(X_train_full),
        'test_samples': len(X_test),
        'data_balancing': 'hybrid',
        'feature_selection': 'elastic_net',
        'augmentation': 'mixup',
        'available_libraries': {
            'xgboost': HAS_XGBOOST,
            'lightgbm': HAS_LIGHTGBM,
            'catboost': HAS_CATBOOST,
            'pytorch': HAS_TORCH,
            'optuna': HAS_OPTUNA,
        },
        'note': 'PROOF-OF-CONCEPT model trained on SYNTHETIC data'
    }
    
    save_model(best_model, "models/patchkd_best_model.pkl", metadata)
    
    # Save all model results summary
    summary = {
        'model_comparison': comparison_df.to_dict(orient='records'),
        'best_model': best_model_name,
        'best_model_metrics': {
            'test_rmse_log': float(best_metrics['test_rmse_log']),
            'test_r2': float(best_metrics['test_r2']),
            'test_mae_mM': float(best_metrics['test_mae_mM']),
        },
        'regularization_methods': [
            'L1/L2 regularization (ElasticNet feature selection)',
            'Dropout (adaptive for deep layers)',
            'Spectral normalization',
            'Early stopping',
            'Weight decay (AdamW)',
            'Gradient clipping',
        ],
        'data_balancing_methods': [
            'Kd bin stratification',
            'MixUp augmentation',
            'Bootstrap sampling for ensembles',
        ],
        'uncertainty_quantification': [
            'Deep ensemble (5 models)',
            'Monte Carlo dropout',
            'Cross-validation averaging',
        ],
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'note': 'SYNTHETIC DATA - For methodological development only'
    }
    
    # Add Optuna best params if available
    if 'XGBoost_Optimized' in all_results and 'best_params' in all_results['XGBoost_Optimized']:
        summary['optuna_best_params'] = {k: v for k, v in all_results['XGBoost_Optimized']['best_params'].items() 
                                         if k not in ['random_state', 'verbosity']}
    
    with open("models/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save comparison table
    comparison_df.to_csv("models/model_comparison.csv", index=False)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest model ({best_model_name}) saved to: models/patchkd_best_model.pkl")
    print(f"Summary saved to: models/training_summary.json")
    print(f"Comparison saved to: models/model_comparison.csv")
    print(f"Plots saved to: plots/")
    print("\n" + "-" * 70)
    print("REGULARIZATION METHODS APPLIED:")
    print("-" * 70)
    print("  1. ElasticNet feature selection (L1+L2)")
    print("  2. Adaptive dropout (deeper layers = higher dropout)")
    print("  3. Spectral normalization (Lipschitz constraint)")
    print("  4. Early stopping with patience")
    print("  5. Weight decay (L2 via AdamW)")
    print("  6. Gradient clipping (max norm = 1.0)")
    print("  7. MixUp data augmentation")
    print("  8. Kd bin stratification for balanced training")
    print("\nIMPORTANT: This model was trained on SYNTHETIC data.")
    print("Real experimental validation is required before application.")
    print("=" * 70)


def plot_overfitting_analysis(comparison_df: pd.DataFrame, save_path: str = None):
    """
    Plot overfitting analysis showing train vs test performance.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = comparison_df['Model'].values
    train_rmse = comparison_df['Train_RMSE_log'].values
    test_rmse = comparison_df['Test_RMSE_log'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_rmse, width, label='Train RMSE', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_rmse, width, label='Test RMSE', color='coral', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE (log10 Kd)')
    ax.set_title('Overfitting Analysis: Train vs Test RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # Add gap annotation
    for i, (tr, te) in enumerate(zip(train_rmse, test_rmse)):
        gap = te - tr
        color = 'green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red'
        ax.annotate(f'{gap:.3f}', xy=(i, max(tr, te) + 0.01), 
                    ha='center', fontsize=8, color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    """
    Plot model comparison bar chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by Test RMSE
    df_sorted = comparison_df.sort_values('Test_RMSE_log')
    
    # RMSE comparison
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))
    bars1 = axes[0].barh(range(len(df_sorted)), df_sorted['Test_RMSE_log'].values, color=colors)
    axes[0].set_yticks(range(len(df_sorted)))
    axes[0].set_yticklabels(df_sorted['Model'].values)
    axes[0].set_xlabel('Test RMSE (log10 Kd)')
    axes[0].set_title('Model Comparison: Test RMSE')
    axes[0].invert_yaxis()
    
    # R2 comparison
    df_sorted_r2 = comparison_df.sort_values('Test_R2', ascending=False)
    bars2 = axes[1].barh(range(len(df_sorted_r2)), df_sorted_r2['Test_R2'].values, color=colors)
    axes[1].set_yticks(range(len(df_sorted_r2)))
    axes[1].set_yticklabels(df_sorted_r2['Model'].values)
    axes[1].set_xlabel('Test R2')
    axes[1].set_title('Model Comparison: Test R2')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()

