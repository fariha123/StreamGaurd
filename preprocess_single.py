# ============================================
# PART 1: ENVIRONMENT SETUP AND IMPORTS (FIXED)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve,matthews_corrcoef
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

    # Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

    # Limit memory growth for GPU

print("✅ Environment setup complete")
print(f"TensorFlow version: {tf.__version__}")

class FixedDataProcessor:
    """
    Fixed data processing with proper class balance handling
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_combine_datasets(self, file_paths):
        """Load and combine multiple parquet files with memory optimization"""
        print("📂 Loading datasets...")
        data_frames = []
        
        for file_path in file_paths:
            print(f"Loading: {file_path.split('/')[-1]}")
            df = pd.read_parquet(file_path)
            
            # Extract attack type from filename for labeling
            attack_type = file_path.split('/')[-1].split('-')[0]
            if attack_type == 'Benign':
                df['Label'] = 0  # Benign
                df['Attack_Type'] = 'Benign'
            else:
                df['Label'] = 1  # Attack
                df['Attack_Type'] = attack_type
                
            data_frames.append(df)
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # Print detailed class distribution
        print(f"\n✅ Total samples loaded: {len(combined_df):,}")
        print("\n📊 Class Distribution:")
        print(combined_df['Label'].value_counts())
        print(f"\nBenign (0): {len(combined_df[combined_df['Label']==0]):,} samples")
        print(f"Attack (1): {len(combined_df[combined_df['Label']==1]):,} samples")
        
        # Calculate ratio
        attack_ratio = len(combined_df[combined_df['Label']==1]) / len(combined_df[combined_df['Label']==0])
        print(f"\n📐 Attack/Benign Ratio: {attack_ratio:.2f}:1")
        
        return combined_df
    
    def prepare_features(self, df):
        """
        Prepare features with proper preprocessing
        """
        # Separate features and labels
        if 'Label' in df.columns:
            y = df['Label'].values
            attack_types = df['Attack_Type'].values if 'Attack_Type' in df.columns else None
            X = df.drop(['Label', 'Attack_Type'], axis=1, errors='ignore')
        else:
            y = None
            attack_types = None
            X = df
        
        # Select numerical columns only
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numerical_cols]
        
        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Scale features
        print("📊 Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, attack_types

# File paths
file_paths = [
    '/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/archive/Benign-Monday-no-metadata.parquet',
    '/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/archive/Botnet-Friday-no-metadata.parquet',
    '/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/archive/Bruteforce-Tuesday-no-metadata.parquet',
    '/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/archive/DDoS-Friday-no-metadata.parquet',
    '/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/archive/DoS-Wednesday-no-metadata.parquet',
    '/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/archive/Infiltration-Thursday-no-metadata.parquet',
    '/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/archive/Portscan-Friday-no-metadata.parquet',
    '/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/archive/WebAttacks-Thursday-no-metadata.parquet'
]

# Initialize processor and load data
processor = FixedDataProcessor()
#df = processor.load_and_combine_datasets(file_paths)
df = processor.load_and_combine_datasets(file_paths)
df.to_csv(r"/Users/farihatahosinboishakhi/Documents/Uconn_Courses/Spring26/CSE5300/Safegaurd/data/cleaned/combined.csv")
X, y, attack_types = processor.prepare_features(df)

print(f"\n✅ Final feature matrix shape: {X.shape}")
print(f"✅ Labels shape: {y.shape}")
print(f"✅ Class distribution: {np.bincount(y.astype(int))}")



