"""
ML Model Trainer for Membership Churn Prediction
Trains a machine learning model using the membership churn dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
from datetime import datetime


class ChurnModelTrainer:
    """Train and evaluate churn prediction models"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.model_metrics = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the training data"""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Drop rows with missing values
        df_clean = df.dropna()
        if len(df_clean) < len(df):
            print(f"\nDropped {len(df) - len(df_clean)} rows with missing values")
            print(f"Remaining records: {len(df_clean)}")
        
        print(f"\nChurn distribution:\n{df_clean['Churn'].value_counts()}")
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing data"""
        df = df.copy()
        
        # Tenure categories
        df['Tenure_Category'] = pd.cut(df['Tenure'], 
                                       bins=[0, 12, 24, 36, 48, 100],
                                       labels=['0-12', '13-24', '25-36', '37-48', '49+'])
        
        # Usage frequency categories
        df['Usage_Category'] = pd.cut(df['Usage Frequency'],
                                      bins=[0, 10, 20, 30],
                                      labels=['Low', 'Medium', 'High'])
        
        # Payment delay severity
        df['Payment_Delay_Severe'] = (df['Payment Delay'] > 15).astype(int)
        
        # Spend per month
        df['Spend_Per_Month'] = df['Total Spend'] / (df['Tenure'] + 1)
        
        # Support call intensity
        df['Support_Call_Rate'] = df['Support Calls'] / (df['Tenure'] + 1)
        
        # Days since last interaction (assuming Last Interaction is days)
        df['Interaction_Recency'] = df['Last Interaction']
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'],
                                bins=[0, 25, 35, 45, 55, 100],
                                labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess data for model training"""
        # Drop CustomerID as it's not a feature
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        # Separate features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\nCategorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and select the best one"""
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                    random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                           random_state=42)
        }
        
        best_model = None
        best_score = 0
        best_model_name = None
        
        print("\n" + "="*60)
        print("Training and Evaluating Models")
        print("="*60)
        
        for name, model in models.items():
            print(f"\n{name}:")
            print("-" * 40)
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Store metrics
            self.model_metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'cv_roc_auc': cv_scores.mean()
            }
            
            # Track best model by ROC-AUC
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_model_name = name
        
        print("\n" + "="*60)
        print(f"Best Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        print("="*60)
        
        self.model = best_model
        
        # Print confusion matrix for best model
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix ({best_model_name}):")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return best_model, best_model_name
    
    def save_model(self, model_dir: str = 'models'):
        """Save the trained model and preprocessing objects"""
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = os.path.join(model_dir, 'churn_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save label encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)
        print(f"Label encoders saved to: {encoders_path}")
        
        # Save feature columns
        features_path = os.path.join(model_dir, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, features_path)
        print(f"Feature columns saved to: {features_path}")
        
        # Save metrics
        metrics_path = os.path.join(model_dir, 'model_metrics.pkl')
        joblib.dump(self.model_metrics, metrics_path)
        print(f"Metrics saved to: {metrics_path}")
    
    def train_pipeline(self):
        """Complete training pipeline"""
        # Load data
        df = self.load_and_prepare_data()
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Preprocess
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train models
        best_model, best_model_name = self.train_models(X_train, X_test, y_train, y_test)
        
        # Save everything
        self.save_model()
        
        return best_model, self.model_metrics


if __name__ == '__main__':
    # Path to training data
    data_path = 'archive membership from Kaggle/customer_churn_dataset-training-master.csv'
    
    print("="*60)
    print("Membership Churn Prediction - Model Training")
    print("="*60)
    
    # Initialize trainer
    trainer = ChurnModelTrainer(data_path)
    
    # Train
    model, metrics = trainer.train_pipeline()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nModel and artifacts saved to 'models/' directory")
    print("Ready for deployment!")
