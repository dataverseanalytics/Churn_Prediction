"""
ML Model for Churn Prediction
Loads trained model and provides predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Any


class ChurnPredictor:
    """Load trained model and make churn predictions"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            model_path = os.path.join(self.model_dir, 'churn_model.pkl')
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
            
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded")
            
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            self.label_encoders = joblib.load(encoders_path)
            print(f"✓ Label encoders loaded")
            
            features_path = os.path.join(self.model_dir, 'feature_columns.pkl')
            self.feature_columns = joblib.load(features_path)
            print(f"✓ Feature columns loaded")
            
            print(f"✓ Model ready for predictions!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please train the model first by running: python model_trainer.py")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create same features as training"""
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
        
        # Days since last interaction
        df['Interaction_Recency'] = df['Last Interaction']
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'],
                                bins=[0, 25, 35, 45, 55, 100],
                                labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        
        return df
    
    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Drop CustomerID if present
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        # Drop Churn if present (target variable)
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the columns used in training
        df = df[self.feature_columns]
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=self.feature_columns)
        
        return df_scaled
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn probability for a single member
        
        Args:
            data: Dictionary with member attributes
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        X = self.preprocess_input(data)
        
        # Predict
        churn_proba = self.model.predict_proba(X)[0][1]  # Probability of churn
        churn_prediction = int(self.model.predict(X)[0])
        
        # Categorize risk
        if churn_proba <= 0.3:
            risk_category = 'low'
        elif churn_proba <= 0.6:
            risk_category = 'medium'
        else:
            risk_category = 'high'
        
        return {
            'churn_probability': round(float(churn_proba * 100), 2),  # Convert to percentage
            'churn_prediction': churn_prediction,
            'risk_category': risk_category,
            'will_churn': bool(churn_prediction == 1)
        }
    
    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict churn for multiple members
        
        Args:
            data_list: List of dictionaries with member attributes
        
        Returns:
            List of prediction results
        """
        results = []
        
        for data in data_list:
            try:
                prediction = self.predict(data)
                prediction['member_id'] = data.get('CustomerID', 'Unknown')
                results.append(prediction)
            except Exception as e:
                print(f"Error predicting for member {data.get('CustomerID', 'Unknown')}: {str(e)}")
                results.append({
                    'member_id': data.get('CustomerID', 'Unknown'),
                    'error': str(e),
                    'churn_probability': 0,
                    'risk_category': 'unknown'
                })
        
        return results
    
    def get_feature_importance(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most important features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = [
                {'feature': col, 'importance': float(imp)}
                for col, imp in zip(self.feature_columns, importances)
            ]
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            return feature_importance[:top_n]
        else:
            return []


# Example usage
if __name__ == '__main__':
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Example member data
    example_member = {
        'CustomerID': 'TEST001',
        'Age': 45,
        'Gender': 'Male',
        'Tenure': 24,
        'Usage Frequency': 15,
        'Support Calls': 3,
        'Payment Delay': 10,
        'Subscription Type': 'Premium',
        'Contract Length': 'Annual',
        'Total Spend': 1200,
        'Last Interaction': 5
    }
    
    # Predict
    result = predictor.predict(example_member)
    
    print("\n" + "="*60)
    print("Churn Prediction Example")
    print("="*60)
    print(f"Member ID: {example_member['CustomerID']}")
    print(f"Churn Probability: {result['churn_probability']}%")
    print(f"Risk Category: {result['risk_category'].upper()}")
    print(f"Will Churn: {'Yes' if result['will_churn'] else 'No'}")
    
    # Feature importance
    print("\n" + "="*60)
    print("Top 10 Most Important Features")
    print("="*60)
    importance = predictor.get_feature_importance(10)
    for i, feat in enumerate(importance, 1):
        print(f"{i}. {feat['feature']}: {feat['importance']:.4f}")
