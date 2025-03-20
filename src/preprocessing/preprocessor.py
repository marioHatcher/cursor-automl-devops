"""
Data preprocessing and feature engineering module.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

class LoanPreprocessor:
    """Preprocessor for loan application data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_stats = {}
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Create copy to avoid modifying original data
        processed_df = df.copy()
        
        # Calculate derived features
        processed_df = self._create_derived_features(processed_df)
        
        # Handle categorical variables
        processed_df = self._encode_categorical_features(processed_df)
        
        # Scale numerical features
        num_features = ['age', 'income', 'loan_amount', 'loan_term', 
                       'credit_score', 'existing_loans', 
                       'debt_to_income', 'loan_to_income']
        
        processed_df[num_features] = self.scaler.fit_transform(processed_df[num_features])
        
        # Store feature statistics for future reference
        self._store_feature_stats(processed_df)
        
        return processed_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        processed_df = df.copy()
        processed_df = self._create_derived_features(processed_df)
        processed_df = self._encode_categorical_features(processed_df)
        
        num_features = ['age', 'income', 'loan_amount', 'loan_term', 
                       'credit_score', 'existing_loans',
                       'debt_to_income', 'loan_to_income']
        
        processed_df[num_features] = self.scaler.transform(processed_df[num_features])
        return processed_df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from raw data."""
        df = df.copy()
        
        # Debt to income ratio
        df['debt_to_income'] = df['loan_amount'] / (df['income'] * df['loan_term']/12)
        
        # Loan to income ratio
        df['loan_to_income'] = df['loan_amount'] / df['income']
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        # Employment status encoding
        employment_map = {
            'Employed': 2,
            'Self-Employed': 1,
            'Unemployed': 0
        }
        df['employment_status'] = df['employment_status'].map(employment_map)
        
        # Loan purpose one-hot encoding
        df = pd.get_dummies(df, columns=['loan_purpose'], prefix='purpose')
        
        # Income variability binary encoding
        df['income_variability'] = (df['income_variability'] == 'High').astype(int)
        
        return df
    
    def _store_feature_stats(self, df: pd.DataFrame):
        """Store feature statistics for monitoring."""
        self.feature_stats = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
            'min': df.min().to_dict(),
            'max': df.max().to_dict()
        } 