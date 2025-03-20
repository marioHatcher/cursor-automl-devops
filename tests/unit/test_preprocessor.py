"""
Unit tests for the preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.preprocessor import LoanPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'age': [30, 40, 50],
        'income': [50000, 60000, 70000],
        'loan_amount': [20000, 30000, 40000],
        'loan_term': [12, 24, 36],
        'credit_score': [650, 700, 750],
        'employment_status': ['Employed', 'Self-Employed', 'Unemployed'],
        'loan_purpose': ['Car', 'Home', 'Business'],
        'existing_loans': [1, 2, 3],
        'income_variability': ['Low', 'High', 'Low'],
        'loan_status': [1, 0, 1]
    })

def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = LoanPreprocessor()
    assert preprocessor is not None
    assert hasattr(preprocessor, 'scaler')
    assert hasattr(preprocessor, 'feature_stats')

def test_derived_features(sample_data):
    """Test creation of derived features."""
    preprocessor = LoanPreprocessor()
    processed_data = preprocessor._create_derived_features(sample_data)
    
    # Check if new features are created
    assert 'debt_to_income' in processed_data.columns
    assert 'loan_to_income' in processed_data.columns
    
    # Check calculations
    expected_loan_to_income = sample_data['loan_amount'] / sample_data['income']
    pd.testing.assert_series_equal(
        processed_data['loan_to_income'],
        expected_loan_to_income,
        check_names=False
    )

def test_categorical_encoding(sample_data):
    """Test categorical feature encoding."""
    preprocessor = LoanPreprocessor()
    processed_data = preprocessor._encode_categorical_features(sample_data)
    
    # Check employment status encoding
    assert processed_data['employment_status'].dtype == np.int64
    assert set(processed_data['employment_status'].unique()) <= {0, 1, 2}
    
    # Check loan purpose one-hot encoding
    assert 'purpose_Car' in processed_data.columns
    assert 'purpose_Home' in processed_data.columns
    assert 'purpose_Business' in processed_data.columns
    
    # Check income variability encoding
    assert processed_data['income_variability'].dtype == np.int64
    assert set(processed_data['income_variability'].unique()) <= {0, 1}

def test_full_transformation(sample_data):
    """Test complete preprocessing pipeline."""
    preprocessor = LoanPreprocessor()
    processed_data = preprocessor.fit_transform(sample_data)
    
    # Check if all expected features are present
    expected_features = [
        'age', 'income', 'loan_amount', 'loan_term', 'credit_score',
        'existing_loans', 'employment_status', 'income_variability',
        'debt_to_income', 'loan_to_income', 'purpose_Car', 'purpose_Home',
        'purpose_Business', 'loan_status'
    ]
    
    assert all(feature in processed_data.columns for feature in expected_features)
    
    # Check if numerical features are scaled
    numerical_features = [
        'age', 'income', 'loan_amount', 'loan_term', 'credit_score',
        'existing_loans', 'debt_to_income', 'loan_to_income'
    ]
    
    for feature in numerical_features:
        assert abs(processed_data[feature].mean()) < 1e-10  # Close to 0
        assert abs(processed_data[feature].std() - 1.0) < 1e-10  # Close to 1

def test_transform_consistency(sample_data):
    """Test consistency between fit_transform and transform."""
    preprocessor = LoanPreprocessor()
    
    # First transformation
    first_transform = preprocessor.fit_transform(sample_data)
    
    # Second transformation
    second_transform = preprocessor.transform(sample_data)
    
    # Check if both transformations produce the same results
    pd.testing.assert_frame_equal(first_transform, second_transform)

def test_feature_stats(sample_data):
    """Test feature statistics calculation."""
    preprocessor = LoanPreprocessor()
    preprocessor.fit_transform(sample_data)
    
    assert 'mean' in preprocessor.feature_stats
    assert 'std' in preprocessor.feature_stats
    assert 'min' in preprocessor.feature_stats
    assert 'max' in preprocessor.feature_stats
    
    # Check if stats are calculated for all numerical features
    numerical_features = [
        'age', 'income', 'loan_amount', 'loan_term', 'credit_score',
        'existing_loans', 'debt_to_income', 'loan_to_income'
    ]
    
    for feature in numerical_features:
        assert feature in preprocessor.feature_stats['mean']
        assert feature in preprocessor.feature_stats['std'] 