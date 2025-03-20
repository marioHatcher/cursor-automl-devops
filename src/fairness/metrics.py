"""
Fairness metrics for evaluating model bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix

class FairnessMetrics:
    """Calculate and track fairness metrics for the loan approval model."""
    
    def __init__(self, protected_attribute: str = 'income_variability'):
        """
        Initialize fairness metrics calculator.
        
        Args:
            protected_attribute: Name of the protected attribute to evaluate fairness on
        """
        self.protected_attribute = protected_attribute
        self.metrics = {}
    
    def calculate_metrics(self, 
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        protected_values: np.ndarray) -> Dict:
        """
        Calculate fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_values: Values of the protected attribute
            
        Returns:
            Dictionary containing fairness metrics
        """
        # Split data by protected attribute
        mask_privileged = protected_values == 0  # Low income variability
        mask_unprivileged = protected_values == 1  # High income variability
        
        # Calculate metrics for each group
        priv_metrics = self._group_metrics(y_true[mask_privileged], 
                                         y_pred[mask_privileged])
        unpriv_metrics = self._group_metrics(y_true[mask_unprivileged], 
                                           y_pred[mask_unprivileged])
        
        # Calculate disparate impact
        disparate_impact = unpriv_metrics['approval_rate'] / priv_metrics['approval_rate']
        
        # Calculate equal opportunity difference
        equal_opp_diff = (unpriv_metrics['true_positive_rate'] - 
                         priv_metrics['true_positive_rate'])
        
        # Calculate demographic parity difference
        demo_parity_diff = (unpriv_metrics['approval_rate'] - 
                          priv_metrics['approval_rate'])
        
        self.metrics = {
            'disparate_impact': disparate_impact,
            'equal_opportunity_difference': equal_opp_diff,
            'demographic_parity_difference': demo_parity_diff,
            'privileged_group_metrics': priv_metrics,
            'unprivileged_group_metrics': unpriv_metrics
        }
        
        return self.metrics
    
    def _group_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate metrics for a specific group."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'approval_rate': (tp + fp) / len(y_true) if len(y_true) > 0 else 0,
            'accuracy': (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        }
        
        return metrics
    
    def get_fairness_report(self) -> str:
        """Generate a human-readable fairness report."""
        if not self.metrics:
            return "No metrics have been calculated yet."
        
        report = [
            "Fairness Metrics Report",
            "=====================",
            f"Protected Attribute: {self.protected_attribute}",
            "",
            "Overall Fairness Metrics:",
            f"- Disparate Impact: {self.metrics['disparate_impact']:.3f}",
            f"- Equal Opportunity Difference: {self.metrics['equal_opportunity_difference']:.3f}",
            f"- Demographic Parity Difference: {self.metrics['demographic_parity_difference']:.3f}",
            "",
            "Group-specific Metrics:",
            "",
            "Privileged Group (Low Income Variability):",
            f"- Approval Rate: {self.metrics['privileged_group_metrics']['approval_rate']:.3f}",
            f"- True Positive Rate: {self.metrics['privileged_group_metrics']['true_positive_rate']:.3f}",
            "",
            "Unprivileged Group (High Income Variability):",
            f"- Approval Rate: {self.metrics['unprivileged_group_metrics']['approval_rate']:.3f}",
            f"- True Positive Rate: {self.metrics['unprivileged_group_metrics']['true_positive_rate']:.3f}"
        ]
        
        return "\n".join(report)
    
    def is_fair(self, threshold: float = 0.2) -> Tuple[bool, str]:
        """
        Check if the model meets fairness criteria.
        
        Args:
            threshold: Maximum acceptable difference for fairness metrics
            
        Returns:
            Tuple of (is_fair, reason)
        """
        if not self.metrics:
            return False, "No metrics have been calculated yet."
        
        checks = [
            (abs(1 - self.metrics['disparate_impact']) <= threshold,
             "Disparate impact exceeds threshold"),
            (abs(self.metrics['equal_opportunity_difference']) <= threshold,
             "Equal opportunity difference exceeds threshold"),
            (abs(self.metrics['demographic_parity_difference']) <= threshold,
             "Demographic parity difference exceeds threshold")
        ]
        
        failed_checks = [(check[1]) for check in checks if not check[0]]
        
        if failed_checks:
            return False, "Fairness criteria not met: " + "; ".join(failed_checks)
        
        return True, "All fairness criteria met" 