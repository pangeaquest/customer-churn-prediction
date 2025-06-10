"""
Evaluation metrics for customer churn prediction models.

This module provides comprehensive metrics calculation and analysis
for binary classification models used in churn prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)


class ChurnMetrics:
    """
    Comprehensive metrics calculation for churn prediction models.
    
    This class provides methods to calculate various performance metrics
    specifically relevant for customer churn prediction tasks.
    """
    
    def __init__(self):
        """Initialize the ChurnMetrics calculator."""
        self.metrics_history = []
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Dictionary containing basic metrics
        """
        try:
            # TODO: Implement basic metrics calculation
            # Calculate accuracy, precision, recall, F1-score
            # Handle edge cases and class imbalance
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': self._calculate_specificity(y_true, y_pred),
                'npv': self._calculate_npv(y_true, y_pred)  # Negative Predictive Value
            }
            
            logger.info("Basic metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            raise
    
    def calculate_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics that require probability predictions.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary containing probability-based metrics
        """
        try:
            # TODO: Implement probability-based metrics
            # Calculate ROC-AUC, PR-AUC, log loss
            # Handle probability calibration metrics
            
            metrics = {
                'roc_auc': roc_auc_score(y_true, y_proba),
                'pr_auc': self._calculate_pr_auc(y_true, y_proba),
                'log_loss': self._calculate_log_loss(y_true, y_proba),
                'brier_score': self._calculate_brier_score(y_true, y_proba)
            }
            
            logger.info("Probability metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating probability metrics: {str(e)}")
            raise
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 customer_values: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate business-relevant metrics for churn prediction.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            customer_values: Optional array of customer lifetime values
            
        Returns:
            Dictionary containing business metrics
        """
        try:
            # TODO: Implement business metrics calculation
            # Calculate cost-sensitive metrics
            # Include customer value considerations
            # Calculate retention campaign effectiveness
            
            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Basic business metrics
            metrics = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'churn_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
            }
            
            # Customer value-based metrics
            if customer_values is not None:
                metrics.update(self._calculate_value_based_metrics(
                    y_true, y_pred, customer_values
                ))
            
            logger.info("Business metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating business metrics: {str(e)}")
            raise
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_proba: np.ndarray = None,
                                      customer_values: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive set of metrics for model evaluation.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Optional predicted probabilities
            customer_values: Optional customer lifetime values
            
        Returns:
            Dictionary containing all calculated metrics
        """
        try:
            # Basic metrics
            basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
            
            # Probability metrics (if available)
            prob_metrics = {}
            if y_proba is not None:
                prob_metrics = self.calculate_probability_metrics(y_true, y_proba)
            
            # Business metrics
            business_metrics = self.calculate_business_metrics(y_true, y_pred, customer_values)
            
            # Combine all metrics
            all_metrics = {
                'basic_metrics': basic_metrics,
                'probability_metrics': prob_metrics,
                'business_metrics': business_metrics,
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            # Store in history
            self.metrics_history.append(all_metrics)
            
            logger.info("Comprehensive metrics calculated successfully")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {str(e)}")
            raise
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate negative predictive value."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def _calculate_pr_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Precision-Recall AUC."""
        from sklearn.metrics import auc
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return auc(recall, precision)
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate log loss."""
        from sklearn.metrics import log_loss
        return log_loss(y_true, y_proba)
    
    def _calculate_brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Brier score."""
        from sklearn.metrics import brier_score_loss
        return brier_score_loss(y_true, y_proba)
    
    def _calculate_value_based_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     customer_values: np.ndarray) -> Dict[str, float]:
        """Calculate customer value-based metrics."""
        # TODO: Implement value-based metrics
        # Calculate total value at risk
        # Calculate value of correctly identified churners
        # Calculate cost of false positives
        
        metrics = {
            'total_customer_value': float(np.sum(customer_values)),
            'churn_value_at_risk': float(np.sum(customer_values[y_true == 1])),
            'correctly_identified_value': float(np.sum(customer_values[(y_true == 1) & (y_pred == 1)])),
            'false_positive_value': float(np.sum(customer_values[(y_true == 0) & (y_pred == 1)]))
        }
        
        # Calculate value-based precision and recall
        if metrics['churn_value_at_risk'] > 0:
            metrics['value_based_recall'] = metrics['correctly_identified_value'] / metrics['churn_value_at_risk']
        else:
            metrics['value_based_recall'] = 0.0
        
        total_predicted_value = metrics['correctly_identified_value'] + metrics['false_positive_value']
        if total_predicted_value > 0:
            metrics['value_based_precision'] = metrics['correctly_identified_value'] / total_predicted_value
        else:
            metrics['value_based_precision'] = 0.0
        
        return metrics
    
    def get_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                            metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal threshold for binary classification.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        try:
            # TODO: Implement threshold optimization
            # Test different thresholds
            # Calculate metric for each threshold
            # Return optimal threshold and score
            
            thresholds = np.arange(0.1, 1.0, 0.01)
            best_score = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred_thresh = (y_proba >= threshold).astype(int)
                
                if metric == 'f1':
                    score = f1_score(y_true, y_pred_thresh, zero_division=0)
                elif metric == 'precision':
                    score = precision_score(y_true, y_pred_thresh, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_true, y_pred_thresh, zero_division=0)
                elif metric == 'accuracy':
                    score = accuracy_score(y_true, y_pred_thresh)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            logger.info(f"Optimal threshold found: {best_threshold:.3f} (score: {best_score:.3f})")
            return best_threshold, best_score
            
        except Exception as e:
            logger.error(f"Error finding optimal threshold: {str(e)}")
            raise
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics.
        
        Args:
            model_results: Dictionary with model names as keys and metrics as values
            
        Returns:
            DataFrame with model comparison
        """
        try:
            # TODO: Implement model comparison
            # Extract key metrics from each model
            # Create comparison DataFrame
            # Rank models by performance
            
            comparison_data = []
            
            for model_name, results in model_results.items():
                basic_metrics = results.get('basic_metrics', {})
                prob_metrics = results.get('probability_metrics', {})
                
                model_data = {
                    'Model': model_name,
                    'Accuracy': basic_metrics.get('accuracy', 0),
                    'Precision': basic_metrics.get('precision', 0),
                    'Recall': basic_metrics.get('recall', 0),
                    'F1-Score': basic_metrics.get('f1_score', 0),
                    'ROC-AUC': prob_metrics.get('roc_auc', 0)
                }
                comparison_data.append(model_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Sort by F1-score (or another primary metric)
            comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
            
            logger.info("Model comparison completed successfully")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def generate_metrics_report(self, metrics: Dict[str, Any], model_name: str = "") -> str:
        """
        Generate a formatted metrics report.
        
        Args:
            metrics: Calculated metrics dictionary
            model_name: Optional model name for the report
            
        Returns:
            Formatted metrics report string
        """
        try:
            report_lines = []
            
            if model_name:
                report_lines.append(f"METRICS REPORT - {model_name}")
            else:
                report_lines.append("METRICS REPORT")
            
            report_lines.append("=" * 50)
            
            # Basic metrics
            if 'basic_metrics' in metrics:
                report_lines.append("\nBASIC METRICS:")
                for metric, value in metrics['basic_metrics'].items():
                    report_lines.append(f"  {metric.capitalize()}: {value:.4f}")
            
            # Probability metrics
            if 'probability_metrics' in metrics and metrics['probability_metrics']:
                report_lines.append("\nPROBABILITY METRICS:")
                for metric, value in metrics['probability_metrics'].items():
                    report_lines.append(f"  {metric.upper()}: {value:.4f}")
            
            # Business metrics
            if 'business_metrics' in metrics:
                report_lines.append("\nBUSINESS METRICS:")
                for metric, value in metrics['business_metrics'].items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            
            report_lines.append("\n" + "=" * 50)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating metrics report: {str(e)}")
            return f"Error generating report: {str(e)}"
