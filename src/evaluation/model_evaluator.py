"""
Model evaluation utilities for customer churn prediction.

This module provides comprehensive model evaluation functionality
including cross-validation, performance analysis, and comparison.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from typing import Dict, List, Any, Optional, Tuple
import logging

from .metrics import ChurnMetrics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for churn prediction models.
    
    This class provides methods for evaluating model performance,
    conducting cross-validation, and comparing multiple models.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.metrics_calculator = ChurnMetrics()
        self.evaluation_history = []
    
    def evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                            model_name: str = "", customer_values: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            customer_values: Optional customer lifetime values
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # TODO: Implement single model evaluation
            # Make predictions
            # Calculate comprehensive metrics
            # Generate evaluation report
            
            logger.info(f"Evaluating model: {model_name}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                # For SVM, convert decision function to probabilities
                decision_scores = model.decision_function(X_test)
                y_proba = self._sigmoid(decision_scores)
            
            # Calculate comprehensive metrics
            metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                y_test.values, y_pred, y_proba, customer_values
            )
            
            # Add model-specific information
            evaluation_result = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'test_size': len(X_test),
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            # Store in history
            self.evaluation_history.append(evaluation_result)
            
            logger.info(f"Model evaluation completed for {model_name}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5, scoring: str = 'accuracy',
                           model_name: str = "") -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Target
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            model_name: Name of the model
            
        Returns:
            Dictionary containing cross-validation results
        """
        try:
            # TODO: Implement cross-validation
            # Use stratified k-fold for balanced splits
            # Calculate multiple metrics across folds
            # Return comprehensive CV results
            
            logger.info(f"Cross-validating model: {model_name}")
            
            # Use stratified k-fold to maintain class balance
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Define multiple scoring metrics
            scoring_metrics = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
            
            cv_results = {}
            
            for metric_name, metric_scorer in scoring_metrics.items():
                try:
                    scores = cross_val_score(
                        model, X, y, cv=cv, scoring=metric_scorer, n_jobs=-1
                    )
                    
                    cv_results[metric_name] = {
                        'scores': scores.tolist(),
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'min': float(scores.min()),
                        'max': float(scores.max())
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not calculate {metric_name} in CV: {str(e)}")
                    cv_results[metric_name] = None
            
            # Overall CV summary
            cv_summary = {
                'model_name': model_name,
                'cv_folds': cv_folds,
                'cv_results': cv_results,
                'primary_metric': scoring,
                'primary_score_mean': cv_results.get(scoring, {}).get('mean', 0),
                'primary_score_std': cv_results.get(scoring, {}).get('std', 0)
            }
            
            logger.info(f"Cross-validation completed for {model_name}")
            return cv_summary
            
        except Exception as e:
            logger.error(f"Error in cross-validation for {model_name}: {str(e)}")
            raise
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models based on their evaluation results.
        
        Args:
            model_results: Dictionary with model names as keys and evaluation results as values
            
        Returns:
            Dictionary containing model comparison results
        """
        try:
            # TODO: Implement model comparison
            # Extract key metrics from each model
            # Rank models by different criteria
            # Generate comparison visualizations
            
            logger.info("Comparing models...")
            
            # Extract metrics for comparison
            comparison_data = []
            
            for model_name, results in model_results.items():
                metrics = results.get('metrics', {})
                basic_metrics = metrics.get('basic_metrics', {})
                prob_metrics = metrics.get('probability_metrics', {})
                business_metrics = metrics.get('business_metrics', {})
                
                model_summary = {
                    'model_name': model_name,
                    'accuracy': basic_metrics.get('accuracy', 0),
                    'precision': basic_metrics.get('precision', 0),
                    'recall': basic_metrics.get('recall', 0),
                    'f1_score': basic_metrics.get('f1_score', 0),
                    'roc_auc': prob_metrics.get('roc_auc', 0),
                    'true_positives': business_metrics.get('true_positives', 0),
                    'false_positives': business_metrics.get('false_positives', 0),
                    'false_negatives': business_metrics.get('false_negatives', 0)
                }
                
                comparison_data.append(model_summary)
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            # Rank models by different metrics
            rankings = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                if metric in comparison_df.columns:
                    rankings[metric] = comparison_df.nlargest(len(comparison_df), metric)['model_name'].tolist()
            
            # Find overall best model (based on F1-score)
            best_model = comparison_df.loc[comparison_df['f1_score'].idxmax(), 'model_name']
            
            # Calculate model stability (if CV results available)
            stability_analysis = self._analyze_model_stability(model_results)
            
            comparison_results = {
                'comparison_table': comparison_df,
                'rankings': rankings,
                'best_model': best_model,
                'stability_analysis': stability_analysis,
                'summary_stats': self._calculate_comparison_stats(comparison_df)
            }
            
            logger.info("Model comparison completed")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def analyze_model_performance(self, model_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze detailed performance of a single model.
        
        Args:
            model_result: Model evaluation result
            
        Returns:
            Dictionary containing detailed performance analysis
        """
        try:
            # TODO: Implement detailed performance analysis
            # Analyze prediction patterns
            # Identify model strengths and weaknesses
            # Generate actionable insights
            
            logger.info("Analyzing model performance...")
            
            metrics = model_result.get('metrics', {})
            model_name = model_result.get('model_name', 'Unknown')
            
            # Extract key performance indicators
            basic_metrics = metrics.get('basic_metrics', {})
            business_metrics = metrics.get('business_metrics', {})
            
            # Performance analysis
            analysis = {
                'model_name': model_name,
                'overall_performance': self._assess_overall_performance(basic_metrics),
                'class_balance_handling': self._assess_class_balance_handling(business_metrics),
                'business_impact': self._assess_business_impact(business_metrics),
                'recommendations': self._generate_recommendations(metrics),
                'strengths': self._identify_strengths(metrics),
                'weaknesses': self._identify_weaknesses(metrics)
            }
            
            logger.info("Performance analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {str(e)}")
            raise
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function to convert decision scores to probabilities."""
        return 1 / (1 + np.exp(-x))
    
    def _analyze_model_stability(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stability of models based on cross-validation results."""
        stability_analysis = {}
        
        for model_name, results in model_results.items():
            cv_results = results.get('cv_results', {})
            
            if cv_results:
                # Calculate coefficient of variation for key metrics
                stability_scores = {}
                
                for metric, scores in cv_results.items():
                    if isinstance(scores, dict) and 'mean' in scores and 'std' in scores:
                        mean_score = scores['mean']
                        std_score = scores['std']
                        
                        # Coefficient of variation (lower is more stable)
                        cv_score = std_score / mean_score if mean_score > 0 else float('inf')
                        stability_scores[metric] = cv_score
                
                stability_analysis[model_name] = {
                    'stability_scores': stability_scores,
                    'overall_stability': np.mean(list(stability_scores.values())) if stability_scores else 0
                }
        
        return stability_analysis
    
    def _calculate_comparison_stats(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for model comparison."""
        numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_columns:
            if col != 'model_name':
                stats[col] = {
                    'mean': float(comparison_df[col].mean()),
                    'std': float(comparison_df[col].std()),
                    'min': float(comparison_df[col].min()),
                    'max': float(comparison_df[col].max()),
                    'range': float(comparison_df[col].max() - comparison_df[col].min())
                }
        
        return stats
    
    def _assess_overall_performance(self, basic_metrics: Dict[str, float]) -> str:
        """Assess overall model performance level."""
        f1_score = basic_metrics.get('f1_score', 0)
        
        if f1_score >= 0.8:
            return "Excellent"
        elif f1_score >= 0.7:
            return "Good"
        elif f1_score >= 0.6:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_class_balance_handling(self, business_metrics: Dict[str, Any]) -> str:
        """Assess how well the model handles class imbalance."""
        tp = business_metrics.get('true_positives', 0)
        fp = business_metrics.get('false_positives', 0)
        fn = business_metrics.get('false_negatives', 0)
        
        if tp == 0:
            return "Poor - No positive predictions"
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision >= 0.7 and recall >= 0.7:
            return "Good - Balanced performance"
        elif precision >= 0.7:
            return "Conservative - High precision, lower recall"
        elif recall >= 0.7:
            return "Aggressive - High recall, lower precision"
        else:
            return "Poor - Low precision and recall"
    
    def _assess_business_impact(self, business_metrics: Dict[str, Any]) -> str:
        """Assess potential business impact of the model."""
        churn_detection_rate = business_metrics.get('churn_detection_rate', 0)
        false_alarm_rate = business_metrics.get('false_alarm_rate', 0)
        
        if churn_detection_rate >= 0.7 and false_alarm_rate <= 0.2:
            return "High - Good churn detection with low false alarms"
        elif churn_detection_rate >= 0.6:
            return "Medium - Moderate churn detection"
        else:
            return "Low - Limited churn detection capability"
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        basic_metrics = metrics.get('basic_metrics', {})
        business_metrics = metrics.get('business_metrics', {})
        
        # Precision-based recommendations
        precision = basic_metrics.get('precision', 0)
        if precision < 0.6:
            recommendations.append("Consider adjusting classification threshold to improve precision")
        
        # Recall-based recommendations
        recall = basic_metrics.get('recall', 0)
        if recall < 0.6:
            recommendations.append("Consider feature engineering or ensemble methods to improve recall")
        
        # False alarm recommendations
        false_alarm_rate = business_metrics.get('false_alarm_rate', 0)
        if false_alarm_rate > 0.3:
            recommendations.append("High false alarm rate - consider cost-sensitive learning")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Model performance is satisfactory - consider deployment")
        
        return recommendations
    
    def _identify_strengths(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify model strengths based on metrics."""
        strengths = []
        
        basic_metrics = metrics.get('basic_metrics', {})
        
        if basic_metrics.get('accuracy', 0) >= 0.8:
            strengths.append("High overall accuracy")
        
        if basic_metrics.get('precision', 0) >= 0.7:
            strengths.append("Good precision - low false positive rate")
        
        if basic_metrics.get('recall', 0) >= 0.7:
            strengths.append("Good recall - captures most churners")
        
        if basic_metrics.get('f1_score', 0) >= 0.7:
            strengths.append("Balanced precision and recall")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify model weaknesses based on metrics."""
        weaknesses = []
        
        basic_metrics = metrics.get('basic_metrics', {})
        business_metrics = metrics.get('business_metrics', {})
        
        if basic_metrics.get('precision', 0) < 0.6:
            weaknesses.append("Low precision - high false positive rate")
        
        if basic_metrics.get('recall', 0) < 0.6:
            weaknesses.append("Low recall - missing many churners")
        
        if business_metrics.get('false_alarm_rate', 0) > 0.3:
            weaknesses.append("High false alarm rate")
        
        if basic_metrics.get('accuracy', 0) < 0.7:
            weaknesses.append("Low overall accuracy")
        
        return weaknesses
