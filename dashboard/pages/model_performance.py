"""
Model Performance page for the Customer Churn Prediction Dashboard.

This page displays detailed model evaluation metrics and comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_page(df: pd.DataFrame, models: dict):
    """
    Display the model performance page.
    
    Args:
        df: Customer churn dataset
        models: Dictionary of trained models
    """
    st.header("📈 Model Performance")
    st.markdown("Compare and analyze the performance of different machine learning models.")
    
    if not models:
        st.error("No trained models available for evaluation.")
        return
    
    # Model comparison overview
    show_model_comparison(models)
    
    # Detailed model analysis
    show_detailed_analysis(models)
    
    # Model interpretability
    show_model_interpretability(models)
    
    # Performance recommendations
    show_performance_recommendations(models)


def show_model_comparison(models: dict):
    """Show comparison of all models."""
    st.subheader("🏆 Model Comparison Overview")
    
    # Get trained models
    trained_models = {name: model for name, model in models.items() if model.is_fitted}
    
    if not trained_models:
        st.warning("No trained models available for comparison.")
        return
    
    # Create comparison data
    comparison_data = []
    
    for model_name, model in trained_models.items():
        # TODO: Get actual metrics from models
        # For now, use placeholder metrics
        metrics = {
            'Model': model_name,
            'Accuracy': np.random.uniform(0.75, 0.90),
            'Precision': np.random.uniform(0.70, 0.88),
            'Recall': np.random.uniform(0.65, 0.85),
            'F1-Score': np.random.uniform(0.70, 0.86),
            'ROC-AUC': np.random.uniform(0.80, 0.95)
        }
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display metrics table
    st.markdown("#### Performance Metrics")
    
    # Format the dataframe for better display
    formatted_df = comparison_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Best model highlight
    best_model_idx = comparison_df['Accuracy'].idxmax()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    best_accuracy = comparison_df.loc[best_model_idx, 'Accuracy']
    
    st.success(f"🏆 **Best Performing Model**: {best_model} (Accuracy: {best_accuracy:.3f})")
    
    # Metrics comparison charts
    show_metrics_comparison_charts(comparison_df)


def show_metrics_comparison_charts(comparison_df: pd.DataFrame):
    """Show comparison charts for different metrics."""
    st.markdown("#### Metrics Comparison")
    
    # Radar chart for all metrics
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for _, row in comparison_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Comparison (Radar Chart)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar charts for individual metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = px.bar(
            comparison_df,
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='Blues'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC-AUC comparison
        fig = px.bar(
            comparison_df,
            x='Model',
            y='ROC-AUC',
            title="Model ROC-AUC Comparison",
            color='ROC-AUC',
            color_continuous_scale='Greens'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


def show_detailed_analysis(models: dict):
    """Show detailed analysis for selected model."""
    st.subheader("🔍 Detailed Model Analysis")
    
    # Model selection
    trained_models = {name: model for name, model in models.items() if model.is_fitted}
    
    if not trained_models:
        st.warning("No trained models available for detailed analysis.")
        return
    
    selected_model_name = st.selectbox("Select model for detailed analysis:", list(trained_models.keys()))
    selected_model = trained_models[selected_model_name]
    
    # Model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Information")
        st.write(f"**Model Type**: {selected_model_name}")
        st.write(f"**Status**: {'Trained' if selected_model.is_fitted else 'Not Trained'}")
        
        # Hyperparameters
        if hasattr(selected_model, 'hyperparameters') and selected_model.hyperparameters:
            st.write("**Hyperparameters**:")
            for param, value in selected_model.hyperparameters.items():
                st.write(f"  • {param}: {value}")
    
    with col2:
        st.markdown("#### Performance Metrics")
        
        # TODO: Get actual metrics from model evaluation
        # For now, use placeholder metrics
        metrics = {
            'Accuracy': 0.872,
            'Precision': 0.846,
            'Recall': 0.789,
            'F1-Score': 0.816,
            'ROC-AUC': 0.923
        }
        
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.3f}")
    
    # Confusion Matrix
    show_confusion_matrix(selected_model_name)
    
    # ROC Curve
    show_roc_curve(selected_model_name)
    
    # Learning Curves (if available)
    show_learning_curves(selected_model, selected_model_name)


def show_confusion_matrix(model_name: str):
    """Show confusion matrix for the selected model."""
    st.markdown("#### Confusion Matrix")
    
    # TODO: Generate actual confusion matrix from model predictions
    # For now, create a placeholder confusion matrix
    
    # Placeholder confusion matrix data
    cm_data = np.array([[1200, 150], [200, 450]])
    
    # Create confusion matrix heatmap
    fig = px.imshow(
        cm_data,
        text_auto=True,
        aspect="auto",
        title=f"Confusion Matrix - {model_name}",
        labels=dict(x="Predicted", y="Actual"),
        x=['No Churn', 'Churn'],
        y=['No Churn', 'Churn'],
        color_continuous_scale='Blues'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm_data.ravel()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("True Negatives", tn)
    with col2:
        st.metric("False Positives", fp)
    with col3:
        st.metric("False Negatives", fn)
    with col4:
        st.metric("True Positives", tp)


def show_roc_curve(model_name: str):
    """Show ROC curve for the selected model."""
    st.markdown("#### ROC Curve")
    
    # TODO: Generate actual ROC curve from model predictions
    # For now, create a placeholder ROC curve
    
    # Placeholder ROC curve data
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Placeholder curve
    auc_score = 0.923
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_learning_curves(model, model_name: str):
    """Show learning curves if available."""
    st.markdown("#### Learning Curves")
    
    # Check if model has training history
    if hasattr(model, 'get_training_history'):
        training_history = model.get_training_history()
        
        if training_history and training_history.get('training_scores'):
            # Plot learning curves
            epochs = range(1, len(training_history['training_scores']) + 1)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=training_history['training_scores'],
                mode='lines',
                name='Training Score',
                line=dict(color='blue')
            ))
            
            if training_history.get('validation_scores'):
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=training_history['validation_scores'],
                    mode='lines',
                    name='Validation Score',
                    line=dict(color='red')
                ))
            
            fig.update_layout(
                title=f'Learning Curves - {model_name}',
                xaxis_title='Epoch',
                yaxis_title='Score',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Learning curves not available for this model.")
    else:
        st.info("Learning curves not available for this model type.")


def show_model_interpretability(models: dict):
    """Show model interpretability analysis."""
    st.subheader("🧠 Model Interpretability")
    
    trained_models = {name: model for name, model in models.items() if model.is_fitted}
    
    if not trained_models:
        st.warning("No trained models available for interpretability analysis.")
        return
    
    # Model selection for interpretability
    selected_model_name = st.selectbox(
        "Select model for interpretability analysis:",
        list(trained_models.keys()),
        key="interpretability_model"
    )
    selected_model = trained_models[selected_model_name]
    
    # Feature importance
    feature_importance = selected_model.get_feature_importance()
    
    if feature_importance:
        st.markdown("#### Feature Importance")
        
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False).head(10)
        
        # Feature importance chart
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top 10 Feature Importance - {selected_model_name}",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.dataframe(importance_df, use_container_width=True)
    
    else:
        st.info("Feature importance not available for this model.")
    
    # Model-specific interpretability
    if "Logistic Regression" in selected_model_name:
        st.markdown("#### Logistic Regression Interpretation")
        st.info("Logistic regression coefficients represent the change in log-odds for a one-unit change in the feature.")
        
        # TODO: Add odds ratios interpretation
        
    elif "Random Forest" in selected_model_name:
        st.markdown("#### Random Forest Interpretation")
        st.info("Feature importance is based on the average decrease in impurity across all trees in the forest.")
        
        # TODO: Add tree statistics
        
    elif "Gradient Boosting" in selected_model_name:
        st.markdown("#### Gradient Boosting Interpretation")
        st.info("Feature importance represents the total gain contributed by each feature across all boosting rounds.")


def show_performance_recommendations(models: dict):
    """Show recommendations for improving model performance."""
    st.subheader("🚀 Performance Improvement Recommendations")
    
    trained_models = {name: model for name, model in models.items() if model.is_fitted}
    
    if not trained_models:
        st.warning("No trained models available for recommendations.")
        return
    
    # General recommendations
    st.markdown("#### General Recommendations")
    
    recommendations = [
        {
            "category": "Data Quality",
            "recommendations": [
                "Collect more recent data to improve model relevance",
                "Address any data quality issues identified in EDA",
                "Consider feature engineering for better predictive power"
            ]
        },
        {
            "category": "Model Improvement",
            "recommendations": [
                "Experiment with ensemble methods for better performance",
                "Tune hyperparameters using advanced optimization techniques",
                "Consider deep learning models for complex patterns"
            ]
        },
        {
            "category": "Validation",
            "recommendations": [
                "Implement time-based validation for temporal data",
                "Use stratified sampling to maintain class balance",
                "Monitor model performance over time for drift detection"
            ]
        },
        {
            "category": "Deployment",
            "recommendations": [
                "Set up automated model retraining pipeline",
                "Implement A/B testing for model comparison",
                "Create monitoring dashboards for production performance"
            ]
        }
    ]
    
    for rec_group in recommendations:
        with st.expander(f"📋 {rec_group['category']}"):
            for rec in rec_group['recommendations']:
                st.write(f"• {rec}")
    
    # Model-specific recommendations
    st.markdown("#### Model-Specific Recommendations")
    
    # TODO: Analyze actual model performance and provide specific recommendations
    # For now, provide general model-specific advice
    
    for model_name in trained_models.keys():
        with st.expander(f"🎯 {model_name} Recommendations"):
            if "Logistic Regression" in model_name:
                st.write("• Consider polynomial features for non-linear relationships")
                st.write("• Regularization tuning may improve generalization")
                st.write("• Feature scaling is important for this model")
            
            elif "Random Forest" in model_name:
                st.write("• Tune the number of trees and max depth")
                st.write("• Consider feature selection to reduce overfitting")
                st.write("• Monitor out-of-bag error for validation")
            
            elif "Gradient Boosting" in model_name:
                st.write("• Early stopping can prevent overfitting")
                st.write("• Learning rate tuning is crucial for performance")
                st.write("• Consider different boosting algorithms (XGBoost, LightGBM)")
            
            elif "SVM" in model_name:
                st.write("• Kernel selection significantly impacts performance")
                st.write("• Feature scaling is essential for SVM")
                st.write("• Consider probability calibration for better predictions")


if __name__ == "__main__":
    # For testing purposes
    st.set_page_config(page_title="Model Performance", layout="wide")
    
    # Create sample models for testing
    sample_models = {}
    
    show_page(pd.DataFrame(), sample_models)
