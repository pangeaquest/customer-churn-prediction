"""
Data Overview page for the Customer Churn Prediction Dashboard.

This page provides comprehensive data exploration and visualization
of the customer churn dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from visualization.eda_plots import EDAPlots


def show_page(df: pd.DataFrame):
    """
    Display the data overview page.
    
    Args:
        df: Customer churn dataset
    """
    st.header("📊 Data Overview")
    st.markdown("Explore the customer churn dataset with interactive visualizations and statistics.")
    
    if df.empty:
        st.error("No data available to display.")
        return
    
    # Dataset Summary
    show_dataset_summary(df)
    
    # Data Quality Assessment
    show_data_quality(df)
    
    # Target Variable Analysis
    show_target_analysis(df)
    
    # Feature Distributions
    show_feature_distributions(df)
    
    # Correlation Analysis
    show_correlation_analysis(df)


def show_dataset_summary(df: pd.DataFrame):
    """Show dataset summary statistics."""
    st.subheader("📋 Dataset Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        st.metric("Total Features", len(df.columns))
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    with col4:
        if 'Churn' in df.columns:
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    # Data types breakdown
    st.markdown("#### Data Types")
    col1, col2 = st.columns(2)
    
    with col1:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.info(f"**Numerical Features**: {len(numerical_cols)}")
        if st.expander("View numerical columns"):
            for col in numerical_cols:
                st.write(f"• {col}")
    
    with col2:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.info(f"**Categorical Features**: {len(categorical_cols)}")
        if st.expander("View categorical columns"):
            for col in categorical_cols:
                unique_count = df[col].nunique()
                st.write(f"• {col} ({unique_count} unique)")


def show_data_quality(df: pd.DataFrame):
    """Show data quality assessment."""
    st.subheader("🔍 Data Quality Assessment")
    
    # Missing values analysis
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        st.warning(f"Found missing values in {len(missing_data)} columns")
        
        # Missing values chart
        fig = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            title="Missing Values by Column",
            labels={'x': 'Number of Missing Values', 'y': 'Column'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"Found {duplicate_count} duplicate rows")
    else:
        st.success("✅ No duplicate rows found!")
    
    # Data consistency checks
    st.markdown("#### Data Consistency")
    
    # TODO: Add more data quality checks
    # Check for inconsistent categorical values
    # Check for outliers in numerical columns
    # Validate data ranges and formats
    
    consistency_issues = []
    
    # Example: Check TotalCharges consistency
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        # Check if TotalCharges is roughly MonthlyCharges * tenure
        expected_total = df['MonthlyCharges'] * df['tenure']
        actual_total = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Allow for some variance (e.g., 20%)
        variance_threshold = 0.2
        inconsistent = abs(expected_total - actual_total) > (expected_total * variance_threshold)
        inconsistent_count = inconsistent.sum()
        
        if inconsistent_count > 0:
            consistency_issues.append(f"TotalCharges inconsistency: {inconsistent_count} records")
    
    if consistency_issues:
        for issue in consistency_issues:
            st.warning(f"⚠️ {issue}")
    else:
        st.success("✅ Data consistency checks passed!")


def show_target_analysis(df: pd.DataFrame):
    """Show target variable analysis."""
    if 'Churn' not in df.columns:
        st.warning("Target variable 'Churn' not found in dataset")
        return
    
    st.subheader("🎯 Target Variable Analysis")
    
    # Churn distribution
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Churn Distribution",
            color_discrete_map={'No': '#2E86AB', 'Yes': '#F24236'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart with percentages
        fig = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            title="Churn Counts",
            text=[f"{count}<br>({pct:.1f}%)" for count, pct in zip(churn_counts.values, churn_pct.values)],
            color=churn_counts.index,
            color_discrete_map={'No': '#2E86AB', 'Yes': '#F24236'}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Class balance assessment
    churn_rate = churn_pct['Yes'] if 'Yes' in churn_pct.index else 0
    
    if churn_rate < 10 or churn_rate > 90:
        st.error(f"⚠️ Highly imbalanced dataset (churn rate: {churn_rate:.1f}%)")
        st.info("Consider using sampling techniques or class weights in modeling")
    elif churn_rate < 20 or churn_rate > 80:
        st.warning(f"⚠️ Moderately imbalanced dataset (churn rate: {churn_rate:.1f}%)")
        st.info("Monitor model performance carefully")
    else:
        st.success(f"✅ Reasonably balanced dataset (churn rate: {churn_rate:.1f}%)")


def show_feature_distributions(df: pd.DataFrame):
    """Show feature distribution analysis."""
    st.subheader("📈 Feature Distributions")
    
    # Separate numerical and categorical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID columns
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')
    
    # Numerical features
    if numerical_cols:
        st.markdown("#### Numerical Features")
        
        # Feature selection
        selected_num_features = st.multiselect(
            "Select numerical features to visualize:",
            numerical_cols,
            default=numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols
        )
        
        if selected_num_features:
            # Create subplots for histograms
            n_features = len(selected_num_features)
            cols = min(3, n_features)
            rows = (n_features + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=selected_num_features,
                vertical_spacing=0.1
            )
            
            for i, feature in enumerate(selected_num_features):
                row = i // cols + 1
                col = i % cols + 1
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(x=df[feature], name=feature, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=300*rows, title="Numerical Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical features
    if categorical_cols:
        st.markdown("#### Categorical Features")
        
        # Feature selection
        selected_cat_features = st.multiselect(
            "Select categorical features to visualize:",
            categorical_cols,
            default=categorical_cols[:3] if len(categorical_cols) >= 3 else categorical_cols
        )
        
        if selected_cat_features:
            for feature in selected_cat_features:
                value_counts = df[feature].value_counts()
                
                # Limit to top 10 categories for readability
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                    st.info(f"Showing top 10 categories for {feature}")
                
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{feature} Distribution",
                    labels={'x': feature, 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)


def show_correlation_analysis(df: pd.DataFrame):
    """Show correlation analysis for numerical features."""
    st.subheader("🔗 Correlation Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        st.info("Need at least 2 numerical features for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto",
        text_auto=True
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # High correlation pairs
    st.markdown("#### High Correlation Pairs")
    
    # Find pairs with correlation > 0.7 or < -0.7
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(high_corr_df, use_container_width=True)
        
        st.warning("⚠️ High correlation detected! Consider feature selection to avoid multicollinearity.")
    else:
        st.success("✅ No high correlation pairs found (threshold: 0.7)")


if __name__ == "__main__":
    # For testing purposes
    st.set_page_config(page_title="Data Overview", layout="wide")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'customerID': ['CUST_001', 'CUST_002', 'CUST_003'],
        'tenure': [12, 24, 6],
        'MonthlyCharges': [50.0, 75.0, 30.0],
        'TotalCharges': [600.0, 1800.0, 180.0],
        'Churn': ['No', 'Yes', 'No']
    })
    
    show_page(sample_data)
