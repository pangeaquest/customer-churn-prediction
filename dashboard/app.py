"""
Customer Churn Prediction Dashboard

Main Streamlit application for the interactive customer churn prediction dashboard.
This dashboard provides data overview, model predictions, insights, and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import custom modules
try:
    from data.data_loader import DataLoader
    from models.logistic_regression import LogisticRegressionModel
    from models.random_forest import RandomForestModel
    from models.gradient_boosting import GradientBoostingModel
    from evaluation.model_evaluator import ModelEvaluator
    from visualization.eda_plots import EDAPlots
    from utils.config import Config
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Import dashboard pages
from pages import data_overview, predictions, insights, model_performance
from components.widgets import DashboardWidgets

# Configure page
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the customer churn dataset."""
    try:
        # TODO: Implement data loading
        # Load data using DataLoader
        # Cache for performance
        
        data_loader = DataLoader(data_path="../data")
        df = data_loader.load_telco_data()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data for demonstration
        return pd.DataFrame()


@st.cache_resource
def load_models():
    """Load and cache trained models."""
    try:
        # TODO: Implement model loading
        # Load pre-trained models from models directory
        # Cache for performance
        
        models = {
            "Logistic Regression": LogisticRegressionModel(),
            "Random Forest": RandomForestModel(),
            "Gradient Boosting": GradientBoostingModel()
        }
        
        # Load trained models if available
        # for name, model in models.items():
        #     model_path = f"../models/{name.lower().replace(' ', '_')}_model.pkl"
        #     if os.path.exists(model_path):
        #         model.load_model(model_path)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 📊 Dashboard Navigation")
        st.markdown("Use the menu below to navigate between different sections of the dashboard.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation menu
        page = st.selectbox(
            "Select Page",
            ["Data Overview", "Model Predictions", "Business Insights", "Model Performance"],
            index=0
        )
        
        st.markdown("---")
        
        # Dashboard info
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard provides:
        - **Data Overview**: Dataset statistics and visualizations
        - **Predictions**: Real-time churn predictions for customers
        - **Insights**: Business insights and feature analysis
        - **Performance**: Model evaluation and comparison
        """)
        
        st.markdown("---")
        
        # Model status
        st.markdown("### 🤖 Model Status")
        models = load_models()
        
        if models:
            for model_name in models.keys():
                # TODO: Check if model is trained and display status
                status = "🟢 Ready" if models[model_name].is_fitted else "🔴 Not Trained"
                st.markdown(f"**{model_name}**: {status}")
        else:
            st.warning("No models loaded")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.error("No data available. Please check data loading.")
        st.stop()
    
    # Display selected page
    if page == "Data Overview":
        data_overview.show_page(df)
    elif page == "Model Predictions":
        predictions.show_page(df, models)
    elif page == "Business Insights":
        insights.show_page(df, models)
    elif page == "Model Performance":
        model_performance.show_page(df, models)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Customer Churn Prediction Dashboard | Built with Streamlit | 
        <a href='https://github.com/pangeaquest/customer-churn-prediction' target='_blank'>View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


def show_welcome_message():
    """Show welcome message and instructions."""
    st.markdown("""
    ## 👋 Welcome to the Customer Churn Prediction Dashboard!
    
    This interactive dashboard helps you:
    
    ### 📊 **Data Overview**
    - Explore customer dataset statistics
    - Visualize data distributions and patterns
    - Understand customer demographics and behavior
    
    ### 🎯 **Model Predictions**
    - Make real-time churn predictions for individual customers
    - Input customer characteristics and get instant predictions
    - Understand prediction confidence and reasoning
    
    ### 💡 **Business Insights**
    - Discover key factors driving customer churn
    - Analyze feature importance and relationships
    - Get actionable recommendations for retention strategies
    
    ### 📈 **Model Performance**
    - Compare different machine learning models
    - View detailed performance metrics and visualizations
    - Understand model strengths and limitations
    
    ---
    
    ### 🚀 Getting Started
    1. Use the sidebar to navigate between different sections
    2. Start with **Data Overview** to understand the dataset
    3. Try **Model Predictions** to see the models in action
    4. Explore **Business Insights** for strategic recommendations
    5. Check **Model Performance** for technical details
    
    ### 📋 Requirements
    - Ensure the dataset is loaded in the `data/` directory
    - Models should be trained and saved in the `models/` directory
    - All dependencies should be installed (see `requirements.txt`)
    
    ---
    
    **Ready to explore? Use the sidebar to get started! 🎉**
    """)


if __name__ == "__main__":
    # Check if this is the first run
    if "first_run" not in st.session_state:
        st.session_state.first_run = True
        show_welcome_message()
    else:
        main()
