"""
Predictions page for the Customer Churn Prediction Dashboard.

This page allows users to make real-time churn predictions for individual customers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


def show_page(df: pd.DataFrame, models: dict):
    """
    Display the predictions page.
    
    Args:
        df: Customer churn dataset
        models: Dictionary of trained models
    """
    st.header("🎯 Customer Churn Predictions")
    st.markdown("Make real-time churn predictions for individual customers.")
    
    if not models:
        st.error("No trained models available. Please train models first.")
        return
    
    # Model selection
    model_names = list(models.keys())
    selected_model_name = st.selectbox("Select Model", model_names)
    selected_model = models[selected_model_name]
    
    # Check if model is trained
    if not selected_model.is_fitted:
        st.warning(f"Model '{selected_model_name}' is not trained yet.")
        return
    
    # Prediction interface
    show_prediction_interface(df, selected_model, selected_model_name)
    
    # Batch prediction
    show_batch_prediction(df, selected_model, selected_model_name)


def show_prediction_interface(df: pd.DataFrame, model, model_name: str):
    """Show single customer prediction interface."""
    st.subheader("🔮 Single Customer Prediction")
    
    # Create input form
    with st.form("prediction_form"):
        st.markdown("#### Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # TODO: Create input fields based on dataset features
            # Dynamically generate form fields based on dataset columns
            
            # Demographics
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            # Account Information
            st.markdown("**Account Information**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        
        with col2:
            # Services
            st.markdown("**Services**")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            
            # Add-on Services
            st.markdown("**Add-on Services**")
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
            # Financial
            st.markdown("**Financial**")
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, 0.01)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, 0.01)
        
        # Predict button
        predict_button = st.form_submit_button("🔮 Predict Churn", type="primary")
        
        if predict_button:
            # Create customer data
            customer_data = create_customer_data(
                gender, senior_citizen, partner, dependents, tenure, contract,
                paperless_billing, payment_method, phone_service, multiple_lines,
                internet_service, online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies, monthly_charges, total_charges
            )
            
            # Make prediction
            prediction, probability = make_prediction(model, customer_data)
            
            # Display results
            show_prediction_results(prediction, probability, customer_data, model_name)


def create_customer_data(gender, senior_citizen, partner, dependents, tenure, contract,
                        paperless_billing, payment_method, phone_service, multiple_lines,
                        internet_service, online_security, online_backup, device_protection,
                        tech_support, streaming_tv, streaming_movies, monthly_charges, total_charges):
    """Create customer data dictionary from form inputs."""
    return {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }


def make_prediction(model, customer_data):
    """Make churn prediction for a customer."""
    try:
        # TODO: Implement actual prediction logic
        # Convert customer data to model input format
        # Apply same preprocessing as training data
        # Make prediction and return results
        
        # Create DataFrame from customer data
        customer_df = pd.DataFrame([customer_data])
        
        # Placeholder prediction logic
        # In actual implementation, you would:
        # 1. Apply same preprocessing as training
        # 2. Use model.predict() and model.predict_proba()
        
        # For now, return dummy prediction
        prediction = np.random.choice([0, 1])  # 0 = No Churn, 1 = Churn
        probability = np.random.random()  # Random probability
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None


def show_prediction_results(prediction, probability, customer_data, model_name):
    """Display prediction results."""
    if prediction is None:
        return
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Main prediction result
    col1, col2, col3 = st.columns(3)
    
    with col1:
        churn_label = "Will Churn" if prediction == 1 else "Will Stay"
        churn_color = "red" if prediction == 1 else "green"
        st.markdown(f"### :{churn_color}[{churn_label}]")
    
    with col2:
        confidence = probability if prediction == 1 else (1 - probability)
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        risk_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
        st.markdown(f"**Risk Level**: :{risk_color}[{risk_level}]")
    
    # Probability breakdown
    st.markdown("#### Probability Breakdown")
    prob_churn = probability if prediction == 1 else (1 - probability)
    prob_stay = 1 - prob_churn
    
    prob_df = pd.DataFrame({
        'Outcome': ['Will Churn', 'Will Stay'],
        'Probability': [prob_churn, prob_stay]
    })
    
    import plotly.express as px
    fig = px.bar(prob_df, x='Outcome', y='Probability', 
                 color='Outcome', color_discrete_map={'Will Churn': 'red', 'Will Stay': 'green'})
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    show_recommendations(prediction, confidence, customer_data)


def show_recommendations(prediction, confidence, customer_data):
    """Show business recommendations based on prediction."""
    st.markdown("#### 💡 Recommendations")
    
    if prediction == 1:  # Customer likely to churn
        if confidence > 0.7:
            st.error("🚨 **High Risk Customer** - Immediate action required!")
            recommendations = [
                "Contact customer immediately with retention offer",
                "Assign dedicated account manager",
                "Offer service upgrade or discount",
                "Schedule satisfaction survey call"
            ]
        else:
            st.warning("⚠️ **Medium Risk Customer** - Monitor closely")
            recommendations = [
                "Include in next retention campaign",
                "Monitor usage patterns",
                "Send satisfaction survey",
                "Offer loyalty program enrollment"
            ]
        
        for rec in recommendations:
            st.write(f"• {rec}")
    
    else:  # Customer likely to stay
        st.success("✅ **Low Risk Customer** - Continue standard service")
        recommendations = [
            "Include in upselling campaigns",
            "Maintain current service level",
            "Consider for referral program",
            "Monitor for service expansion opportunities"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")


def show_batch_prediction(df: pd.DataFrame, model, model_name: str):
    """Show batch prediction interface."""
    st.subheader("📋 Batch Predictions")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data",
        type=['csv'],
        help="Upload a CSV file with customer data for batch predictions"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! {len(batch_df)} customers found.")
            
            # Show preview
            if st.checkbox("Show data preview"):
                st.dataframe(batch_df.head())
            
            # Make batch predictions
            if st.button("🔮 Run Batch Predictions"):
                with st.spinner("Making predictions..."):
                    # TODO: Implement batch prediction logic
                    # Apply preprocessing and make predictions for all customers
                    
                    # Placeholder: Add random predictions
                    batch_df['Churn_Prediction'] = np.random.choice(['Will Churn', 'Will Stay'], len(batch_df))
                    batch_df['Churn_Probability'] = np.random.random(len(batch_df))
                    batch_df['Risk_Level'] = batch_df['Churn_Probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
                    )
                
                st.success("Batch predictions completed!")
                
                # Show results summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    churn_count = (batch_df['Churn_Prediction'] == 'Will Churn').sum()
                    st.metric("Predicted Churners", churn_count)
                
                with col2:
                    high_risk_count = (batch_df['Risk_Level'] == 'High').sum()
                    st.metric("High Risk Customers", high_risk_count)
                
                with col3:
                    avg_probability = batch_df['Churn_Probability'].mean()
                    st.metric("Average Churn Probability", f"{avg_probability:.1%}")
                
                # Show results table
                st.markdown("#### Prediction Results")
                st.dataframe(batch_df[['Churn_Prediction', 'Churn_Probability', 'Risk_Level']], use_container_width=True)
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample format
        st.info("💡 Upload a CSV file with customer data to make batch predictions")
        
        if st.checkbox("Show expected file format"):
            sample_data = pd.DataFrame({
                'customerID': ['CUST_001', 'CUST_002'],
                'gender': ['Male', 'Female'],
                'tenure': [12, 24],
                'MonthlyCharges': [50.0, 75.0],
                'Contract': ['Month-to-month', 'One year']
            })
            st.dataframe(sample_data)


if __name__ == "__main__":
    # For testing purposes
    st.set_page_config(page_title="Predictions", layout="wide")
    
    # Create sample data and models for testing
    sample_data = pd.DataFrame()
    sample_models = {}
    
    show_page(sample_data, sample_models)
