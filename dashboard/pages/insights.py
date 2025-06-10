"""
Business Insights page for the Customer Churn Prediction Dashboard.

This page provides business insights and feature analysis for customer churn.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_page(df: pd.DataFrame, models: dict):
    """
    Display the business insights page.
    
    Args:
        df: Customer churn dataset
        models: Dictionary of trained models
    """
    st.header("💡 Business Insights")
    st.markdown("Discover key factors driving customer churn and actionable business recommendations.")
    
    if df.empty:
        st.error("No data available for analysis.")
        return
    
    # Feature Importance Analysis
    show_feature_importance(models)
    
    # Churn Analysis by Segments
    show_churn_analysis(df)
    
    # Customer Segmentation
    show_customer_segmentation(df)
    
    # Business Recommendations
    show_business_recommendations(df)


def show_feature_importance(models: dict):
    """Show feature importance analysis from trained models."""
    st.subheader("🎯 Feature Importance Analysis")
    
    if not models:
        st.warning("No trained models available for feature importance analysis.")
        return
    
    # Model selection for feature importance
    model_names = [name for name, model in models.items() if model.is_fitted]
    
    if not model_names:
        st.warning("No trained models available.")
        return
    
    selected_model_name = st.selectbox("Select model for feature importance:", model_names)
    selected_model = models[selected_model_name]
    
    # Get feature importance
    feature_importance = selected_model.get_feature_importance()
    
    if feature_importance:
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        # Top 15 features
        top_features = importance_df.head(15)
        
        # Feature importance chart
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top 15 Feature Importance ({selected_model_name})",
            labels={'Importance': 'Importance Score', 'Feature': 'Features'}
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance insights
        st.markdown("#### 🔍 Key Insights")
        
        top_3_features = top_features.head(3)['Feature'].tolist()
        st.success(f"**Top 3 Most Important Features**: {', '.join(top_3_features)}")
        
        # Business interpretation
        feature_interpretations = {
            'TotalCharges': "Customer lifetime value - higher spending customers may be more valuable to retain",
            'MonthlyCharges': "Price sensitivity - customers with higher monthly charges may be more likely to churn",
            'tenure': "Customer loyalty - longer tenure indicates stronger relationship",
            'Contract': "Commitment level - contract type strongly influences churn behavior",
            'InternetService': "Service quality - internet service type affects satisfaction",
            'PaymentMethod': "Payment behavior - payment method indicates customer preferences",
            'TechSupport': "Service satisfaction - technical support affects customer experience"
        }
        
        for feature in top_3_features:
            if feature in feature_interpretations:
                st.info(f"**{feature}**: {feature_interpretations[feature]}")
    
    else:
        st.warning("Feature importance not available for selected model.")


def show_churn_analysis(df: pd.DataFrame):
    """Show churn analysis by different customer segments."""
    st.subheader("📊 Churn Analysis by Customer Segments")
    
    if 'Churn' not in df.columns:
        st.warning("Churn column not found in dataset.")
        return
    
    # Calculate churn rates by different segments
    segments = ['Contract', 'InternetService', 'PaymentMethod', 'gender', 'SeniorCitizen']
    available_segments = [seg for seg in segments if seg in df.columns]
    
    if not available_segments:
        st.warning("No segmentation columns available.")
        return
    
    # Segment selection
    selected_segment = st.selectbox("Select customer segment:", available_segments)
    
    # Calculate churn rate by segment
    churn_by_segment = df.groupby(selected_segment)['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).sort_values(ascending=False)
    
    # Count by segment
    count_by_segment = df[selected_segment].value_counts()
    
    # Create visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn rate chart
        fig = px.bar(
            x=churn_by_segment.index,
            y=churn_by_segment.values,
            title=f"Churn Rate by {selected_segment}",
            labels={'x': selected_segment, 'y': 'Churn Rate (%)'},
            color=churn_by_segment.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer count chart
        fig = px.bar(
            x=count_by_segment.index,
            y=count_by_segment.values,
            title=f"Customer Count by {selected_segment}",
            labels={'x': selected_segment, 'y': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown("#### 📈 Segment Insights")
    
    highest_churn_segment = churn_by_segment.index[0]
    highest_churn_rate = churn_by_segment.iloc[0]
    lowest_churn_segment = churn_by_segment.index[-1]
    lowest_churn_rate = churn_by_segment.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error(f"**Highest Risk**: {highest_churn_segment} ({highest_churn_rate:.1f}% churn rate)")
    
    with col2:
        st.success(f"**Lowest Risk**: {lowest_churn_segment} ({lowest_churn_rate:.1f}% churn rate)")
    
    # Additional analysis for specific segments
    if selected_segment == 'Contract':
        st.info("💡 **Contract Insight**: Month-to-month contracts typically show highest churn rates. Consider incentivizing longer-term contracts.")
    elif selected_segment == 'InternetService':
        st.info("💡 **Service Insight**: Fiber optic customers may have higher churn due to service quality issues or higher expectations.")
    elif selected_segment == 'PaymentMethod':
        st.info("💡 **Payment Insight**: Electronic check users often show higher churn. Consider promoting automatic payment methods.")


def show_customer_segmentation(df: pd.DataFrame):
    """Show customer segmentation analysis."""
    st.subheader("👥 Customer Segmentation")
    
    # Create customer segments based on tenure and charges
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        
        # Define segments
        def categorize_customer(row):
            if row['tenure'] <= 12:
                if row['MonthlyCharges'] <= 50:
                    return "New Low-Value"
                else:
                    return "New High-Value"
            elif row['tenure'] <= 36:
                if row['MonthlyCharges'] <= 50:
                    return "Medium Low-Value"
                else:
                    return "Medium High-Value"
            else:
                if row['MonthlyCharges'] <= 50:
                    return "Loyal Low-Value"
                else:
                    return "Loyal High-Value"
        
        df_temp = df.copy()
        df_temp['Customer_Segment'] = df_temp.apply(categorize_customer, axis=1)
        
        # Segment analysis
        segment_analysis = df_temp.groupby('Customer_Segment').agg({
            'Churn': lambda x: (x == 'Yes').mean() * 100,
            'MonthlyCharges': 'mean',
            'tenure': 'mean'
        }).round(2)
        
        segment_counts = df_temp['Customer_Segment'].value_counts()
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment distribution
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Segment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn rate by segment
            fig = px.bar(
                x=segment_analysis.index,
                y=segment_analysis['Churn'],
                title="Churn Rate by Customer Segment",
                labels={'x': 'Customer Segment', 'y': 'Churn Rate (%)'},
                color=segment_analysis['Churn'],
                color_continuous_scale='Reds'
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment details table
        st.markdown("#### Segment Details")
        segment_analysis['Customer_Count'] = segment_counts
        segment_analysis = segment_analysis.rename(columns={
            'Churn': 'Churn_Rate_(%)',
            'MonthlyCharges': 'Avg_Monthly_Charges',
            'tenure': 'Avg_Tenure_Months'
        })
        st.dataframe(segment_analysis, use_container_width=True)
        
        # Segment recommendations
        st.markdown("#### 🎯 Segment-Specific Strategies")
        
        highest_risk_segment = segment_analysis['Churn_Rate_(%)'].idxmax()
        st.error(f"**Focus Area**: {highest_risk_segment} segment has the highest churn risk")
        
        segment_strategies = {
            "New Low-Value": "Onboarding programs, value demonstration, service education",
            "New High-Value": "Premium support, loyalty programs, service optimization",
            "Medium Low-Value": "Upselling opportunities, value-added services, retention offers",
            "Medium High-Value": "Account management, service expansion, loyalty rewards",
            "Loyal Low-Value": "Appreciation programs, referral incentives, service upgrades",
            "Loyal High-Value": "VIP treatment, exclusive offers, partnership opportunities"
        }
        
        for segment, strategy in segment_strategies.items():
            if segment in segment_analysis.index:
                churn_rate = segment_analysis.loc[segment, 'Churn_Rate_(%)']
                st.write(f"**{segment}** ({churn_rate:.1f}% churn): {strategy}")


def show_business_recommendations(df: pd.DataFrame):
    """Show actionable business recommendations."""
    st.subheader("🚀 Business Recommendations")
    
    # Calculate key metrics for recommendations
    if 'Churn' in df.columns:
        overall_churn_rate = (df['Churn'] == 'Yes').mean() * 100
        
        # Priority recommendations based on data insights
        st.markdown("#### 🎯 High Priority Actions")
        
        recommendations = []
        
        # Contract-based recommendations
        if 'Contract' in df.columns:
            monthly_churn = df[df['Contract'] == 'Month-to-month']['Churn']
            if len(monthly_churn) > 0:
                monthly_churn_rate = (monthly_churn == 'Yes').mean() * 100
                if monthly_churn_rate > overall_churn_rate * 1.5:
                    recommendations.append({
                        "priority": "High",
                        "action": "Contract Incentive Program",
                        "description": f"Month-to-month customers have {monthly_churn_rate:.1f}% churn rate. Offer discounts for annual contracts.",
                        "impact": "Could reduce churn by 15-20%"
                    })
        
        # Service quality recommendations
        if 'InternetService' in df.columns:
            fiber_customers = df[df['InternetService'] == 'Fiber optic']
            if len(fiber_customers) > 0:
                fiber_churn_rate = (fiber_customers['Churn'] == 'Yes').mean() * 100
                if fiber_churn_rate > overall_churn_rate * 1.2:
                    recommendations.append({
                        "priority": "High",
                        "action": "Fiber Service Quality Improvement",
                        "description": f"Fiber optic customers have {fiber_churn_rate:.1f}% churn rate. Investigate service quality issues.",
                        "impact": "Could improve customer satisfaction by 25%"
                    })
        
        # New customer recommendations
        if 'tenure' in df.columns:
            new_customers = df[df['tenure'] <= 12]
            if len(new_customers) > 0:
                new_customer_churn = (new_customers['Churn'] == 'Yes').mean() * 100
                if new_customer_churn > overall_churn_rate * 1.3:
                    recommendations.append({
                        "priority": "High",
                        "action": "Enhanced Onboarding Program",
                        "description": f"New customers (≤12 months) have {new_customer_churn:.1f}% churn rate. Implement 90-day onboarding.",
                        "impact": "Could reduce early churn by 30%"
                    })
        
        # Payment method recommendations
        if 'PaymentMethod' in df.columns:
            electronic_check = df[df['PaymentMethod'] == 'Electronic check']
            if len(electronic_check) > 0:
                ec_churn_rate = (electronic_check['Churn'] == 'Yes').mean() * 100
                if ec_churn_rate > overall_churn_rate * 1.2:
                    recommendations.append({
                        "priority": "Medium",
                        "action": "Payment Method Migration",
                        "description": f"Electronic check users have {ec_churn_rate:.1f}% churn rate. Incentivize automatic payments.",
                        "impact": "Could reduce payment-related churn by 20%"
                    })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            priority_color = "red" if rec["priority"] == "High" else "orange"
            
            with st.container():
                st.markdown(f"**{i}. {rec['action']}** :{priority_color}[{rec['priority']} Priority]")
                st.write(f"📋 {rec['description']}")
                st.write(f"📈 Expected Impact: {rec['impact']}")
                st.markdown("---")
        
        # Implementation roadmap
        st.markdown("#### 📅 Implementation Roadmap")
        
        roadmap = [
            {"Phase": "Immediate (0-30 days)", "Actions": [
                "Deploy churn prediction model",
                "Identify high-risk customers",
                "Launch retention campaigns for month-to-month customers"
            ]},
            {"Phase": "Short-term (1-3 months)", "Actions": [
                "Implement enhanced onboarding program",
                "Investigate fiber service quality issues",
                "Launch payment method migration campaign"
            ]},
            {"Phase": "Medium-term (3-6 months)", "Actions": [
                "Develop customer segmentation strategies",
                "Implement personalized retention offers",
                "Enhance customer service training"
            ]},
            {"Phase": "Long-term (6-12 months)", "Actions": [
                "Optimize service offerings based on insights",
                "Implement predictive customer lifetime value",
                "Develop advanced retention automation"
            ]}
        ]
        
        for phase_info in roadmap:
            with st.expander(f"📋 {phase_info['Phase']}"):
                for action in phase_info['Actions']:
                    st.write(f"• {action}")
        
        # ROI Estimation
        st.markdown("#### 💰 Expected ROI")
        
        if 'MonthlyCharges' in df.columns:
            avg_monthly_revenue = df['MonthlyCharges'].mean()
            total_customers = len(df)
            churned_customers = (df['Churn'] == 'Yes').sum()
            
            # Estimate potential savings
            potential_churn_reduction = 0.15  # 15% reduction
            customers_saved = churned_customers * potential_churn_reduction
            annual_revenue_saved = customers_saved * avg_monthly_revenue * 12
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Potential Customers Saved", f"{customers_saved:.0f}")
            
            with col2:
                st.metric("Annual Revenue Protected", f"${annual_revenue_saved:,.0f}")
            
            with col3:
                roi_multiple = annual_revenue_saved / 100000  # Assuming $100k investment
                st.metric("Estimated ROI", f"{roi_multiple:.1f}x")


if __name__ == "__main__":
    # For testing purposes
    st.set_page_config(page_title="Business Insights", layout="wide")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'Contract': ['Month-to-month', 'One year', 'Two year'] * 100,
        'Churn': ['Yes', 'No'] * 150,
        'tenure': np.random.randint(1, 72, 300),
        'MonthlyCharges': np.random.uniform(20, 100, 300)
    })
    
    show_page(sample_data, {})
