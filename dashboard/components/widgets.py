"""
Reusable widgets and components for the Customer Churn Prediction Dashboard.

This module contains common UI components used across different pages
of the dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional


class DashboardWidgets:
    """Collection of reusable dashboard widgets and components."""
    
    @staticmethod
    def metric_card(title: str, value: str, delta: Optional[str] = None, 
                   delta_color: str = "normal") -> None:
        """
        Display a metric card with title, value, and optional delta.
        
        Args:
            title: Metric title
            value: Metric value
            delta: Optional delta value
            delta_color: Color for delta ("normal", "inverse", "off")
        """
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color
        )
    
    @staticmethod
    def info_box(title: str, content: str, box_type: str = "info") -> None:
        """
        Display an information box with title and content.
        
        Args:
            title: Box title
            content: Box content
            box_type: Type of box ("info", "success", "warning", "error")
        """
        if box_type == "info":
            st.info(f"**{title}**\n\n{content}")
        elif box_type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif box_type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif box_type == "error":
            st.error(f"**{title}**\n\n{content}")
    
    @staticmethod
    def progress_bar(value: float, max_value: float = 1.0, 
                    label: str = "", format_str: str = "%.1f") -> None:
        """
        Display a progress bar with label.
        
        Args:
            value: Current value
            max_value: Maximum value
            label: Progress bar label
            format_str: Format string for value display
        """
        progress = value / max_value
        st.progress(progress)
        if label:
            st.caption(f"{label}: {format_str % value}")
    
    @staticmethod
    def data_table(df: pd.DataFrame, title: str = "", 
                  max_rows: int = 10, searchable: bool = True) -> None:
        """
        Display a formatted data table.
        
        Args:
            df: DataFrame to display
            title: Table title
            max_rows: Maximum rows to display
            searchable: Whether to make table searchable
        """
        if title:
            st.markdown(f"#### {title}")
        
        if searchable and len(df) > max_rows:
            # Add search functionality
            search_term = st.text_input("🔍 Search table:", key=f"search_{title}")
            if search_term:
                # Search across all string columns
                string_cols = df.select_dtypes(include=['object']).columns
                mask = df[string_cols].astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                df = df[mask]
        
        # Display table
        if len(df) > max_rows:
            st.dataframe(df.head(max_rows), use_container_width=True)
            st.caption(f"Showing first {max_rows} of {len(df)} rows")
        else:
            st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def download_button(data: Any, filename: str, label: str = "Download", 
                       file_type: str = "csv") -> None:
        """
        Create a download button for data.
        
        Args:
            data: Data to download (DataFrame, dict, etc.)
            filename: Name of the file to download
            label: Button label
            file_type: Type of file ("csv", "json", "txt")
        """
        if file_type == "csv" and isinstance(data, pd.DataFrame):
            csv_data = data.to_csv(index=False)
            st.download_button(
                label=f"📥 {label}",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
        elif file_type == "json":
            import json
            json_data = json.dumps(data, indent=2, default=str)
            st.download_button(
                label=f"📥 {label}",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
        elif file_type == "txt":
            st.download_button(
                label=f"📥 {label}",
                data=str(data),
                file_name=filename,
                mime="text/plain"
            )
    
    @staticmethod
    def filter_widget(df: pd.DataFrame, column: str, 
                     widget_type: str = "multiselect") -> List[Any]:
        """
        Create a filter widget for a DataFrame column.
        
        Args:
            df: DataFrame to filter
            column: Column name to filter on
            widget_type: Type of widget ("multiselect", "selectbox", "slider")
            
        Returns:
            Selected values
        """
        if column not in df.columns:
            st.error(f"Column '{column}' not found in data")
            return []
        
        unique_values = df[column].unique()
        
        if widget_type == "multiselect":
            return st.multiselect(
                f"Filter by {column}:",
                options=unique_values,
                default=unique_values
            )
        elif widget_type == "selectbox":
            return [st.selectbox(f"Select {column}:", options=unique_values)]
        elif widget_type == "slider" and df[column].dtype in ['int64', 'float64']:
            min_val, max_val = df[column].min(), df[column].max()
            return list(st.slider(
                f"Filter {column}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            ))
        else:
            st.error(f"Unsupported widget type: {widget_type}")
            return []
    
    @staticmethod
    def comparison_chart(data: Dict[str, float], title: str = "Comparison", 
                        chart_type: str = "bar") -> None:
        """
        Create a comparison chart from dictionary data.
        
        Args:
            data: Dictionary with labels as keys and values as values
            title: Chart title
            chart_type: Type of chart ("bar", "pie", "line")
        """
        df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
        
        if chart_type == "bar":
            fig = px.bar(df, x='Category', y='Value', title=title)
        elif chart_type == "pie":
            fig = px.pie(df, values='Value', names='Category', title=title)
        elif chart_type == "line":
            fig = px.line(df, x='Category', y='Value', title=title)
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def status_indicator(status: str, label: str = "") -> None:
        """
        Display a status indicator with color coding.
        
        Args:
            status: Status value ("success", "warning", "error", "info")
            label: Optional label for the status
        """
        status_config = {
            "success": {"color": "green", "icon": "✅"},
            "warning": {"color": "orange", "icon": "⚠️"},
            "error": {"color": "red", "icon": "❌"},
            "info": {"color": "blue", "icon": "ℹ️"}
        }
        
        config = status_config.get(status, status_config["info"])
        
        if label:
            st.markdown(f"{config['icon']} **{label}**: :{config['color']}[{status.upper()}]")
        else:
            st.markdown(f"{config['icon']} :{config['color']}[{status.upper()}]")
    
    @staticmethod
    def feature_importance_chart(importance_data: Dict[str, float], 
                               top_n: int = 10, title: str = "Feature Importance") -> None:
        """
        Create a feature importance chart.
        
        Args:
            importance_data: Dictionary with feature names and importance scores
            top_n: Number of top features to display
            title: Chart title
        """
        # Sort and get top N features
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
        
        fig = px.bar(
            df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=title,
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def model_metrics_display(metrics: Dict[str, float]) -> None:
        """
        Display model performance metrics in a formatted layout.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        # Create columns for metrics
        n_metrics = len(metrics)
        cols = st.columns(n_metrics)
        
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i]:
                # Format metric value
                if isinstance(metric_value, float):
                    formatted_value = f"{metric_value:.3f}"
                else:
                    formatted_value = str(metric_value)
                
                st.metric(metric_name, formatted_value)
    
    @staticmethod
    def sidebar_filters(df: pd.DataFrame, filter_columns: List[str]) -> Dict[str, Any]:
        """
        Create sidebar filters for multiple columns.
        
        Args:
            df: DataFrame to create filters for
            filter_columns: List of column names to create filters for
            
        Returns:
            Dictionary of filter values
        """
        filters = {}
        
        st.sidebar.markdown("### 🔍 Filters")
        
        for column in filter_columns:
            if column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    # Numeric filter
                    min_val, max_val = df[column].min(), df[column].max()
                    filters[column] = st.sidebar.slider(
                        f"{column}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                else:
                    # Categorical filter
                    unique_values = df[column].unique()
                    filters[column] = st.sidebar.multiselect(
                        f"{column}:",
                        options=unique_values,
                        default=unique_values
                    )
        
        return filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to a DataFrame.
        
        Args:
            df: DataFrame to filter
            filters: Dictionary of filter values
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        for column, filter_value in filters.items():
            if column in filtered_df.columns:
                if isinstance(filter_value, tuple):
                    # Range filter for numeric columns
                    min_val, max_val = filter_value
                    filtered_df = filtered_df[
                        (filtered_df[column] >= min_val) & 
                        (filtered_df[column] <= max_val)
                    ]
                elif isinstance(filter_value, list):
                    # Multi-select filter for categorical columns
                    if filter_value:  # Only apply if values are selected
                        filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
        
        return filtered_df


# Convenience functions for common widgets
def show_kpi_cards(kpis: Dict[str, Dict[str, Any]]) -> None:
    """
    Display KPI cards in a row.
    
    Args:
        kpis: Dictionary with KPI data
              Format: {"KPI Name": {"value": "123", "delta": "+5%", "delta_color": "normal"}}
    """
    cols = st.columns(len(kpis))
    
    for i, (kpi_name, kpi_data) in enumerate(kpis.items()):
        with cols[i]:
            DashboardWidgets.metric_card(
                title=kpi_name,
                value=kpi_data.get("value", "N/A"),
                delta=kpi_data.get("delta"),
                delta_color=kpi_data.get("delta_color", "normal")
            )


def show_data_summary(df: pd.DataFrame) -> None:
    """
    Show a quick data summary with key statistics.
    
    Args:
        df: DataFrame to summarize
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
