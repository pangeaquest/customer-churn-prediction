# Customer Churn Prediction and Analysis System

A comprehensive machine learning pipeline to predict customer churn using Python, scikit-learn, and create an interactive dashboard for business insights.

## 📋 Project Overview

This project develops an end-to-end data science solution for predicting customer churn in a telecommunications company. The system includes comprehensive data analysis, feature engineering, multiple ML models, performance evaluation, and an interactive dashboard for business stakeholders.

## 🎯 Problem Statement

**Problem ID**: P002
**Difficulty**: Intermediate
**Category**: Data Science
**Time Estimate**: 2-3 weeks

Build a complete machine learning pipeline that:
- Predicts customer churn with high accuracy
- Provides interpretable insights for business decisions
- Offers an interactive dashboard for stakeholders
- Follows data science best practices and is production-ready

## 🛠️ Technology Stack

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **XGBoost/LightGBM**: Gradient boosting models
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Interactive dashboard
- **SHAP**: Model interpretability
- **Jupyter**: Notebook environment

## 📁 Project Structure

```
customer-churn-prediction/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── .gitignore                    # Git ignore rules
├── data/                         # Data storage
│   ├── raw/                      # Original datasets
│   ├── processed/                # Cleaned datasets
│   └── sample/                   # Sample data for testing
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_model_optimization.ipynb
├── src/                          # Source code modules
│   ├── data/                     # Data processing modules
│   ├── models/                   # ML model implementations
│   ├── evaluation/               # Model evaluation tools
│   ├── visualization/            # Plotting utilities
│   └── utils/                    # Helper functions
├── dashboard/                    # Streamlit dashboard
│   ├── app.py                    # Main dashboard app
│   ├── pages/                    # Dashboard pages
│   └── components/               # Reusable components
├── models/                       # Saved model files
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── scripts/                      # Utility scripts
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/pangeaquest/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup the package
pip install -e .
```

### 2. Data Setup

```bash
# Download the dataset
python scripts/download_data.py

# Or manually download from:
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

### 3. Run the Analysis

```bash
# Run the complete pipeline
python scripts/run_pipeline.py

# Or run individual notebooks in order:
# 01_data_exploration.ipynb → 07_model_optimization.ipynb
```

### 4. Launch Dashboard

```bash
# Start the Streamlit dashboard
streamlit run dashboard/app.py
```

## 📊 Project Steps

This project follows a structured 12-step approach:

### Phase 1: Setup and Data Understanding
1. **Environment Setup** - Create virtual environment and install dependencies
2. **Data Exploration** - Initial data loading and basic analysis
3. **Exploratory Data Analysis** - Comprehensive data visualization and insights

### Phase 2: Data Preparation
4. **Data Preprocessing** - Handle missing values, outliers, and data cleaning
5. **Feature Engineering** - Create new features and select optimal feature subset

### Phase 3: Model Development
6. **Model Training** - Build multiple ML models (Logistic Regression, Random Forest, XGBoost, SVM)
7. **Model Evaluation** - Compare models using various metrics and cross-validation
8. **Model Optimization** - Hyperparameter tuning for best performing model

### Phase 4: Analysis and Deployment
9. **Model Interpretability** - Feature importance and SHAP analysis
10. **Dashboard Creation** - Interactive Streamlit dashboard for business insights
11. **Testing Framework** - Unit tests and validation scripts
12. **Documentation** - Business presentation and deployment guide

## 🎯 Acceptance Criteria

Each step includes specific acceptance criteria:
- ✅ Functional code implementations
- ✅ Comprehensive documentation
- ✅ Performance benchmarks
- ✅ Business value demonstration
- ✅ Production-ready code quality

## 📈 Expected Outcomes

- **High-accuracy churn prediction model** (target: >85% accuracy)
- **Interpretable business insights** from feature analysis
- **Interactive dashboard** for real-time predictions
- **Comprehensive documentation** for stakeholders
- **Production-ready codebase** with testing framework

## 🔧 Development Guidelines

### Code Quality
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Implement error handling
- Write unit tests for all functions

### Data Science Best Practices
- Document all assumptions and decisions
- Ensure reproducible results with random seeds
- Validate models on holdout test sets
- Include model interpretability analysis

### Version Control
- Commit frequently with descriptive messages
- Use feature branches for development
- Include .gitignore for data files and models

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://streamlit.io/docs)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Augment Agent** - Initial template creation
- **Your Name** - Implementation and customization

## 🙏 Acknowledgments

- Kaggle for providing the Telco Customer Churn dataset
- The open-source community for excellent ML libraries
- Business stakeholders for domain expertise and requirements

---

**Note**: This is a template codebase. Each step contains predefined function signatures and structure, but implementations need to be completed according to the project requirements.