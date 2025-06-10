# Data Directory

This directory contains all data files for the Customer Churn Prediction project.

## Directory Structure

```
data/
├── raw/                    # Original, immutable data
├── processed/              # Cleaned and preprocessed data
├── sample/                 # Sample data for testing
├── external/               # External data sources
└── interim/                # Intermediate data processing files
```

## Data Sources

### Primary Dataset: Telco Customer Churn
- **Source**: Kaggle - [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File**: `telco_customer_churn.csv`
- **Size**: ~7,000 customer records
- **Features**: 21 attributes including demographics, services, and billing information
- **Target**: Binary churn indicator (Yes/No)

### Data Description

#### Customer Demographics
- `customerID`: Unique customer identifier
- `gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Whether customer is senior citizen (0/1)
- `Partner`: Whether customer has a partner (Yes/No)
- `Dependents`: Whether customer has dependents (Yes/No)

#### Account Information
- `tenure`: Number of months customer has stayed with company
- `Contract`: Contract term (Month-to-month, One year, Two year)
- `PaperlessBilling`: Whether customer uses paperless billing (Yes/No)
- `PaymentMethod`: Payment method used by customer
- `MonthlyCharges`: Monthly charges for customer
- `TotalCharges`: Total charges for customer

#### Services
- `PhoneService`: Whether customer has phone service (Yes/No)
- `MultipleLines`: Whether customer has multiple lines (Yes/No/No phone service)
- `InternetService`: Type of internet service (DSL/Fiber optic/No)
- `OnlineSecurity`: Whether customer has online security (Yes/No/No internet service)
- `OnlineBackup`: Whether customer has online backup (Yes/No/No internet service)
- `DeviceProtection`: Whether customer has device protection (Yes/No/No internet service)
- `TechSupport`: Whether customer has tech support (Yes/No/No internet service)
- `StreamingTV`: Whether customer has streaming TV (Yes/No/No internet service)
- `StreamingMovies`: Whether customer has streaming movies (Yes/No/No internet service)

#### Target Variable
- `Churn`: Whether customer churned (Yes/No)

## Data Quality

### Known Issues
- Some `TotalCharges` values may be stored as strings with spaces
- No missing values in most columns
- Class imbalance: ~26.5% churn rate

### Data Validation Rules
1. `customerID` should be unique for each record
2. `tenure` should be non-negative integer
3. `MonthlyCharges` and `TotalCharges` should be positive numbers
4. Categorical variables should have consistent values
5. `TotalCharges` should roughly equal `MonthlyCharges * tenure` (allowing for promotions/changes)

## Usage Instructions

### 1. Download Data
```bash
# Option 1: Use the download script
python scripts/download_data.py --source kaggle

# Option 2: Manual download
# Download from Kaggle and place in data/raw/

# Option 3: Generate sample data for testing
python scripts/download_data.py --source sample
```

### 2. Data Processing
```python
from src.data.data_loader import DataLoader

# Load data
loader = DataLoader(data_path="data")
df = loader.load_telco_data()

# Validate data
validation_results = loader.validate_data_schema(df)
```

### 3. Data Preprocessing
```python
from src.data.data_preprocessor import DataPreprocessor

# Preprocess data
preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df)
```

## File Naming Conventions

### Raw Data
- Original files: `{dataset_name}.{extension}`
- Example: `telco_customer_churn.csv`

### Processed Data
- Cleaned data: `{dataset_name}_cleaned.{extension}`
- Preprocessed data: `{dataset_name}_preprocessed.{extension}`
- Feature engineered: `{dataset_name}_features.{extension}`
- Example: `telco_customer_churn_preprocessed.csv`

### Sample Data
- Sample files: `sample_{dataset_name}.{extension}`
- Example: `sample_telco_churn.csv`

## Data Versioning

When updating datasets:
1. Keep original files in `raw/` directory
2. Version processed files with timestamps
3. Document changes in this README
4. Update data validation rules if needed

## Security and Privacy

- No personally identifiable information (PII) should be stored
- Customer IDs are anonymized
- Follow data protection regulations (GDPR, CCPA)
- Secure data access and storage

## Data Backup

- Raw data should be backed up regularly
- Processed data can be regenerated from raw data
- Use version control for data processing scripts
- Document data lineage and transformations

## Contact

For questions about data sources, quality, or access:
- Data Team: data@company.com
- Project Lead: project-lead@company.com

---

**Note**: This directory structure follows data science best practices for reproducible research and production deployment.
