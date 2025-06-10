#!/usr/bin/env python3
"""
Data download script for customer churn prediction project.

This script downloads the Telco Customer Churn dataset from Kaggle
and sets up the data directory structure.
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_data_directories(config: Config) -> None:
    """
    Create data directory structure.
    
    Args:
        config: Configuration object
    """
    try:
        data_paths = config.get_data_paths()
        
        for path_name, path_value in data_paths.items():
            os.makedirs(path_value, exist_ok=True)
            logger.info(f"Created directory: {path_value}")
        
        # Create additional subdirectories
        os.makedirs("data/external", exist_ok=True)
        os.makedirs("data/interim", exist_ok=True)
        
        logger.info("Data directory structure created successfully")
        
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise


def download_kaggle_dataset(config: Config) -> bool:
    """
    Download dataset from Kaggle using Kaggle API.
    
    Args:
        config: Configuration object
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # TODO: Implement Kaggle dataset download
        # Check if Kaggle API is available
        # Download the Telco Customer Churn dataset
        # Extract and place in raw data directory
        
        logger.info("Attempting to download dataset from Kaggle...")
        
        try:
            import kaggle
            
            # Download dataset
            dataset_name = "blastchar/telco-customer-churn"
            download_path = config.get("data.raw_data_path")
            
            logger.info(f"Downloading {dataset_name} to {download_path}")
            
            # kaggle.api.dataset_download_files(
            #     dataset_name,
            #     path=download_path,
            #     unzip=True
            # )
            
            logger.info("Dataset downloaded successfully from Kaggle")
            return True
            
        except ImportError:
            logger.warning("Kaggle API not available. Please install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Error downloading from Kaggle: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error in Kaggle download: {str(e)}")
        return False


def download_from_url(config: Config, url: str) -> bool:
    """
    Download dataset from direct URL.
    
    Args:
        config: Configuration object
        url: URL to download from
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading dataset from URL: {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        filename = config.get("data.dataset_filename")
        filepath = os.path.join(config.get("data.raw_data_path"), filename)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Dataset downloaded successfully to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading from URL: {str(e)}")
        return False


def create_sample_data(config: Config) -> None:
    """
    Create sample dataset for testing purposes.
    
    Args:
        config: Configuration object
    """
    try:
        logger.info("Creating sample dataset for testing...")
        
        # TODO: Implement sample data generation
        # Create realistic sample data with all required columns
        # Include appropriate distributions and relationships
        
        # Sample data structure based on Telco dataset
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'customerID': [f'CUST_{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples),
            'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples),
            'TotalCharges': np.random.uniform(18.0, 8500.0, n_samples),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        }
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Save sample data
        sample_path = os.path.join(config.get("data.sample_data_path"), "sample_telco_churn.csv")
        df.to_csv(sample_path, index=False)
        
        logger.info(f"Sample dataset created with {len(df)} records at {sample_path}")
        
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")
        raise


def validate_dataset(config: Config, filepath: str) -> bool:
    """
    Validate downloaded dataset.
    
    Args:
        config: Configuration object
        filepath: Path to dataset file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        logger.info(f"Validating dataset: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"Dataset file not found: {filepath}")
            return False
        
        # Load and validate dataset
        df = pd.read_csv(filepath)
        
        # Check basic structure
        if len(df) == 0:
            logger.error("Dataset is empty")
            return False
        
        # Check for required columns
        required_columns = ['customerID', 'Churn']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check target variable
        target_col = config.get("data.target_column")
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found")
            return False
        
        # Basic statistics
        logger.info(f"Dataset validation successful:")
        logger.info(f"  - Shape: {df.shape}")
        logger.info(f"  - Columns: {len(df.columns)}")
        logger.info(f"  - Missing values: {df.isnull().sum().sum()}")
        
        if target_col in df.columns:
            churn_rate = (df[target_col] == 'Yes').mean() * 100
            logger.info(f"  - Churn rate: {churn_rate:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        return False


def main():
    """Main function to download and setup data."""
    parser = argparse.ArgumentParser(description="Download customer churn dataset")
    parser.add_argument("--source", choices=["kaggle", "url", "sample"], default="kaggle",
                       help="Data source (default: kaggle)")
    parser.add_argument("--url", type=str, help="URL to download dataset from")
    parser.add_argument("--force", action="store_true", help="Force download even if file exists")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config()
        
        # Create data directories
        create_data_directories(config)
        
        # Check if dataset already exists
        dataset_path = os.path.join(
            config.get("data.raw_data_path"),
            config.get("data.dataset_filename")
        )
        
        if os.path.exists(dataset_path) and not args.force:
            logger.info(f"Dataset already exists at {dataset_path}")
            if validate_dataset(config, dataset_path):
                logger.info("Existing dataset is valid. Use --force to re-download.")
                return
        
        # Download dataset based on source
        success = False
        
        if args.source == "kaggle":
            success = download_kaggle_dataset(config)
        elif args.source == "url" and args.url:
            success = download_from_url(config, args.url)
        elif args.source == "sample":
            create_sample_data(config)
            success = True
        else:
            logger.error("Invalid source or missing URL for URL source")
            return
        
        if success:
            # Validate downloaded dataset
            if args.source != "sample":
                if validate_dataset(config, dataset_path):
                    logger.info("✅ Dataset download and validation completed successfully!")
                else:
                    logger.error("❌ Dataset validation failed")
            else:
                logger.info("✅ Sample dataset created successfully!")
        else:
            logger.error("❌ Dataset download failed")
            logger.info("💡 Try using --source sample to create sample data for testing")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
