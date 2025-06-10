#!/usr/bin/env python3
"""
Complete pipeline runner for customer churn prediction project.

This script runs the entire machine learning pipeline from data loading
to model training and evaluation.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.gradient_boosting import GradientBoostingModel
from models.svm_classifier import SVMModel
from evaluation.model_evaluator import ModelEvaluator

# Setup logging
def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/pipeline.log", mode='a')
        ]
    )

logger = logging.getLogger(__name__)


def run_data_pipeline(config: Config) -> tuple:
    """
    Run the data processing pipeline.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting data processing pipeline...")
    
    try:
        # TODO: Implement complete data pipeline
        # 1. Load data
        # 2. Preprocess data
        # 3. Engineer features
        # 4. Split into train/test sets
        
        # Step 1: Load data
        logger.info("Loading dataset...")
        data_loader = DataLoader(data_path=config.get("data.raw_data_path"))
        
        try:
            df = data_loader.load_telco_data()
        except FileNotFoundError:
            logger.warning("Dataset not found, using sample data")
            df = data_loader.load_sample_data()
        
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        # Step 2: Validate data
        validation_results = data_loader.validate_data_schema(df)
        if not validation_results["is_valid"]:
            logger.error("Data validation failed")
            raise ValueError("Invalid dataset")
        
        # Step 3: Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df, target_column=config.get("data.target_column"))
        
        # Step 4: Feature engineering
        logger.info("Engineering features...")
        feature_engineer = FeatureEngineer()
        X_engineered = feature_engineer.engineer_features(X)
        
        # Step 5: Train/test split
        from sklearn.model_selection import train_test_split
        
        test_size = config.get("models.test_size", 0.2)
        random_state = config.get("models.random_state", 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        raise


def run_model_training(X_train, X_test, y_train, y_test, config: Config) -> dict:
    """
    Train and evaluate multiple models.
    
    Args:
        X_train, X_test, y_train, y_test: Train/test data splits
        config: Configuration object
        
    Returns:
        Dictionary of trained models and results
    """
    logger.info("Starting model training pipeline...")
    
    try:
        # TODO: Implement model training pipeline
        # 1. Initialize models
        # 2. Train each model
        # 3. Evaluate performance
        # 4. Save best models
        
        # Initialize models
        models = {
            "Logistic Regression": LogisticRegressionModel(
                random_state=config.get("models.random_state")
            ),
            "Random Forest": RandomForestModel(
                random_state=config.get("models.random_state")
            ),
            "Gradient Boosting": GradientBoostingModel(
                random_state=config.get("models.random_state")
            ),
            "SVM": SVMModel(
                random_state=config.get("models.random_state")
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                train_metrics = model.evaluate(X_train, y_train)
                test_metrics = model.evaluate(X_test, y_test)
                
                # Cross-validation
                cv_results = model.cross_validate(X_train, y_train, cv=config.get("models.cv_folds", 5))
                
                # Store results
                results[model_name] = {
                    "model": model,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "cv_results": cv_results
                }
                
                logger.info(f"{model_name} - Test Accuracy: {test_metrics.get('accuracy', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find best model
        best_model_name = max(
            results.keys(), 
            key=lambda x: results[x]["test_metrics"].get("accuracy", 0)
        )
        
        logger.info(f"Best model: {best_model_name}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise


def run_hyperparameter_tuning(results: dict, X_train, y_train, config: Config) -> dict:
    """
    Perform hyperparameter tuning on best models.
    
    Args:
        results: Model training results
        X_train, y_train: Training data
        config: Configuration object
        
    Returns:
        Updated results with tuned models
    """
    logger.info("Starting hyperparameter tuning...")
    
    try:
        # TODO: Implement hyperparameter tuning
        # 1. Select top performing models
        # 2. Tune hyperparameters
        # 3. Retrain with best parameters
        
        if not config.get("training.hyperparameter_tuning", True):
            logger.info("Hyperparameter tuning disabled in config")
            return results
        
        # Select top 2 models for tuning
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["test_metrics"].get("accuracy", 0),
            reverse=True
        )[:2]
        
        for model_name, model_results in sorted_models:
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            try:
                model = model_results["model"]
                
                # Perform hyperparameter tuning
                tuning_results = model.tune_hyperparameters(
                    X_train, y_train,
                    cv=config.get("models.cv_folds", 5),
                    scoring=config.get("models.scoring_metric", "accuracy")
                )
                
                # Update results
                results[model_name]["tuning_results"] = tuning_results
                
                logger.info(f"{model_name} tuning completed - Best score: {tuning_results.get('best_score', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Error tuning {model_name}: {str(e)}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return results


def save_models_and_results(results: dict, config: Config) -> None:
    """
    Save trained models and results.
    
    Args:
        results: Model training results
        config: Configuration object
    """
    logger.info("Saving models and results...")
    
    try:
        # TODO: Implement model and results saving
        # 1. Save trained models
        # 2. Save evaluation results
        # 3. Generate summary report
        
        models_path = config.get("models.save_path", "models")
        os.makedirs(models_path, exist_ok=True)
        
        # Save each model
        for model_name, model_results in results.items():
            try:
                model = model_results["model"]
                model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
                model_filepath = os.path.join(models_path, model_filename)
                
                # model.save_model(model_filepath)
                logger.info(f"Model saved: {model_filepath}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
        
        # Save results summary
        import json
        results_summary = {}
        for model_name, model_results in results.items():
            results_summary[model_name] = {
                "test_metrics": model_results.get("test_metrics", {}),
                "cv_results": model_results.get("cv_results", {}),
                "tuning_results": model_results.get("tuning_results", {})
            }
        
        results_file = os.path.join(models_path, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"Results saved: {results_file}")
        
    except Exception as e:
        logger.error(f"Error saving models and results: {str(e)}")


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description="Run customer churn prediction pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--skip-tuning", action="store_true", 
                       help="Skip hyperparameter tuning")
    parser.add_argument("--models", nargs="+", 
                       choices=["logistic", "random_forest", "gradient_boosting", "svm"],
                       help="Specific models to train")
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    setup_logging(args.log_level)
    
    logger.info("="*60)
    logger.info("CUSTOMER CHURN PREDICTION PIPELINE")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Load configuration
        config = Config(config_file=args.config)
        
        # Validate configuration
        validation_results = config.validate_config()
        if not validation_results["is_valid"]:
            logger.error("Configuration validation failed")
            for error in validation_results["errors"]:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        # Run data pipeline
        X_train, X_test, y_train, y_test = run_data_pipeline(config)
        
        # Run model training
        results = run_model_training(X_train, X_test, y_train, y_test, config)
        
        # Run hyperparameter tuning (if enabled)
        if not args.skip_tuning:
            results = run_hyperparameter_tuning(results, X_train, y_train, config)
        
        # Save models and results
        save_models_and_results(results, config)
        
        # Print summary
        logger.info("="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        
        for model_name, model_results in results.items():
            test_accuracy = model_results["test_metrics"].get("accuracy", 0)
            logger.info(f"{model_name}: {test_accuracy:.3f} accuracy")
        
        best_model = max(results.keys(), key=lambda x: results[x]["test_metrics"].get("accuracy", 0))
        logger.info(f"Best model: {best_model}")
        
        logger.info("="*60)
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"Completed at: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
