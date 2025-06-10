"""
Configuration management for customer churn prediction project.

This module provides centralized configuration management for all
project settings, paths, and parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for the customer churn prediction project.
    
    This class handles loading and managing configuration settings
    from various sources including files, environment variables, and defaults.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file (YAML format)
        """
        self.config_file = config_file
        self.config = self._load_default_config()
        
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        self._load_environment_variables()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration settings.
        
        Returns:
            Dictionary containing default configuration
        """
        return {
            # Project settings
            "project": {
                "name": "customer-churn-prediction",
                "version": "1.0.0",
                "description": "Customer churn prediction and analysis system"
            },
            
            # Data settings
            "data": {
                "raw_data_path": "data/raw",
                "processed_data_path": "data/processed",
                "sample_data_path": "data/sample",
                "dataset_filename": "telco_customer_churn.csv",
                "target_column": "Churn",
                "customer_id_column": "customerID"
            },
            
            # Model settings
            "models": {
                "save_path": "models",
                "random_state": 42,
                "test_size": 0.2,
                "validation_size": 0.2,
                "cv_folds": 5,
                "scoring_metric": "accuracy"
            },
            
            # Feature engineering settings
            "features": {
                "scaling_method": "standard",
                "encoding_method": "auto",
                "feature_selection_method": "auto",
                "max_features": 50,
                "correlation_threshold": 0.95
            },
            
            # Training settings
            "training": {
                "hyperparameter_tuning": True,
                "early_stopping": True,
                "class_weight": "balanced",
                "n_jobs": -1
            },
            
            # Evaluation settings
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
                "threshold": 0.5,
                "plot_confusion_matrix": True,
                "plot_roc_curve": True,
                "plot_feature_importance": True
            },
            
            # Dashboard settings
            "dashboard": {
                "host": "localhost",
                "port": 8501,
                "debug": False,
                "cache_data": True,
                "max_upload_size": 200  # MB
            },
            
            # Logging settings
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/churn_prediction.log"
            }
        }
    
    def _load_config_file(self, config_file: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Merge with default config
            self._deep_update(self.config, file_config)
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {str(e)}")
            raise
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        # TODO: Implement environment variable loading
        # Map environment variables to config keys
        # Override config values with environment variables
        
        env_mappings = {
            "CHURN_DATA_PATH": ("data", "raw_data_path"),
            "CHURN_MODEL_PATH": ("models", "save_path"),
            "CHURN_RANDOM_STATE": ("models", "random_state"),
            "CHURN_DEBUG": ("dashboard", "debug"),
            "CHURN_LOG_LEVEL": ("logging", "level")
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert string values to appropriate types
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                
                self.config[section][key] = value
                logger.info(f"Config override from environment: {env_var} = {value}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Deep update of nested dictionaries.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration key (e.g., 'data.raw_data_path')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                value = value[key]
            
            return value
            
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration key
            value: Value to set
        """
        keys = key_path.split('.')
        config_dict = self.config
        
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]
        
        config_dict[keys[-1]] = value
    
    def get_data_paths(self) -> Dict[str, str]:
        """
        Get all data-related paths.
        
        Returns:
            Dictionary of data paths
        """
        return {
            "raw": self.get("data.raw_data_path"),
            "processed": self.get("data.processed_data_path"),
            "sample": self.get("data.sample_data_path")
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration settings.
        
        Returns:
            Dictionary of model settings
        """
        return self.config.get("models", {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """
        Get feature engineering configuration.
        
        Returns:
            Dictionary of feature settings
        """
        return self.config.get("features", {})
    
    def save_config(self, output_file: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_file: Path to output YAML file
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving config to {output_file}: {str(e)}")
            raise
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration settings.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # TODO: Implement configuration validation
            # Check required settings
            # Validate data types and ranges
            # Check file paths exist
            
            # Validate data paths
            data_paths = self.get_data_paths()
            for path_name, path_value in data_paths.items():
                if not os.path.exists(path_value):
                    validation_results["warnings"].append(f"Data path does not exist: {path_value}")
            
            # Validate model settings
            random_state = self.get("models.random_state")
            if not isinstance(random_state, int) or random_state < 0:
                validation_results["errors"].append("models.random_state must be a non-negative integer")
                validation_results["is_valid"] = False
            
            # Validate test size
            test_size = self.get("models.test_size")
            if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
                validation_results["errors"].append("models.test_size must be a float between 0 and 1")
                validation_results["is_valid"] = False
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["is_valid"] = False
        
        return validation_results
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(file={self.config_file}, sections={list(self.config.keys())})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Config(config_file='{self.config_file}', config={self.config})"


# Global configuration instance
config = Config()
