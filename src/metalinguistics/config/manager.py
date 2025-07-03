"""
Configuration Manager for Metadiscourse Analysis System
Phase 3.7: Externalize rules/features/model parameters to configurable files
"""

import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

class ConfigManager:
    """
    Centralized configuration management for the metadiscourse analysis system
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configuration structure
        self.default_configs = {
            'patterns': 'metadiscourse_patterns.json',
            'models': 'model_config.json',
            'features': 'feature_config.json',
            'analysis': 'analysis_config.yaml'
        }
    
    def load_config(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            config_name: Name of configuration ('patterns', 'models', etc.)
            use_cache: Whether to use cached version
            
        Returns:
            Configuration dictionary
        """
        if use_cache and config_name in self.config_cache:
            return self.config_cache[config_name]
        
        config_file = self.default_configs.get(config_name)
        if not config_file:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            # Create default configuration if it doesn't exist
            self._create_default_config(config_name, config_path)
        
        # Load based on file extension
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        if use_cache:
            self.config_cache[config_name] = config
        
        return config
    
    def save_config(self, config_name: str, config: Dict[str, Any], 
                   backup: bool = True) -> None:
        """
        Save configuration to file
        
        Args:
            config_name: Name of configuration
            config: Configuration dictionary to save
            backup: Whether to create backup of existing file
        """
        config_file = self.default_configs.get(config_name)
        if not config_file:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        config_path = self.config_dir / config_file
        
        # Create backup if requested and file exists
        if backup and config_path.exists():
            backup_path = config_path.with_suffix(f'{config_path.suffix}.backup')
            config_path.rename(backup_path)
        
        # Save based on file extension
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # Update cache
        self.config_cache[config_name] = config
    
    def get_category_config(self, category: str) -> Dict[str, Any]:
        """Get configuration for a specific metadiscourse category"""
        patterns_config = self.load_config('patterns')
        categories = patterns_config.get('categories', {})
        
        if category not in categories:
            raise ValueError(f"Unknown category: {category}")
        
        return categories[category]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all configured metadiscourse categories"""
        patterns_config = self.load_config('patterns')
        return list(patterns_config.get('categories', {}).keys())
    
    def get_keywords_for_category(self, category: str) -> List[str]:
        """Get keywords for a specific category"""
        category_config = self.get_category_config(category)
        return category_config.get('keywords', [])
    
    def get_academic_vocabulary(self) -> Dict[str, List[str]]:
        """Get academic vocabulary configuration"""
        patterns_config = self.load_config('patterns')
        return patterns_config.get('academic_context', {})
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters configuration"""
        patterns_config = self.load_config('patterns')
        return patterns_config.get('model_parameters', {})
    
    def get_boundary_detection_config(self) -> Dict[str, Any]:
        """Get boundary detection configuration"""
        patterns_config = self.load_config('patterns')
        return patterns_config.get('boundary_detection', {})
    
    def get_deduplication_config(self) -> Dict[str, Any]:
        """Get deduplication configuration"""
        patterns_config = self.load_config('patterns')
        return patterns_config.get('deduplication', {})
    
    def get_calibration_config(self) -> Dict[str, Any]:
        """Get calibration configuration"""
        patterns_config = self.load_config('patterns')
        return patterns_config.get('calibration', {})
    
    def update_category_keywords(self, category: str, keywords: List[str]) -> None:
        """Update keywords for a category"""
        patterns_config = self.load_config('patterns', use_cache=False)
        
        if 'categories' not in patterns_config:
            patterns_config['categories'] = {}
        
        if category not in patterns_config['categories']:
            patterns_config['categories'][category] = {}
        
        patterns_config['categories'][category]['keywords'] = keywords
        self.save_config('patterns', patterns_config)
    
    def add_new_category(self, category: str, config: Dict[str, Any]) -> None:
        """Add a new metadiscourse category"""
        patterns_config = self.load_config('patterns', use_cache=False)
        
        if 'categories' not in patterns_config:
            patterns_config['categories'] = {}
        
        patterns_config['categories'][category] = config
        self.save_config('patterns', patterns_config)
    
    def export_config(self, output_path: str, format: str = 'json') -> None:
        """Export all configurations to a single file"""
        all_configs = {}
        
        for config_name in self.default_configs.keys():
            try:
                all_configs[config_name] = self.load_config(config_name)
            except Exception as e:
                print(f"Warning: Could not load {config_name}: {e}")
        
        output_path = Path(output_path)
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_configs, f, indent=2, ensure_ascii=False)
        elif format.lower() in ['yaml', 'yml']:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_configs, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_config(self, input_path: str) -> None:
        """Import configurations from a file"""
        input_path = Path(input_path)
        
        if input_path.suffix.lower() == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                all_configs = json.load(f)
        elif input_path.suffix.lower() in ['.yaml', '.yml']:
            with open(input_path, 'r', encoding='utf-8') as f:
                all_configs = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported import format: {input_path.suffix}")
        
        # Save each configuration
        for config_name, config_data in all_configs.items():
            if config_name in self.default_configs:
                self.save_config(config_name, config_data)
    
    def _create_default_config(self, config_name: str, config_path: Path) -> None:
        """Create default configuration file"""
        if config_name == 'models':
            default_config = {
                "ml_models": {
                    "default_model_path": "metadiscourse_model_balanced_5k.joblib",
                    "model_types": ["random_forest", "svm", "logistic_regression"],
                    "training_parameters": {
                        "test_size": 0.2,
                        "random_state": 42,
                        "cv_folds": 5
                    }
                },
                "spacy_config": {
                    "model_name": "en_core_web_trf",
                    "use_mps": True,
                    "batch_size": 100,
                    "max_length": 2000000
                }
            }
        elif config_name == 'features':
            default_config = {
                "feature_extraction": {
                    "lexical_features": ["marker_length", "marker_word_count", "is_capitalized", "has_punctuation"],
                    "syntactic_features": ["pos_tag", "dependency_relation", "head_pos", "syntactic_children_count"],
                    "contextual_features": ["sentence_position", "distance_to_sentence_start", "distance_to_sentence_end"],
                    "semantic_features": ["is_sentence_start", "is_sentence_end", "follows_punctuation", "precedes_punctuation"],
                    "academic_features": ["in_academic_verb_phrase", "academic_context_score"]
                },
                "feature_weights": {
                    "confidence": 0.4,
                    "specificity": 0.3,
                    "academic_context": 0.2,
                    "ml_prediction": 0.1
                }
            }
        elif config_name == 'analysis':
            default_config = {
                "analysis_modes": {
                    "default": {
                        "use_ml": True,
                        "confidence_threshold": 0.6,
                        "enable_deduplication": True,
                        "enable_calibration": True
                    },
                    "high_precision": {
                        "use_ml": True,
                        "confidence_threshold": 0.8,
                        "enable_deduplication": True,
                        "enable_calibration": False
                    },
                    "exploratory": {
                        "use_ml": True,
                        "confidence_threshold": 0.2,
                        "enable_deduplication": False,
                        "enable_calibration": True
                    }
                },
                "output_formats": ["json", "csv", "html"],
                "export_options": {
                    "include_confidence": True,
                    "include_features": False,
                    "include_context": True
                }
            }
        else:
            # Default empty config
            default_config = {}
        
        # Save the default configuration
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    def validate_config(self, config_name: str) -> Dict[str, Any]:
        """Validate configuration and return validation report"""
        config = self.load_config(config_name)
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if config_name == 'patterns':
            # Validate pattern configuration
            categories = config.get('categories', {})
            
            for category, cat_config in categories.items():
                if 'keywords' not in cat_config:
                    validation_report['errors'].append(f"Category '{category}' missing keywords")
                    validation_report['valid'] = False
                
                if 'weight' not in cat_config:
                    validation_report['warnings'].append(f"Category '{category}' missing weight, using default")
                
                keywords = cat_config.get('keywords', [])
                if not keywords:
                    validation_report['warnings'].append(f"Category '{category}' has no keywords")
        
        return validation_report

def test_config_manager():
    """Test the configuration manager"""
    print("Testing Configuration Manager...")
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Test loading patterns
    try:
        patterns = config_manager.load_config('patterns')
        print(f"✓ Loaded patterns config with {len(patterns.get('categories', {}))} categories")
        
        # Test category access
        categories = config_manager.get_all_categories()
        print(f"✓ Found categories: {categories}")
        
        # Test keyword access
        if 'transitions' in categories:
            keywords = config_manager.get_keywords_for_category('transitions')
            print(f"✓ Transitions keywords: {len(keywords)} items")
        
        # Test model parameters
        model_params = config_manager.get_model_parameters()
        print(f"✓ Model parameters loaded: {list(model_params.keys())}")
        
        # Test validation
        validation = config_manager.validate_config('patterns')
        print(f"✓ Config validation: {'PASS' if validation['valid'] else 'FAIL'}")
        if validation['warnings']:
            print(f"  Warnings: {len(validation['warnings'])}")
        
        # Test export
        config_manager.export_config('exported_config.json')
        print("✓ Exported configuration to exported_config.json")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_config_manager()