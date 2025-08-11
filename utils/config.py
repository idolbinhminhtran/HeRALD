"""
Configuration management for HGAT-LDA model.
"""

import yaml
import argparse
from typing import Dict, Any
from pathlib import Path


class Config:
    """
    Configuration class for HGAT-LDA model.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict
        
        # Set attributes from config
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """
        Create configuration from command line arguments.
        Args:
            args: Command line arguments
        Returns:
            Config object
        """
        # Manually nest arguments to match the YAML structure
        config_dict = {
            'model': {
                'emb_dim': getattr(args, 'emb_dim', 64),
                'num_layers': getattr(args, 'num_layers', 2),
                'dropout': getattr(args, 'dropout', 0.5),
            },
            'training': {
                'lr': getattr(args, 'lr', 1e-3),
                'weight_decay': getattr(args, 'weight_decay', 1e-5),
                'num_epochs': getattr(args, 'num_epochs', 50),
                'batch_size': getattr(args, 'batch_size', 64),
                'val_split': getattr(args, 'val_split', 0.1),
                'early_stopping_patience': getattr(args, 'early_stopping_patience', 10),
            },
            'data': {
                'threshold': getattr(args, 'threshold', 0.0),
                'symmetric': getattr(args, 'symmetric', True),
                'data_dir': getattr(args, 'data_dir', 'Dataset'),
            },
            'evaluation': {
                'loocv_epochs': getattr(args, 'loocv_epochs', 30),
                'loocv_batch_size': getattr(args, 'batch_size', 256),
                'loocv_lr': getattr(args, 'lr', 1e-3),
            },
            'system': {
                'device': getattr(args, 'device', 'auto'),
                'seed': getattr(args, 'seed', 42),
                'num_workers': getattr(args, 'num_workers', 4),
            }
        }
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        result = {}
        for key, value in self.__dict__.items():
            if key == 'config':
                continue
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def save(self, config_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'emb_dim': 64,
            'num_layers': 2,
            'dropout': 0.5
        },
        'training': {
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'num_epochs': 50,
            'batch_size': 256,
            'val_split': 0.1,
            'early_stopping_patience': 10
        },
        'data': {
            'threshold': 0.0,
            'symmetric': True,
            'data_dir': 'Dataset'
        },
        'evaluation': {
            'loocv_epochs': 30,
            'loocv_batch_size': 256,
            'loocv_lr': 1e-3
        },
        'system': {
            'device': 'auto',  # 'auto', 'cuda', 'cpu'
            'seed': 42,
            'num_workers': 4
        }
    }


def create_config_file(config_path: str = 'configs/default.yaml'):
    """
    Create default configuration file.
    
    Args:
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = get_default_config()
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Created default configuration file: {config_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='HGAT-LDA Training and Evaluation')
    
    # Model arguments
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='Dataset', help='Data directory')
    parser.add_argument('--threshold', type=float, default=0.0, help='Edge threshold')
    parser.add_argument('--symmetric', action='store_true', help='Use symmetric edges')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # File arguments
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--model_path', type=str, help='Model save/load path')
    parser.add_argument('--results_path', type=str, default='results', help='Results directory')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'loocv'], 
                       default='train', help='Mode to run')
    parser.add_argument('--loocv_epochs', type=int, default=30, help='LOOCV training epochs')
    
    return parser.parse_args() 