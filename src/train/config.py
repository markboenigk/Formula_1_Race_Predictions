"""Model configuration loader with Pydantic validation."""
import os
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model metadata configuration."""
    name: str
    type: str = "XGBRegressor"
    version: str = "1.0.0"


class HyperparametersConfig(BaseModel):
    """XGBoost hyperparameters."""
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    objective: str = "reg:squarederror"
    random_state: int = 42
    n_jobs: int = 1


class CVConfig(BaseModel):
    """Cross-validation settings."""
    strategy: str = "leave-one-season-out"
    min_precision_at_3: float = 0.0


class FeaturesConfig(BaseModel):
    """Feature configuration."""
    primary_feature: list[str] = Field(default_factory=lambda: ["qualifying_position"])
    numerical: list[str] = Field(default_factory=list)
    optional_numerical: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    use_qualifying_gaps: bool = False
    use_overtake_difficulty: bool = False


class TrainingConfig(BaseModel):
    """Training settings."""
    test_size: float = 0.2
    random_state: int = 42
    use_ranker: bool = False


class ModelSettingsConfig(BaseModel):
    """Complete model configuration from YAML."""
    model: ModelConfig
    hyperparameters: HyperparametersConfig
    cv: CVConfig
    features: FeaturesConfig
    training: TrainingConfig
    description: str = ""
    author: str = ""


def load_model_config(config_path: Union[str, Path]) -> ModelSettingsConfig:
    """Load and validate a model configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    
    return ModelSettingsConfig(**raw_config)


def get_config_path(config_name: str) -> Path:
    """Get the full path to a model config file.
    
    Args:
        config_name: Name like 'grid_only_v1' or 'grid_plus_quali_v1'
    
    Returns:
        Full path to the config file
    """
    # Look relative to project root
    project_root = Path(__file__).parent.parent.parent
    config_dir = project_root / "config" / "models"
    
    # Add .yaml extension if not present
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    
    return config_dir / config_name


def load_config_by_name(config_name: str) -> ModelSettingsConfig:
    """Load a model config by name.
    
    Args:
        config_name: Name like 'grid_only_v1' or full path
    
    Returns:
        Validated ModelSettingsConfig
    """
    config_path = get_config_path(config_name)
    return load_model_config(config_path)


# Convenience function for environment-based config loading
def load_config_from_env() -> ModelSettingsConfig:
    """Load config from MODEL_CONFIG environment variable.
    
    Expected format: MODEL_CONFIG=grid_only_v1
    """
    config_name = os.getenv("MODEL_CONFIG", "grid_only_v1")
    return load_config_by_name(config_name)
