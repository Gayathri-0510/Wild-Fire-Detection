import yaml
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config(config_path="config/settings.yaml"):
    """Loads configuration from YAML file and expands env vars."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Replace environment variables in config
    if 'alerts' in config:
        for key, value in config['alerts'].items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config['alerts'][key] = os.getenv(env_var, value)
                
    return config

# Global config instance
CONFIG = load_config()
