# Define the base directory for files
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INFERE_DIR = BASE_DIR / 'model_config' / 'model_inference'
RUNTIME_DIR = BASE_DIR / 'model_config' / 'runtimes'
STORAGE_DIR = BASE_DIR / 'storage_config'
#S3_SECRET_YAML = BASE_DIR / 's3_seceret.yaml'
#SA_YAML = BASE_DIR / 'sa.yaml'
