import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()  

class BaseConfig():
    """基础配置，所有模型共用的部分"""
    LOGGING_LEVEL = 'INFO'  # DEBUG  INFO
    LOG_DIR = "logs"

    CHROME_DRIVER_PATH = os.getenv('CHROME_DRIVER_PATH')

    EMBEDDING_URL = os.getenv('EMBEDDING_URL')
    EMBEDDING_API = os.getenv('EMBEDDING_API')

    MLLM_URL = os.getenv('MLLM_URL')
    MLLM_API = os.getenv('MLLM_API')

    SUMMARY_URL = os.getenv('SUMMARY_URL')
    SUMMARY_API = os.getenv('SUMMARY_API')


class TextEmbedding3LargeConfig(BaseConfig):
    EMBEDDING_MODEL_NAME = "text-embedding-3-large"
    EMBEDDING_DIM = 3072
    EMBEDDING_MAX_TOKENS = 8192


class TextEmbedding3SmallConfig(BaseConfig):
    MODEL_NAME = "text-embedding-3-small"
    EMBEDDING_DIM = 1536
    MAX_TOKENS = 8192

class MLLMConfig(BaseConfig):
    MMLLM = "gpt-4o"  

class SUMMARYConfig(BaseConfig):
    SUMMARY_MODEL = "gpt-4o"  




