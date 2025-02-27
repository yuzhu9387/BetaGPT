# 这个文件可以完全空着 

from .airtable_client import AirtableClient
from .llm_service import LLMService

# 版本信息
__version__ = '1.0.0'

# 导出的类和函数
__all__ = ['AirtableClient', 'LLMService']

# 包级别的配置
DEFAULT_CONFIG = {
    'timeout': 30,
    'retry_attempts': 3
}

# 初始化日志
import logging
logging.basicConfig(level=logging.INFO) 