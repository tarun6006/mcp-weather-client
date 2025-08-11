"""Logging configuration and utilities"""
import logging

def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('slack_sdk').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def safe_log_token(token_name, token_value):
    """
    Safely log sensitive tokens by masking most characters
    Shows first 4 and last 4 characters for debugging
    """
    logger = logging.getLogger(__name__)
    if token_value and len(token_value) > 8:
        masked = f"{token_value[:4]}...{token_value[-4:]}"
        logger.info(f"  {token_name}: {masked}")
    elif token_value:
        logger.info(f"  {token_name}: ***")
    else:
        logger.warning(f"  {token_name}: NOT SET")

def safe_debug_log(message, sensitive_data=None):
    """
    Log debug messages while masking sensitive data
    """
    logger = logging.getLogger(__name__)
    if sensitive_data:
        logger.debug(f"{message} [SENSITIVE DATA MASKED]")
    else:
        logger.debug(message)
