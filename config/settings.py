"""Configuration and environment variable management"""
import os
import yaml
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

# Load configuration
CONFIG = load_config()

# Slack configuration
SLACK_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# Weather MCP Server Configuration
WEATHER_MCP_HOST = os.getenv("WEATHER_MCP_HOST", "localhost")
WEATHER_MCP_PORT = int(os.getenv("WEATHER_MCP_PORT", "5001"))
WEATHER_MCP_PROTOCOL = os.getenv("WEATHER_MCP_PROTOCOL", "http")
WEATHER_MCP_PATH = os.getenv("WEATHER_MCP_PATH", "/mcp")
WEATHER_MCP_URL = f"{WEATHER_MCP_PROTOCOL}://{WEATHER_MCP_HOST}:{WEATHER_MCP_PORT}{WEATHER_MCP_PATH}"

# Calculator MCP Server Configuration
# Get calculator host and clean it if it includes protocol
_calc_host_raw = os.getenv("CALC_MCP_HOST", "localhost")
# Remove protocol if present (e.g., "https://example.com" -> "example.com")
if "://" in _calc_host_raw:
    CALC_MCP_HOST = _calc_host_raw.split("://", 1)[1]
else:
    CALC_MCP_HOST = _calc_host_raw
CALC_MCP_PORT = int(os.getenv("CALC_MCP_PORT", "5003"))
CALC_MCP_PROTOCOL = os.getenv("CALC_MCP_PROTOCOL", "http")
CALC_MCP_PATH = os.getenv("CALC_MCP_PATH", "/mcp")
CALC_MCP_URL = f"{CALC_MCP_PROTOCOL}://{CALC_MCP_HOST}:{CALC_MCP_PORT}{CALC_MCP_PATH}"

# Google AI Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Application Configuration
MESSAGE_EXPIRY_SECONDS = 300  # 5 minutes
MIN_REQUEST_INTERVAL = 3      # 3 seconds between requests per user
MAX_PROCESSED_MESSAGES = 1000 # Maximum messages to keep in memory
GREEN_TICK_EMOJI = "white_check_mark"

def log_configuration():
    """Log configuration information"""
    logger.info("MCP Server Configuration:")
    logger.info("Weather MCP Server:")
    logger.info(f"  Host: {WEATHER_MCP_HOST}")
    logger.info(f"  Port: {WEATHER_MCP_PORT}")
    logger.info(f"  Protocol: {WEATHER_MCP_PROTOCOL}")
    logger.info(f"  Path: {WEATHER_MCP_PATH}")
    logger.info(f"  Full URL: {WEATHER_MCP_URL}")
    logger.info("Calculator MCP Server:")
    logger.info(f"  Host: {CALC_MCP_HOST}")
    logger.info(f"  Port: {CALC_MCP_PORT}")
    logger.info(f"  Protocol: {CALC_MCP_PROTOCOL}")
    logger.info(f"  Path: {CALC_MCP_PATH}")
    logger.info(f"  Full URL: {CALC_MCP_URL}")
    logger.info("Gemini Configuration:")
    logger.info(f"  Model: {GEMINI_MODEL}")
