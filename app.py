#!/usr/bin/env python3
"""
MCP Weather Bot Client - Main Application

Lightweight Flask application that orchestrates various services to provide
weather information and mathematical calculations via Slack integration.
"""

import os
import logging
import atexit

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Reduce noise from external libraries
logging.getLogger('slack_sdk').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# Import services
from services.config_service import ConfigService
from services.weather_calculator_service import WeatherCalculatorService
from services.slack_webhook_service import SlackWebhookService
from utils.core_utility import safe_log_token

def create_app():
    """Create and configure the Flask application"""
    
    # Initialize configuration service
    config_service = ConfigService()
    config_service.log_configuration_summary()
    
    # Get configuration values
    config = config_service.get_config()
    weather_server_url = config_service.get_weather_server_url()
    calc_host, calc_port, calc_protocol = config_service.get_calc_server_config()
    slack_token, slack_secret = config_service.get_slack_credentials()
    gemini_model = config_service.get_gemini_model()
    
    # Load error and logging messages from config
    error_messages = config.get('error_messages', {})
    logging_messages = config.get('logging_messages', {})
    
    logger.info(logging_messages.get('app_startup', "Starting MCP Weather Bot Client..."))
    
    # Validate required configuration
    if not safe_log_token("SLACK_BOT_TOKEN", slack_token):
        logger.error(error_messages.get('slack_token_missing', "Slack bot token not configured. Please set SLACK_BOT_TOKEN environment variable."))
        return None
    
    if not safe_log_token("SLACK_SIGNING_SECRET", slack_secret):
        logger.error(error_messages.get('slack_secret_missing', "Slack signing secret not configured. Please set SLACK_SIGNING_SECRET environment variable."))
        return None
    
    # Initialize weather calculator service
    logger.info(logging_messages.get('weather_service_init', "Initializing weather calculator service..."))
    weather_calculator_service = WeatherCalculatorService(
        weather_server_url=weather_server_url,
        calc_server_host=calc_host,
        calc_server_port=calc_port,
        calc_server_protocol=calc_protocol,
        gemini_model=gemini_model,
        config=config
    )
    
    # Initialize Slack webhook service
    logger.info(logging_messages.get('slack_service_init', "Initializing Slack webhook service..."))
    slack_webhook_service = SlackWebhookService(
        config=config,
        slack_token=slack_token,
        slack_secret=slack_secret,
        weather_calculator_service=weather_calculator_service
    )
    
    # Get Flask app instance
    app = slack_webhook_service.get_app()
    
    # Setup cleanup function for graceful shutdown
    def cleanup():
        """Cleanup function called on application exit"""
        logger.info(logging_messages.get('app_shutdown', "Shutting down client..."))
        if hasattr(weather_calculator_service, 'calc_service'):
            weather_calculator_service.calc_service.disconnect()
            logger.info(logging_messages.get('calculator_disconnected', "Calculator MCP service disconnected"))
    
    atexit.register(cleanup)
    
    logger.info(logging_messages.get('app_initialized', "MCP Weather Bot Client initialized successfully"))
    return app

# Create the Flask application
app = create_app()

if __name__ == "__main__":
    if app is None:
        # Load error messages for main execution
        try:
            config_service = ConfigService()
            config = config_service.get_config()
            error_messages = config.get('error_messages', {})
            error_msg = error_messages.get('app_creation_failed', "Failed to create application. Exiting.")
        except:
            error_msg = "Failed to create application. Exiting."
        logger.error(error_msg)
        exit(1)
    
    # Get configuration service to load Flask config
    config_service = ConfigService()
    config = config_service.get_config()
    flask_config = config.get('flask_config', {})
    logging_messages = config.get('logging_messages', {})
    
    # Get runtime configuration
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logging.getLogger().setLevel(getattr(logging, log_level))
        logger.info(logging_messages.get('log_level_set', "Log level set to: {level}").format(level=log_level))
    
    # Load Flask configuration from config file
    host = flask_config.get('host', '0.0.0.0')
    port = int(os.environ.get("PORT", flask_config.get('default_port', 5002)))
    is_cloud_run = os.environ.get("K_SERVICE") is not None
    environment = "Cloud Run" if is_cloud_run else "Local"
    
    # Get environment-specific settings
    if is_cloud_run:
        env_config = flask_config.get('cloud_run', {})
    else:
        env_config = flask_config.get('local', {})
    
    debug = env_config.get('debug', not is_cloud_run)
    # Get Flask defaults from utilities config
    utilities_config = config.get('utilities', {})
    flask_defaults = utilities_config.get('flask_defaults', {})
    default_threaded = flask_defaults.get('threaded', True)
    
    threaded = env_config.get('threaded', default_threaded)
    use_reloader = env_config.get('use_reloader', not is_cloud_run)
    
    logger.info(logging_messages.get('app_starting', "Starting MCP Weather Bot on {host}:{port} ({environment})").format(host=host, port=port, environment=environment))
    logger.info(logging_messages.get('security_features_header', "Security Features Enabled:"))
    logger.info(logging_messages.get('security_feature_logging', "   - Secure logging with sensitive data masking"))
    logger.info(logging_messages.get('security_feature_dedup', "   - Enhanced deduplication and rate limiting"))
    logger.info(logging_messages.get('security_feature_validation', "   - Request age validation and signature verification"))
    
    # Run the Flask application with config-based settings
    app.run(host=host, port=port, debug=debug, threaded=threaded, use_reloader=use_reloader)
