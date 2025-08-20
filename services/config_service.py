#!/usr/bin/env python3
"""
Configuration Service

Centralized configuration loading and management for the MCP Weather Bot client.
Handles YAML config loading, environment detection, and server URL configuration.
"""

import os
import yaml
import logging
import google.generativeai as genai
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigService:
    """Service for managing application configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), '..', 'config', 'config.yaml'
        )
        
        # Load environment variables
        load_dotenv()
        
        # Load YAML configuration
        self.config = self._load_yaml_config()
        
        # Environment detection
        self.environment = self._detect_environment()
        
        # Server configuration
        self.weather_server_config = self._get_weather_server_config()
        self.calc_server_config = self._get_calc_server_config()
        
        # Timeout configuration
        self.timeout_config = self.config.get('timeout_config', {})
        
        # Slack configuration
        self.slack_config = self._get_slack_config()
        
        # Gemini configuration
        self.gemini_config = self._get_gemini_config()
        
        # Load logging messages from config
        self.logging_messages = self.config.get('logging_messages', {})
        
        logger.info(self.logging_messages.get('config_loaded', "Configuration loaded for {environment} environment").format(environment=self.environment))
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from: {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _detect_environment(self) -> str:
        """Detect if running in local development or cloud environment"""
        # Check for Google Cloud Run environment
        is_cloud_run = os.environ.get("K_SERVICE") is not None
        # Check for explicit environment setting
        env_mode = os.environ.get("ENVIRONMENT", "").lower()
        
        if env_mode == "local":
            return "local"
        elif env_mode == "cloud" or is_cloud_run:
            return "cloud"
        else:
            # Auto-detect: if no explicit cloud environment vars, assume local
            return "local" if not is_cloud_run else "cloud"
    
    def _get_weather_server_config(self) -> Dict[str, Any]:
        """Get weather MCP server configuration with environment-aware defaults"""
        if self.environment == "local":
            defaults = {
                "host": "localhost",
                "port": 5001,
                "protocol": "http"
            }
        else:
            defaults = {
                "host": "mcp-weather-server-876452898662.us-west1.run.app",
                "port": 443,
                "protocol": "https"
            }
        
        config = {
            "host": os.getenv("WEATHER_MCP_HOST", defaults["host"]),
            "port": int(os.getenv("WEATHER_MCP_PORT", defaults["port"])),
            "protocol": os.getenv("WEATHER_MCP_PROTOCOL", defaults["protocol"]),
            "path": os.getenv("WEATHER_MCP_PATH", "/mcp")
        }
        
        # Build URL
        if config["port"] in [80, 443]:
            config["url"] = f"{config['protocol']}://{config['host']}{config['path']}"
        else:
            config["url"] = f"{config['protocol']}://{config['host']}:{config['port']}{config['path']}"
        
        return config
    
    def _get_calc_server_config(self) -> Dict[str, Any]:
        """Get calculator MCP server configuration with environment-aware defaults"""
        if self.environment == "local":
            defaults = {
                "host": "localhost",
                "port": 5003,
                "protocol": "http"
            }
        else:
            defaults = {
                "host": "calc-mcp-server-876452898662.us-west1.run.app",
                "port": 443,
                "protocol": "https"
            }
        
        # Get host and clean it if it includes protocol
        host_raw = os.getenv("CALC_MCP_HOST", defaults["host"])
        if "://" in host_raw:
            host = host_raw.split("://", 1)[1]
        else:
            host = host_raw
        
        config = {
            "host": host,
            "port": int(os.getenv("CALC_MCP_PORT", defaults["port"])),
            "protocol": os.getenv("CALC_MCP_PROTOCOL", defaults["protocol"]),
            "path": os.getenv("CALC_MCP_PATH", "/mcp")
        }
        
        # Build URL
        if config["port"] in [80, 443]:
            config["url"] = f"{config['protocol']}://{config['host']}{config['path']}"
        else:
            config["url"] = f"{config['protocol']}://{config['host']}:{config['port']}{config['path']}"
        
        return config
    
    def _get_slack_config(self) -> Dict[str, Any]:
        """Get Slack configuration"""
        return {
            "token": os.getenv("SLACK_BOT_TOKEN"),
            "secret": os.getenv("SLACK_SIGNING_SECRET")
        }
    
    def _get_gemini_config(self) -> Dict[str, Any]:
        """Get Gemini configuration and initialize the model"""
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        
        config = {
            "api_key": api_key,
            "model_name": model_name,
            "model": None
        }
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                
                # Load Gemini configuration from YAML
                gemini_yaml_config = self.config.get('gemini_config', {})
                
                # Generation configuration
                gen_config = gemini_yaml_config.get('generation_config', {})
                generation_config = {
                    "temperature": gen_config.get('temperature', 0.1),
                    "top_p": gen_config.get('top_p', 0.8),
                    "top_k": gen_config.get('top_k', 20),
                    "max_output_tokens": gen_config.get('max_output_tokens', 300),
                }
                
                # Safety settings - very permissive for harmless weather/math queries
                safety_settings = gemini_yaml_config.get('safety_settings', [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                ])
                
                config["model"] = genai.GenerativeModel(
                    model_name, 
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                logger.info(f"Gemini model initialized: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                config["model"] = None
        else:
            logger.warning("Gemini API key not configured")
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full YAML configuration"""
        return self.config
    
    def get_environment(self) -> str:
        """Get the detected environment"""
        return self.environment
    
    def get_weather_server_url(self) -> str:
        """Get the weather server URL"""
        return self.weather_server_config["url"]
    
    def get_calc_server_config(self) -> Tuple[str, int, str]:
        """Get calculator server configuration as tuple (host, port, protocol)"""
        return (
            self.calc_server_config["host"],
            self.calc_server_config["port"],
            self.calc_server_config["protocol"]
        )
    
    def get_slack_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """Get Slack credentials as tuple (token, secret)"""
        return (self.slack_config["token"], self.slack_config["secret"])
    
    def get_gemini_model(self):
        """Get the initialized Gemini model"""
        return self.gemini_config["model"]
    
    def get_timeout_config(self) -> Dict[str, int]:
        """Get timeout configuration"""
        return {
            "sse_connection_timeout": self.timeout_config.get('sse_connection_timeout', 30),
            "sse_request_timeout": self.timeout_config.get('sse_request_timeout', 30),
            "sse_response_timeout": self.timeout_config.get('sse_response_timeout', 30),
            "http_request_timeout": self.timeout_config.get('http_request_timeout', 15),
            "gemini_request_timeout": self.timeout_config.get('gemini_request_timeout', 30)
        }
    
    def log_configuration_summary(self):
        """Log a summary of the current configuration"""
        logger.info("=== CONFIGURATION SUMMARY ===")
        logger.info(f"Environment: {self.environment}")
        logger.info("Weather MCP Server:")
        logger.info(f"  Host: {self.weather_server_config['host']}")
        logger.info(f"  Port: {self.weather_server_config['port']}")
        logger.info(f"  Protocol: {self.weather_server_config['protocol']}")
        logger.info(f"  Full URL: {self.weather_server_config['url']}")
        logger.info("Calculator MCP Server:")
        logger.info(f"  Host: {self.calc_server_config['host']}")
        logger.info(f"  Port: {self.calc_server_config['port']}")
        logger.info(f"  Protocol: {self.calc_server_config['protocol']}")
        logger.info(f"  Full URL: {self.calc_server_config['url']}")
        logger.info("Timeouts:")
        for key, value in self.get_timeout_config().items():
            logger.info(f"  {key}: {value}s")
        logger.info(f"Gemini Model: {self.gemini_config['model_name']}")
        logger.info(f"Gemini Available: {'Yes' if self.gemini_config['model'] else 'No'}")
        logger.info("===============================")
