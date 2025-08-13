import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
import yaml
import re
import uuid
import threading
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from requests.exceptions import RequestException
import google.generativeai as genai
import sseclient

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

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

CONFIG = load_config()

class OptimizedHTTPClient:
    """HTTP client with connection pooling and retry logic"""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Configure connection pooling and retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504, 429],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20,      # Max connections per pool
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Keep-alive headers for better performance
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=30, max=100'
        })
    
    def post(self, url, **kwargs):
        """POST request with connection pooling"""
        return self.session.post(url, **kwargs)
    
    def get(self, url, **kwargs):
        """GET request with connection pooling"""
        return self.session.get(url, **kwargs)

# Global HTTP client instance
http_client = OptimizedHTTPClient()

def safe_log_token(token_name, token_value):
    """Safely log token information without exposing the actual value"""
    if not token_value:
        logger.error(f"{token_name} not configured")
        return False
    
    # Show only first 4 and last 4 characters for debugging
    masked_token = f"{token_value[:4]}...{token_value[-4:]}" if len(token_value) > 8 else "***"
    logger.info(f"{token_name} configured: {masked_token}")
    return True

def safe_debug_log(message, sensitive_data=None):
    """Log debug information safely, masking sensitive data"""
    if sensitive_data:
        # Mask sensitive data if provided
        if isinstance(sensitive_data, str) and len(sensitive_data) > 8:
            masked = f"{sensitive_data[:4]}...{sensitive_data[-4:]}"
        else:
            masked = "***"
        logger.debug(f"{message}: {masked}")
    else:
        logger.debug(message)

load_dotenv()

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

# Timeout Configuration from config.yaml
TIMEOUT_CONFIG = CONFIG.get('timeout_config', {})
SSE_CONNECTION_TIMEOUT = TIMEOUT_CONFIG.get('sse_connection_timeout', 30)
SSE_REQUEST_TIMEOUT = TIMEOUT_CONFIG.get('sse_request_timeout', 30)
SSE_RESPONSE_TIMEOUT = TIMEOUT_CONFIG.get('sse_response_timeout', 30)
HTTP_REQUEST_TIMEOUT = TIMEOUT_CONFIG.get('http_request_timeout', 15)
GEMINI_REQUEST_TIMEOUT = TIMEOUT_CONFIG.get('gemini_request_timeout', 30)

# Application Configuration
MESSAGE_EXPIRY_SECONDS = 300  # 5 minutes
MIN_REQUEST_INTERVAL = 3      # 3 seconds between requests per user
MAX_PROCESSED_MESSAGES = 1000 # Maximum messages to keep in memory
GREEN_TICK_EMOJI = "white_check_mark"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

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
logger.info("Timeout Configuration:")
logger.info(f"  SSE Connection: {SSE_CONNECTION_TIMEOUT}s")
logger.info(f"  SSE Request: {SSE_REQUEST_TIMEOUT}s") 
logger.info(f"  SSE Response: {SSE_RESPONSE_TIMEOUT}s")
logger.info(f"  HTTP Request: {HTTP_REQUEST_TIMEOUT}s")
logger.info(f"  Gemini Request: {GEMINI_REQUEST_TIMEOUT}s")

logger.info("Gemini Configuration:")
logger.info(f"  Model: {GEMINI_MODEL}")
safe_log_token("GOOGLE_API_KEY", GOOGLE_API_KEY)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Load Gemini configuration from config YAML
    gemini_config = CONFIG.get('gemini_config', {})
    
    # Generation configuration from config
    gen_config = gemini_config.get('generation_config', {})
    generation_config = {
        "temperature": gen_config.get('temperature', 0.1),
        "top_p": gen_config.get('top_p', 0.8),
        "top_k": gen_config.get('top_k', 20),
        "max_output_tokens": gen_config.get('max_output_tokens', 300),
    }
    
    # Safety settings from config - very permissive for harmless weather/math queries
    safety_settings = gemini_config.get('safety_settings', [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
    ])
    model = genai.GenerativeModel(
        GEMINI_MODEL, 
        generation_config=generation_config,
        safety_settings=safety_settings
    )
else:
    model = None
logger.info("Slack Configuration:")
safe_log_token("SLACK_BOT_TOKEN", SLACK_TOKEN)
safe_log_token("SLACK_SIGNING_SECRET", SLACK_SECRET)

slack = WebClient(token=SLACK_TOKEN)
verifier = SignatureVerifier(SLACK_SECRET)

class SSECalculatorClient:
    """SSE client for calculator MCP server with event-driven architecture"""
    
    def __init__(self, calc_server_host, calc_server_port, calc_server_protocol):
        self.calc_server_host = calc_server_host
        self.calc_server_port = calc_server_port
        self.calc_server_protocol = calc_server_protocol
        self.client_id = str(uuid.uuid4())
        self.sse_client = None
        self.response_events = {}  # request_id -> threading.Event
        self.response_data = {}    # request_id -> response_data
        self.connected = False
        self.lock = threading.Lock()
        
        # Build SSE connection URL
        port_str = f":{calc_server_port}" if str(calc_server_port) not in ["80", "443"] else ""
        self.sse_url = f"{calc_server_protocol}://{calc_server_host}{port_str}/sse/connect?client_id={self.client_id}"
        self.mcp_url = f"{calc_server_protocol}://{calc_server_host}{port_str}/sse/mcp"
        
        self._connect()
    
    def _connect(self):
        """Establish SSE connection"""
        try:
            logger.info(f"Connecting to calculator SSE server: {self.sse_url}")
            response = http_client.get(self.sse_url, stream=True, timeout=10)
            response.raise_for_status()
            
            self.sse_client = sseclient.SSEClient(response)
            self.connected = True
            
            # Start background thread to handle SSE messages
            self.sse_thread = threading.Thread(target=self._handle_sse_messages, daemon=True)
            self.sse_thread.start()
            
            logger.info(f"SSE connection established with calculator server (client_id: {self.client_id})")
            
        except Exception as e:
            logger.error(f"Failed to connect to calculator SSE server: {e}")
            self.connected = False
    
    def _handle_sse_messages(self):
        """Handle incoming SSE messages with event-driven architecture"""
        try:
            for event in self.sse_client.events():
                if not self.connected:
                    break
                
                logger.debug(f"SSE event: {event.event}, data: {event.data}")
                
                if event.event == "connected":
                    try:
                        data = json.loads(event.data)
                        logger.debug(f"SSE connection confirmed: {data}")
                    except json.JSONDecodeError:
                        logger.debug("SSE connection confirmed (non-JSON data)")
                
                elif event.event == "message":
                    try:
                        # This is the actual MCP response
                        data = json.loads(event.data)
                        request_id = data.get("id")
                        if request_id:
                            with self.lock:
                                # Store response data
                                self.response_data[request_id] = data
                                # Signal waiting thread
                                if request_id in self.response_events:
                                    self.response_events[request_id].set()
                                    logger.debug(f"Signaled event for request {request_id}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse SSE message: {event.data}, error: {e}")
                
                elif event.event == "heartbeat":
                    try:
                        data = json.loads(event.data)
                        logger.debug(f"SSE heartbeat received: {data.get('timestamp')}")
                    except json.JSONDecodeError:
                        logger.debug("SSE heartbeat received")
                
                else:
                    logger.debug(f"Unknown SSE event type: {event.event}")
                    
        except Exception as e:
            logger.error(f"SSE message handler error: {e}")
            self.connected = False
    
    def call_tool(self, tool_name, arguments):
        """Make MCP tool call via SSE with event-driven waiting"""
        if not self.connected:
            return "Error: Not connected to calculator SSE server"
        
        request_id = str(uuid.uuid4())
        
        # Create event for this request
        event = threading.Event()
        with self.lock:
            self.response_events[request_id] = event
        
        # Prepare MCP request (simplified format for SSE endpoint)
        mcp_request = {
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            response = http_client.post(
                self.mcp_url,
                json=mcp_request,
                headers={
                    "Content-Type": "application/json",
                    "X-Client-ID": self.client_id
                },
                timeout=SSE_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # Wait for event (no CPU polling!) - configurable timeout for SSE responses
            if event.wait(timeout=SSE_RESPONSE_TIMEOUT):
                with self.lock:
                    response_data = self.response_data.pop(request_id, None)
                    self.response_events.pop(request_id, None)
                    
                if response_data:
                    if "result" in response_data:
                        # Handle calculator server response format
                        result = response_data["result"]
                        if isinstance(result, dict) and "result" in result:
                            # For calculator results like {"result": {"result": 5}}
                            return str(result["result"])
                        else:
                            # For other results
                            return str(result)
                    elif "error" in response_data:
                        error = response_data["error"]
                        if isinstance(error, dict) and "message" in error:
                            return f"Error: {error['message']}"
                        else:
                            return f"Error: {error}"
                    else:
                        return "Unknown error occurred"
                else:
                    return "Error: No response data received"
            else:
                # Cleanup on timeout
                with self.lock:
                    self.response_events.pop(request_id, None)
                    self.response_data.pop(request_id, None)
                return f"Error: Timeout waiting for SSE response ({SSE_RESPONSE_TIMEOUT}s)"
            
        except requests.exceptions.ConnectionError:
            # Cleanup on error
            with self.lock:
                self.response_events.pop(request_id, None)
            return f"Error: Could not connect to calculator SSE server"
        except requests.exceptions.Timeout:
            with self.lock:
                self.response_events.pop(request_id, None)
            return f"Error: Timeout connecting to calculator SSE server"
        except Exception as e:
            with self.lock:
                self.response_events.pop(request_id, None)
            return f"Error calling calculator SSE server: {str(e)}"
    
    def disconnect(self):
        """Disconnect from SSE server"""
        self.connected = False
        if self.sse_client:
            try:
                self.sse_client.close()
            except:
                pass


class GeminiMCPClient:
    """MCP client that uses Gemini LLM to process natural language and call tools"""
    
    def __init__(self, weather_server_url, calc_server_host, calc_server_port, calc_server_protocol, gemini_model=None):
        self.weather_server_url = weather_server_url
        self.calc_server_host = calc_server_host
        self.calc_server_port = calc_server_port
        self.calc_server_protocol = calc_server_protocol
        self.request_id = 0
        self.model = gemini_model
        
        # Initialize SSE client for calculator
        self.calc_sse_client = SSECalculatorClient(calc_server_host, calc_server_port, calc_server_protocol)
        
        # Load tools schema from configuration
        self.tools_schema = CONFIG.get('mcp_tools', {})
        
        # Load client messages and constants
        self.client_messages = CONFIG.get('client_messages', {})
        self.error_messages = self.client_messages.get('error_messages', {})
        self.response_messages = self.client_messages.get('response_messages', {})
        self.tool_categories = self.client_messages.get('tool_categories', {})
        self.calculator_tools = self.tool_categories.get('all_calculator_tools', [])
    
    def _call_mcp_tool(self, tool_name, arguments):
        """Make direct call to appropriate MCP server based on tool"""
        self.request_id += 1
        
        # Determine which server to use based on tool
        tool_info = self.tools_schema.get(tool_name, {})
        server_type = tool_info.get("server", "weather")
        
        if server_type == "calculator":
            # Use SSE for calculator tools
            logger.debug(f"Calling {tool_name} via SSE on calculator server")
            return self.calc_sse_client.call_tool(tool_name, arguments)
        else:
            # Use HTTP for weather tools
            server_url = self.weather_server_url
            logger.debug(f"Calling {tool_name} via HTTP on weather server: {server_url}")
            
            payload = {
                "jsonrpc": "2.0",
                "id": str(self.request_id),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            try:
                response = http_client.post(
                    server_url, 
                    json=payload, 
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "result" in result:
                    return result["result"]["content"][0]["text"]
                elif "error" in result:
                    return f"Error: {result['error']['message']}"
                else:
                    return "Unknown error occurred"
                    
            except requests.exceptions.ConnectionError:
                return f"Error: Could not connect to weather MCP server at {server_url}"
            except requests.exceptions.Timeout:
                return f"Error: Timeout connecting to weather MCP server"
            except Exception as e:
                return f"Error calling weather MCP server: {str(e)}"
    
    def _extract_location_fallback(self, user_input):
        """Extract location from user input using configured patterns"""
        
        text = user_input.lower().strip()
        logger.debug(f"Extracting location from: '{text}'")
        
        patterns = CONFIG['location_extraction']['patterns']
        
        for pattern_config in patterns:
            pattern = pattern_config['pattern']
            pattern_name = pattern_config['name']
            
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                logger.debug(f"Pattern '{pattern_name}' matched: '{location}'")
                
                if re.match(r'^\d{5}$', location):
                    return {"zip_code": location}
                
                location = self._clean_location_name(location)
                if location:
                    return {"city": location}
        
        logger.debug("Using fallback extraction...")
        
        text = re.sub(r'<@[^>]+>', '', text)
        stop_words = set(CONFIG['location_extraction']['stop_words'])
        
        words = text.split()
        location_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\sÀ-ÿ]', '', word).lower()
            
            if (clean_word not in stop_words and 
                clean_word and 
                not word.startswith('@') and
                not clean_word.isdigit() and
                len(clean_word) > 1):
                
                location_words.append(word.strip('.,!?'))
        
        if location_words:
            location = " ".join(location_words).strip()
            logger.debug(f"Fallback extracted: '{location}'")
            
            zip_match = re.search(r'\b\d{5}\b', location)
            if zip_match:
                return {"zip_code": zip_match.group()}
            
            location = self._clean_location_name(location)
            if location:
                return {"city": location}
        
        logger.debug("No location found")
        return None
    
    def _clean_location_name(self, location):
        """Clean and validate location name"""
        if not location:
            return None
            
        location = ' '.join(location.split()).title()
        
        unwanted = CONFIG['location_extraction']['unwanted_words']
        words = location.split()
        cleaned_words = [word for word in words if word not in unwanted]
        
        if cleaned_words:
            result = ' '.join(cleaned_words)
            logger.debug(f"Cleaned location: '{result}'")
            return result if len(result) > 1 else None
        
        return None

    def _format_response(self, tool_result, tool_name, arguments, user_input):
        """Format responses to be user-friendly for both weather and calculation tools"""
        
        if tool_name == "get_weather":
            # Weather response formatting
            requested_location = arguments.get("city", arguments.get("zip_code", "your location"))
            
            if "error" in tool_result.lower() or "not found" in tool_result.lower():
                if "location not found" in tool_result.lower() or "not found" in tool_result.lower():
                    return f"I couldn't find weather information for '{requested_location}'. Please check the spelling or try a different city name or ZIP code."
                elif "could not resolve" in tool_result.lower():
                    return f"I'm having trouble finding '{requested_location}'. Could you try with a different city name or a 5-digit ZIP code?"
                elif "timeout" in tool_result.lower() or "network" in tool_result.lower():
                    return f"I'm having trouble getting weather data for '{requested_location}' right now. Please try again in a moment."
                else:
                    return f"I'm unable to get weather information for '{requested_location}' at the moment. Please try a different location or try again later."
            
            if ":" in tool_result and any(weather_word in tool_result.lower() for weather_word in ["sunny", "cloudy", "rain", "snow", "clear", "partly", "mostly", "°f", "°c"]):
                return f"Here's the weather for {requested_location}: {tool_result}"
            
            return f"Weather update for {requested_location}: {tool_result}"
            
        elif tool_name in ["add", "subtract", "multiply", "divide", "power"]:
            # Calculator response formatting
            if "error" in tool_result.lower():
                return f"I encountered an error with the calculation: {tool_result}"
            
            # The calculator server already provides nicely formatted results like "16384 + 86413 = 102797"
            return f"Here's the calculation result: {tool_result}"
            
        else:
            # Generic response formatting
            return tool_result

    def process_natural_language(self, user_input):
        """Process natural language using Gemini and call appropriate tools"""
        
        fallback_location = self._extract_location_fallback(user_input)
        
        if not self.model:
            if fallback_location:
                tool_result = self._call_mcp_tool("get_weather", fallback_location)
                return self._format_response(tool_result, "get_weather", fallback_location, user_input)
            error_msg = self.error_messages.get('empty_input', 
                "Please specify a city or zip code for weather information or a mathematical calculation.")
            return error_msg
        
        # Load system prompt from configuration
        system_prompts = CONFIG.get('system_prompts', {})
        system_prompt_template = system_prompts.get('main_system_prompt', 
            'You are an AI assistant. Use the available MCP tools: {tools_schema}')
        system_prompt = system_prompt_template.format(tools_schema=json.dumps(self.tools_schema, indent=2))

        try:
            # First attempt with full system prompt
            response = self.model.generate_content(f"{system_prompt}\n\nUser request: {user_input}")
            
            # Check if response was blocked by safety filters
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else 'Unknown'
                logger.warning(f"Gemini response blocked by safety filters. User input: '{user_input}'. Finish reason: {finish_reason}")
                
                # Try with a simpler, less detailed prompt as fallback
                fallback_config = CONFIG.get('client_messages', {}).get('fallback_prompts', {})
                simple_prompt_template = fallback_config.get('simple_prompt_template', 
                    'Convert this to a tool call:\n\n{user_input}\n\nJSON response:')
                simple_prompt = simple_prompt_template.format(user_input=user_input)

                logger.info(f"Attempting fallback with simpler prompt for: {user_input}")
                
                try:
                    fallback_response = self.model.generate_content(simple_prompt)
                    if fallback_response.candidates and fallback_response.candidates[0].content.parts:
                        logger.info(f"Fallback prompt succeeded for: {user_input}")
                        response = fallback_response  # Use the fallback response
                    else:
                        logger.error(f"Both main and fallback prompts failed for: {user_input}")
                        
                        # Final fallback: simple pattern matching for obvious cases
                        user_lower = user_input.lower()
                        pattern_config = CONFIG.get('client_messages', {}).get('pattern_matching', {})
                        
                        # Check for obvious math patterns
                        math_patterns = pattern_config.get('math_patterns', ['calculate', 'what is'])
                        if any(pattern in user_lower for pattern in math_patterns):
                            # Extract the math expression (remove bot mention and common prefixes)
                            math_expression = user_input
                            # Remove bot mention
                            if '<@' in math_expression:
                                math_expression = ' '.join([word for word in math_expression.split() if not word.startswith('<@')])
                            # Remove common prefixes
                            prefixes = pattern_config.get('cleanup_prefixes', ['hey', 'what is'])
                            for prefix in prefixes:
                                if math_expression.lower().startswith(prefix):
                                    math_expression = math_expression[len(prefix):].strip()
                            
                            logger.info(f"Using pattern matching fallback for math: '{math_expression}'")
                            return self._call_mcp_tool('parse_expression', {'expression': math_expression.strip()})
                        
                        # Check for weather patterns
                        weather_patterns = pattern_config.get('weather_patterns', ['weather'])
                        if any(pattern in user_lower for pattern in weather_patterns):
                            # Try to extract city name (basic pattern)
                            words = user_input.split()
                            # Look for words that might be cities (after common words)
                            skip_words = pattern_config.get('skip_words', ['hey', 'weather', 'bot'])
                            for word in words:
                                clean_word = word.strip('.,?!').replace('<@', '').replace('>', '')
                                if len(clean_word) > 2 and clean_word.lower() not in skip_words and not clean_word.startswith('U'):
                                    logger.info(f"Using pattern matching fallback for weather in: '{clean_word}'")
                                    return self._call_mcp_tool('get_weather', {'city': clean_word})
                        
                        # Use configured error message
                        error_msg = self.error_messages.get('safety_fallback_help', 
                            "I had trouble processing your request about '{user_input}'. Please try rephrasing.")
                        return {
                            "tool_result": {
                                "content": [{
                                    "text": error_msg.format(user_input=user_input)
                                }]
                            }
                        }
                except Exception as fallback_error:
                    logger.error(f"Fallback prompt also failed: {fallback_error}")
                    return {
                        "tool_result": {
                            "content": [{
                                "text": f"I had trouble processing your request about '{user_input}'. This seems like a harmless weather or math question. Please try asking: 'What's the weather in [city name]?' or 'Calculate [math expression]'."
                            }]
                        }
                    }
            
            gemini_response = response.text.strip()
            
            # Parse Gemini's JSON response
            try:
                parsed = json.loads(gemini_response)
                
                if "error" in parsed:
                    return parsed["error"]
                
                if "tool" in parsed and "args" in parsed:
                    tool_result = self._call_mcp_tool(parsed["tool"], parsed["args"])
                    
                    # Use improved formatting for all responses
                    return self._format_response(tool_result, parsed["tool"], parsed["args"], user_input)
                else:
                    # Fallback to direct processing if Gemini response is unclear
                    if fallback_location:
                        tool_result = self._call_mcp_tool("get_weather", fallback_location)
                        return self._format_response(tool_result, "get_weather", fallback_location, user_input)
                    return "I couldn't understand your request. Please ask for weather information for a US city/zip code or a mathematical calculation."
                    
            except json.JSONDecodeError:
                # Fallback to direct processing if JSON parsing fails
                if fallback_location:
                    tool_result = self._call_mcp_tool("get_weather", fallback_location)
                    return self._format_response(tool_result, "get_weather", fallback_location, user_input)
                # Try pattern matching as final fallback before giving up
                user_lower = user_input.lower()
                pattern_config = CONFIG.get('client_messages', {}).get('pattern_matching', {})
                
                # Check for obvious math patterns
                math_patterns = pattern_config.get('math_patterns', ['calculate', 'what is', '+', '-', '*', '/'])
                if any(pattern in user_lower for pattern in math_patterns):
                    # Extract the math expression (remove bot mention and common prefixes)
                    math_expression = user_input
                    # Remove bot mention
                    if '<@' in math_expression:
                        math_expression = ' '.join([word for word in math_expression.split() if not word.startswith('<@')])
                    # Remove common prefixes
                    prefixes = pattern_config.get('cleanup_prefixes', ['hey', 'what is'])
                    for prefix in prefixes:
                        if math_expression.lower().startswith(prefix):
                            math_expression = math_expression[len(prefix):].strip()
                    
                    logger.info(f"Using pattern matching fallback for math after JSON parse failure: '{math_expression}'")
                    tool_result = self._call_mcp_tool('parse_expression', {'expression': math_expression.strip()})
                    return self._format_response(tool_result, "parse_expression", {'expression': math_expression.strip()}, user_input)
                
                # Check for weather patterns
                weather_patterns = pattern_config.get('weather_patterns', ['weather'])
                if any(pattern in user_lower for pattern in weather_patterns):
                    # Try to extract city name (basic pattern)
                    words = user_input.split()
                    # Look for words that might be cities (after common words)
                    skip_words = pattern_config.get('skip_words', ['hey', 'weather', 'bot'])
                    for word in words:
                        clean_word = word.strip('.,?!').replace('<@', '').replace('>', '')
                        if len(clean_word) > 2 and clean_word.lower() not in skip_words and not clean_word.startswith('U'):
                            logger.info(f"Using pattern matching fallback for weather after JSON parse failure: '{clean_word}'")
                            tool_result = self._call_mcp_tool('get_weather', {'city': clean_word})
                            return self._format_response(tool_result, "get_weather", {'city': clean_word}, user_input)
                
                # Use configured error message
                error_msg = self.error_messages.get('processing_error', 
                    "I had trouble processing your request. Please ask for weather information for a US city/zip code or a mathematical calculation.")
                return error_msg
                
        except Exception as e:
            error_msg = str(e)
            # Handle rate limiting specifically
            if "429" in error_msg or "quota" in error_msg.lower():
                logger.warning("Gemini API rate limit hit, falling back to direct processing")
                if fallback_location:
                    tool_result = self._call_mcp_tool("get_weather", fallback_location)
                    return self._format_response(tool_result, "get_weather", fallback_location, user_input)
                return "AI service is temporarily busy. Please specify a clear city name/zip code or mathematical calculation."
            else:
                # For other errors, try fallback
                logger.error(f"Gemini API error: {error_msg}")
                if fallback_location:
                    tool_result = self._call_mcp_tool("get_weather", fallback_location)
                    return self._format_response(tool_result, "get_weather", fallback_location, user_input)
                return f"Sorry, I encountered an error processing your request. Please try again with a specific city name/ZIP code or mathematical calculation."
    
    def send(self, request_data):
        """Legacy method for backward compatibility"""
        if "tool" in request_data:
            # Direct tool call (for testing)
            result = self._call_mcp_tool(request_data["tool"], request_data["args"])
            # Format the result for better user experience
            formatted_result = self._format_response(result, request_data["tool"], request_data["args"], "")
            return {
                "tool_result": {
                    "content": [{"type": "text", "text": formatted_result}]
                }
            }
        else:
            # Natural language processing
            result = self.process_natural_language(request_data.get("text", ""))
            return {
                "tool_result": {
                    "content": [{"type": "text", "text": result}]
                }
            }

client = GeminiMCPClient(WEATHER_MCP_URL, CALC_MCP_HOST, CALC_MCP_PORT, CALC_MCP_PROTOCOL, model)

app = Flask(__name__)

# Cleanup function for graceful shutdown
import atexit
def cleanup():
    """Cleanup function called on application exit"""
    logger.info("Shutting down client...")
    if hasattr(client, 'calc_sse_client'):
        client.calc_sse_client.disconnect()
        logger.info("SSE calculator client disconnected")

atexit.register(cleanup)

# Simple stateless message processing (cloud-friendly)
# Note: This approach avoids in-memory persistence for better cloud portability
# In production, consider using external cache (Redis) if duplicate detection is critical

# Import message processing utilities
from utils.security import (
    is_message_processed, mark_message_processed, cleanup_old_messages,
    processed_messages
)

# Constants for backward compatibility with existing code
MESSAGE_EXPIRY_SECONDS = 300
MAX_PROCESSED_MESSAGES = 1000

def cleanup_expired_messages():
    """Clean up expired messages - wrapper for utils.security function"""
    cleanup_old_messages()

# ==========================================
# SLACK THREADING AND REACTION HELPERS
# ==========================================

GREEN_TICK_EMOJI = "white_check_mark"  # Green tick mark reaction name for processed messages

def check_message_already_processed(channel, message_ts):
    """Check if a message already has bot response (thread reply + green tick reaction)"""
    try:
        # Check for existing reactions on the message
        try:
            reactions_response = slack.reactions_get(channel=channel, timestamp=message_ts)
            
            if reactions_response.get("ok"):
                message_data = reactions_response.get("message", {})
                reactions = message_data.get("reactions", [])
                
                # Check if green tick reaction exists from the bot
                bot_user_id = get_bot_user_id()
                for reaction in reactions:
                    if reaction.get("name") == "white_check_mark":  # Green tick emoji name in Slack
                        users = reaction.get("users", [])
                        if bot_user_id in users:
                            logger.info(f"Message {message_ts} already processed (has bot's green tick reaction)")
                            return True
            else:
                error_msg = reactions_response.get('error', 'Unknown error')
                if error_msg == 'missing_scope':
                    error_message = CONFIG.get('client_messages', {}).get('error_messages', {}).get('reactions_read_missing', 
                        "Cannot check reactions - missing 'reactions:read' scope. Please add this scope to your Slack app.")
                    logger.error(error_message)
                else:
                    logger.warning(f"Failed to get reactions for message {message_ts}: {error_msg}")
                    
        except Exception as reaction_error:
            if 'missing_scope' in str(reaction_error):
                logger.error(f"Cannot check reactions - missing 'reactions:read' scope: {reaction_error}")
            else:
                logger.error(f"Error checking reactions for message {message_ts}: {reaction_error}")
        
        # Also check for thread replies from the bot
        try:
            replies_response = slack.conversations_replies(channel=channel, ts=message_ts)
            
            if replies_response.get("ok"):
                messages = replies_response.get("messages", [])
                bot_user_id = get_bot_user_id()
                
                # Skip the original message (first in the list) and check for bot replies
                for message in messages[1:]:
                    if message.get("user") == bot_user_id:
                        logger.info(f"Message {message_ts} already has bot thread reply")
                        return True
            else:
                logger.warning(f"Failed to get thread replies for message {message_ts}: {replies_response.get('error', 'Unknown error')}")
                
        except Exception as thread_error:
            logger.error(f"Error checking thread replies for message {message_ts}: {thread_error}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking if message already processed: {e}")
        return False

def is_successful_response(response_text):
    """Check if response indicates success (not an error or timeout)"""
    error_indicators = [
        "error", "Error", "ERROR",
        "timeout", "Timeout", "TIMEOUT", 
        "failed", "Failed", "FAILED",
        "sorry", "Sorry", "SORRY",
        "unable", "Unable", "UNABLE",
        "cannot", "Cannot", "CANNOT",
        "not found", "Not found", "NOT FOUND",
        "no data", "No data", "NO DATA",
        "try again", "Try again", "TRY AGAIN",
        "encountered an error", "internal error", "service unavailable"
    ]
    
    # Check if response contains error indicators
    response_lower = response_text.lower()
    for indicator in error_indicators:
        if indicator.lower() in response_lower:
            return False
    
    return True

def send_threaded_response_with_reaction(channel, thread_ts, response_text, user_id=None):
    """Send a threaded response and conditionally add green tick reaction for successful responses"""
    try:
        # Add user tag to response if user_id is provided
        if user_id:
            tagged_response = f"<@{user_id}> {response_text}"
        else:
            tagged_response = response_text
            
        # Send response in thread
        thread_response = slack.chat_postMessage(
            channel=channel,
            text=tagged_response,
            thread_ts=thread_ts
        )
        
        if thread_response.get("ok"):
            logger.info(f"Thread response sent successfully to {channel}")
            
            # Only add green tick reaction for successful responses
            if is_successful_response(response_text):
                try:
                    reaction_response = slack.reactions_add(
                        channel=channel,
                        timestamp=thread_ts,
                        name="white_check_mark"  # Green tick emoji name in Slack
                    )
                    
                    if reaction_response.get("ok"):
                        logger.info(f"Green tick reaction added to message {thread_ts} for successful response")
                    else:
                        error_msg = reaction_response.get('error', 'Unknown error')
                        if error_msg == 'missing_scope':
                            logger.error(f"Cannot add reaction - missing 'reactions:write' scope. Please add this scope to your Slack app.")
                        elif error_msg == 'already_reacted':
                            logger.debug(f"Reaction already exists on message {thread_ts}")
                        elif error_msg == 'no_reaction':
                            logger.warning(f"Invalid reaction name 'white_check_mark' for message {thread_ts}")
                        else:
                            logger.warning(f"Failed to add reaction to message {thread_ts}: {error_msg}")
                        
                except Exception as reaction_error:
                    if 'missing_scope' in str(reaction_error):
                        logger.error(f"Cannot add reaction - missing 'reactions:write' scope: {reaction_error}")
                    elif 'already_reacted' in str(reaction_error):
                        logger.debug(f"Reaction already exists on message {thread_ts}: {reaction_error}")
                    else:
                        logger.error(f"Error adding reaction to message {thread_ts}: {reaction_error}")
            else:
                logger.info(f"No green tick added for error/timeout response: {response_text[:50]}...")
            
            return True
        else:
            logger.error(f"Failed to send thread response: {thread_response.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending threaded response: {e}")
        return False

def handle_thread_spam_request(channel, thread_ts, user):
    """Handle requests made in existing threads by asking user to send new message"""
    spam_warning = (
        "Hi! I see you're asking a question in an existing thread. "
        "To keep our conversation organized and follow Slack etiquette, "
        "please send your new question as a fresh message (mention me with @weather-bot). "
        "This helps avoid thread spam and makes it easier for everyone to follow conversations. "
        "Thanks!"
    )
    
    try:
        slack.chat_postMessage(
            channel=channel,
            text=spam_warning,
            thread_ts=thread_ts
        )
        logger.info(f"Sent thread spam warning to user {user} in channel {channel}")
        return True
    except Exception as e:
        logger.error(f"Error sending thread spam warning: {e}")
        return False

def is_request_in_existing_thread(event):
    """Check if the message is part of an existing thread"""
    return event.get("thread_ts") is not None

def get_bot_user_id():
    """Get the bot's user ID from Slack"""
    try:
        if slack:
            auth_response = slack.auth_test()
            if auth_response and auth_response.get("ok"):
                return auth_response.get("user_id")
        return None
    except Exception as e:
        logger.warning(f"Failed to get bot user ID: {e}")
        return None

user_last_request = {}
MIN_REQUEST_INTERVAL = 1  # Reduced from 2 to 1 second for better responsiveness

def is_rate_limited(user_id):
    """Check if user is making requests too quickly"""
    current_time = time.time()
    last_request_time = user_last_request.get(user_id, 0)
    
    if current_time - last_request_time < MIN_REQUEST_INTERVAL:
        return True
    
    user_last_request[user_id] = current_time
    return False

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler to catch and log all errors"""
    import traceback
    
    error_details = {
        'error_type': type(e).__name__,
        'error_message': str(e),
        'traceback': traceback.format_exc(),
        'request_method': request.method if request else 'Unknown',
        'request_path': request.path if request else 'Unknown',
        'request_data': None
    }
    
    # Safely get request data (avoid logging sensitive information)
    try:
        if request and request.is_json:
            json_data = request.get_json()
            # Remove any potential sensitive fields before logging
            if isinstance(json_data, dict):
                safe_data = {k: v for k, v in json_data.items() 
                           if k.lower() not in ['token', 'secret', 'key', 'password', 'signature']}
                error_details['request_data'] = safe_data
            else:
                error_details['request_data'] = 'JSON data (content masked for security)'
        elif request and request.data:
            # Only log first 200 chars and mask any potential tokens
            raw_data = request.data.decode('utf-8', errors='replace')[:200]
            error_details['request_data'] = raw_data
    except:
        error_details['request_data'] = 'Could not parse request data'
    
    logger.error("UNHANDLED EXCEPTION:")
    logger.error(f"   Type: {error_details['error_type']}")
    logger.error(f"   Message: {error_details['error_message']}")
    logger.error(f"   Method: {error_details['request_method']}")
    logger.error(f"   Path: {error_details['request_path']}")
    logger.error(f"   Request Data: {error_details['request_data']}")
    logger.error("   Full Traceback:")
    logger.error(error_details['traceback'])
    
    if request and request.path.startswith('/slack'):
        return jsonify({
            "error": "Internal server error",
            "message": "The bot encountered an unexpected error. Please try again.",
            "error_id": error_details['error_type']
        }), 200
    else:
        return jsonify({
            "error": "Internal server error", 
            "message": str(e),
            "error_type": error_details['error_type']
        }), 500

@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events with signature validation and challenge response"""
    
    payload = request.get_data()
    headers = request.headers
    
    logger.info("Slack events endpoint called")
    logger.debug(f"Raw payload length: {len(payload)}")
    logger.debug(f"Headers: {dict(headers)}")
    
    try:
        data = request.get_json(force=True)
        if not data:
            logger.error("No JSON data received")
            return "No JSON data", 400
            
        logger.debug(f"Parsed JSON: {data}")
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return "Invalid JSON", 400
    
    if not data.get("type"):
        logger.error("Missing 'type' parameter in request")
        return "Missing type parameter", 400
    
    request_type = data.get("type")
    logger.info(f"Request type: {request_type}")
    
    if request_type == "url_verification":
        challenge = data.get("challenge")
        token = data.get("token")
        
        logger.info("URL verification challenge received")
        safe_debug_log("Challenge", challenge)
        safe_debug_log("Verification Token", token)
        
        if not challenge:
            logger.error("Missing challenge parameter")
            return "Missing challenge parameter", 400
        
        response_data = {"challenge": challenge}
        logger.debug(f"Responding with: {response_data}")
        
        return jsonify(response_data), 200
    
    logger.debug("Checking Slack configuration...")
    
    if not safe_log_token("SLACK_SIGNING_SECRET", SLACK_SECRET):
        return "Slack signing secret not configured", 500
    
    if not safe_log_token("SLACK_BOT_TOKEN", SLACK_TOKEN):
        return "Slack bot token not configured", 500
    
    if not verifier:
        logger.error("SignatureVerifier not initialized")
        return "Signature verifier not initialized", 500
    
    if 'X-Slack-Request-Timestamp' not in headers:
        logger.error("Missing X-Slack-Request-Timestamp header")
        return "Missing timestamp header", 400
    
    if 'X-Slack-Signature' not in headers:
        logger.error("Missing X-Slack-Signature header")
        return "Missing signature header", 400
    
    timestamp = headers.get('X-Slack-Request-Timestamp')
    signature = headers.get('X-Slack-Signature')
    
    logger.debug(f"Timestamp: {timestamp}")
    safe_debug_log("Signature", signature)
    logger.debug(f"Payload length: {len(payload)} bytes")
    
    try:
        request_time = int(timestamp)
        current_time = int(time.time())
        time_diff = abs(current_time - request_time)
        
        logger.debug(f"Request time: {request_time}")
        logger.debug(f"Current time: {current_time}")
        logger.debug(f"Time difference: {time_diff} seconds")
        
        if time_diff > 300:
            logger.warning(f"Request too old: {time_diff} seconds")
            return "Request timestamp too old", 400
            
    except ValueError as e:
        logger.error(f"Invalid timestamp format: {e}")
        return "Invalid timestamp", 400
    
    try:
        logger.debug("Verifying signature...")
        is_valid = verifier.is_valid_request(payload, headers)
        
        if not is_valid:
            logger.error("Invalid Slack signature")
            logger.error(f"   Expected signature calculated from:")
            logger.error(f"   - Timestamp: {timestamp}")
            logger.error(f"   - Payload: {payload.decode('utf-8', errors='replace')}")
            return "Invalid signature", 403
            
        logger.debug("Slack signature verified successfully")
        
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        safe_debug_log("SLACK_SIGNING_SECRET length", len(SLACK_SECRET) if SLACK_SECRET else 'None')
        return "Signature verification failed", 403
    
    if request_type == "event_callback":
        event = data.get("event", {})
        
        if not event:
            logger.error("Missing event data in event_callback")
            return "Missing event data", 400
        
        event_type = event.get("type")
        logger.info(f"Event type: {event_type}")
        
        if event_type == "app_mention":
            try:
                user_text = event.get("text")
                channel = event.get("channel")
                user = event.get("user")
                event_ts = event.get("ts")
                thread_ts = event.get("thread_ts")  # Check if this is in a thread
                
                if not user_text:
                    logger.error("Missing text in app_mention event")
                    return "", 200
                
                if not channel:
                    logger.error("Missing channel in app_mention event")
                    return "", 200
                
                if not user:
                    logger.error("Missing user in app_mention event")
                    return "", 200
                
                if not event_ts:
                    logger.error("Missing timestamp in app_mention event")
                    return "", 200
                
                # Check if this is a request in an existing thread
                if thread_ts:
                    logger.info(f"App mention in existing thread from user {user}")
                    handle_thread_spam_request(channel, thread_ts, user)
                    return "", 200
                
                # Check if message already has bot response (reaction + thread reply)
                if check_message_already_processed(channel, event_ts):
                    logger.info(f"App mention already processed (has green tick reaction): {event_ts}")
                    return "", 200
                
                if is_rate_limited(user):
                    logger.warning(f"Rate limited user {user} - ignoring request")
                    return "", 200
                
                message_key = f"mention_{channel}_{user}_{event_ts}_{hash(user_text)}"
                
                if is_message_processed(message_key):
                    logger.info(f"App mention already processed: {message_key}")
                    return "", 200
                
                try:
                    event_time = float(event_ts)
                    current_time = time.time()
                    message_age = current_time - event_time
                    
                    if message_age > MESSAGE_EXPIRY_SECONDS:
                        logger.info(f"Message too old ({message_age:.1f}s), ignoring: {message_key}")
                        return "", 200
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid event timestamp: {event_ts}, error: {e}")
                
                mark_message_processed(message_key)
                
                logger.info(f"App mention from user {user} in channel {channel}")
                logger.debug(f"Message: {user_text}")
                logger.debug(f"Message age: {message_age:.1f}s" if 'message_age' in locals() else "Message age: unknown")
                
                # Process the natural language request with Gemini + MCP
                resp = client.send({"text": user_text})
                content = resp.get("tool_result", {}).get("content", [])
                
                # Extract the response text
                if content and len(content) > 0:
                    result = content[0].get("text", "No weather data received")
                else:
                    result = "Error from MCP client - no content received"
                
                logger.info(f"Sending threaded response: {result}")
                
                # Send response in thread with green tick reaction
                success = send_threaded_response_with_reaction(channel, event_ts, result, user)
                
                if success:
                    # Mark message as processed after successful response
                    mark_message_processed(message_key)
                    logger.debug(f"Marked app mention as processed: {message_key}")
                else:
                    # Fallback to regular message if threading fails
                    try:
                        if slack:
                            response = slack.chat_postMessage(channel=channel, text=result)
                            if response.get("ok"):
                                logger.info(f"Fallback message sent successfully to {channel}")
                            else:
                                logger.error(f"Slack API error: {response.get('error', 'Unknown error')}")
                        else:
                            logger.error("Slack client not initialized")
                    except Exception as slack_error:
                        logger.error(f"Slack API error: {slack_error}")
                
            except Exception as e:
                error_msg = f"Error processing app_mention: {str(e)}"
                logger.error(error_msg)
                if 'channel' in locals() and 'event_ts' in locals():
                    # Try to send error in thread, fallback to regular message
                    try:
                        user_for_error = user if 'user' in locals() else None
                        send_threaded_response_with_reaction(channel, event_ts, error_msg, user_for_error)
                    except:
                        try:
                            slack.chat_postMessage(channel=channel, text=error_msg)
                        except:
                            pass
        
        elif event_type == "message":
            try:
                # Handle direct messages and channel messages
                user_text = event.get("text")
                channel = event.get("channel")
                user = event.get("user")
                channel_type = event.get("channel_type")
                subtype = event.get("subtype")
                event_ts = event.get("ts")  # Slack timestamp for deduplication
                thread_ts = event.get("thread_ts")  # Check if this is in a thread
                
                # Skip bot messages and system messages
                if subtype == "bot_message" or not user or not user_text:
                    logger.debug(f"Skipping message: subtype={subtype}, user={user}, text_present={bool(user_text)}")
                    return "", 200
                
                # Skip messages from the bot itself
                bot_user_id = None
                try:
                    if slack:  # Ensure slack client exists
                        auth_response = slack.auth_test()
                        if auth_response and auth_response.get("ok"):
                            bot_user_id = auth_response.get("user_id")
                        else:
                            logger.warning(f"Slack auth_test failed: {auth_response}")
                except Exception as auth_error:
                    logger.warning(f"Failed to get bot user ID: {auth_error}")
                    # Continue without bot user ID - will still process messages
                
                if bot_user_id and user == bot_user_id:
                    logger.debug("Skipping message from bot itself")
                    return "", 200
                
                if not channel:
                    logger.error("Missing channel in message event")
                    return "", 200
                
                # For direct messages (channel_type = "im") or if bot is mentioned in text
                is_direct_message = channel_type == "im"
                is_bot_mentioned = bot_user_id and f"<@{bot_user_id}>" in user_text
                
                # IMPORTANT: Skip messages with bot mentions in channels since they're handled by app_mention
                # Only process direct messages (DMs) or channel messages WITHOUT bot mentions
                if is_bot_mentioned and not is_direct_message:
                    logger.debug("Skipping channel message with bot mention (handled by app_mention)")
                    return "", 200
                
                # Only respond to direct messages
                if is_direct_message:
                    # Check if this is a request in an existing thread
                    if thread_ts:
                        logger.info(f"DM in existing thread from user {user}")
                        handle_thread_spam_request(channel, thread_ts, user)
                        return "", 200
                    
                    # Check if message already has bot response (reaction + thread reply)
                    if check_message_already_processed(channel, event_ts):
                        logger.info(f"DM already processed (has green tick reaction): {event_ts}")
                        return "", 200
                    
                    # Check for rate limiting
                    if is_rate_limited(user):
                        logger.warning(f"Rate limited user {user} - ignoring DM")
                        return "", 200
                    
                    # Create unique message key for deduplication (include message content hash)
                    message_key = f"message_{channel}_{user}_{event_ts}_{hash(user_text)}"
                    
                    if is_message_processed(message_key):
                        logger.info(f"Message already processed: {message_key}")
                        return "", 200
                    
                    # Additional check for message age using event timestamp
                    try:
                        event_time = float(event_ts)
                        current_time = time.time()
                        message_age = current_time - event_time
                        
                        if message_age > MESSAGE_EXPIRY_SECONDS:
                            logger.info(f"DM too old ({message_age:.1f}s), ignoring: {message_key}")
                            return "", 200
                            
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid event timestamp: {event_ts}, error: {e}")
                        # Continue processing but log the issue
                    
                    mark_message_processed(message_key)
                    
                    logger.info(f"Processing DM from user {user} in channel {channel}")
                    logger.debug(f"Message: {user_text}")
                    logger.debug(f"Message age: {message_age:.1f}s" if 'message_age' in locals() else "Message age: unknown")
                    
                    # Process the natural language request with Gemini + MCP
                    resp = client.send({"text": user_text})
                    content = resp.get("tool_result", {}).get("content", [])
                    
                    # Extract the response text
                    if content and len(content) > 0:
                        result = content[0].get("text", "No weather data received")
                    else:
                        result = "Error from MCP client - no content received"
                    
                    logger.info(f"Sending threaded DM response: {result}")
                    
                    # Send response in thread with green tick reaction
                    success = send_threaded_response_with_reaction(channel, event_ts, result, user)
                    
                    if success:
                        # Mark message as processed after successful response
                        mark_message_processed(message_key)
                        logger.debug(f"Marked DM as processed: {message_key}")
                    else:
                        # Fallback to regular message if threading fails
                        try:
                            if slack:
                                response = slack.chat_postMessage(channel=channel, text=result)
                                if response.get("ok"):
                                    logger.info(f"Fallback DM sent successfully to {channel}")
                                else:
                                    logger.error(f"Slack API error: {response.get('error', 'Unknown error')}")
                            else:
                                logger.error("Slack client not initialized")
                        except Exception as slack_error:
                            logger.error(f"Slack API error: {slack_error}")
                else:
                    logger.debug("Ignoring message (not a DM)")
                
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                logger.error(error_msg)
                if 'channel' in locals() and 'event_ts' in locals():
                    # Try to send error in thread, fallback to regular message
                    try:
                        user_for_error = user if 'user' in locals() else None
                        send_threaded_response_with_reaction(channel, event_ts, error_msg, user_for_error)
                    except:
                        try:
                            slack.chat_postMessage(channel=channel, text=error_msg)
                        except:
                            pass
        
        else:
            logger.warning(f"Unhandled event type: {event_type}")
    
    else:
        logger.warning(f"Unhandled request type: {request_type}")
    
    # Always return 200 to acknowledge receipt
    return "", 200

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint that also performs cleanup"""
    cleanup_expired_messages()  # Clean up old messages
    
    return jsonify({
        "status": "healthy",
        "processed_messages_count": len(processed_messages),
        "user_rate_limits_count": len(user_last_request),
        "weather_mcp_server_url": WEATHER_MCP_URL,
        "calc_mcp_server": {
            "host": CALC_MCP_HOST,
            "port": CALC_MCP_PORT,
            "protocol": CALC_MCP_PROTOCOL,
            "transport": "SSE",
            "connected": client.calc_sse_client.connected if hasattr(client, 'calc_sse_client') else False
        },
        "timestamp": time.time()
    }), 200

if __name__ == "__main__":
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logging.getLogger().setLevel(getattr(logging, log_level))
        logger.info(f"Log level set to: {log_level}")
    
    port = int(os.environ.get("PORT", 5002))
    is_cloud_run = os.environ.get("K_SERVICE") is not None
    environment = "Cloud Run" if is_cloud_run else "Local"
    
    logger.info(f"Starting MCP Weather Bot on port {port} ({environment})")
    logger.info("Security Features Enabled:")
    logger.info("   - Secure logging with sensitive data masking")
    logger.info("   - Enhanced deduplication and rate limiting")
    logger.info("   - Request age validation and signature verification")
    logger.info(f"Configuration:")
    logger.info(f"   - Message expiry: {MESSAGE_EXPIRY_SECONDS} seconds")
    logger.info(f"   - Rate limit: {MIN_REQUEST_INTERVAL} seconds between requests per user")
    logger.info(f"   - Max processed messages: {MAX_PROCESSED_MESSAGES}")
    
    app.run(host="0.0.0.0", port=port, debug=not is_cloud_run)
