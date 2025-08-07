import os
import json
import requests
import time
import logging
import yaml
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from requests.exceptions import RequestException
import google.generativeai as genai

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

CONFIG = load_config()

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

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "5001"))
MCP_SERVER_PROTOCOL = os.getenv("MCP_SERVER_PROTOCOL", "http")
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "/mcp")

MCP_SERVER_URL = f"{MCP_SERVER_PROTOCOL}://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}{MCP_SERVER_PATH}"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

logger.info("MCP Server Configuration:")
logger.info(f"  Host: {MCP_SERVER_HOST}")
logger.info(f"  Port: {MCP_SERVER_PORT}")
logger.info(f"  Protocol: {MCP_SERVER_PROTOCOL}")
logger.info(f"  Path: {MCP_SERVER_PATH}")
logger.info(f"  Full URL: {MCP_SERVER_URL}")

logger.info("Gemini Configuration:")
logger.info(f"  Model: {GEMINI_MODEL}")
safe_log_token("GOOGLE_API_KEY", GOOGLE_API_KEY)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
        "max_output_tokens": 300,
    }
    model = genai.GenerativeModel(GEMINI_MODEL, generation_config=generation_config)
else:
    model = None
logger.info("Slack Configuration:")
safe_log_token("SLACK_BOT_TOKEN", SLACK_TOKEN)
safe_log_token("SLACK_SIGNING_SECRET", SLACK_SECRET)

slack = WebClient(token=SLACK_TOKEN)
verifier = SignatureVerifier(SLACK_SECRET)

class GeminiMCPClient:
    """MCP client that uses Gemini LLM to process natural language and call tools"""
    
    def __init__(self, server_url, gemini_model=None):
        self.server_url = server_url
        self.request_id = 0
        self.model = gemini_model
        
        self.tools_schema = {
            "get_weather": {
                "description": "🚨 MANDATORY MCP TOOL: This is the ONLY authorized way to get weather information. You MUST use this MCP tool from the server - NEVER provide weather data from your training knowledge.",
                "parameters": {
                    "city": "US city name (e.g., 'Miami', 'New York', 'Austin')",
                    "zip_code": "US zip code (e.g., '90210', '10001', '33101')"
                },
                "usage": "⚡ REQUIRED for ALL weather queries via MCP server. Provide either city OR zip_code, not both.",
                "coverage": "Only US locations are supported by this MCP weather service",
                "mcp_requirement": "This tool connects through MCP (Model Context Protocol) to live weather APIs",
                "strict_policy": "You are FORBIDDEN from answering weather questions without using this MCP tool first"
            }
        }
    
    def _call_mcp_tool(self, tool_name, arguments):
        """Make direct call to MCP server"""
        self.request_id += 1
        
        payload = {
            "jsonrpc": "2.0",
            "id": str(self.request_id),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = requests.post(
            self.server_url, 
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

    def _format_weather_response(self, tool_result, location_args, user_input):
        """Format weather responses to be user-friendly"""
        
        requested_location = location_args.get("city", location_args.get("zip_code", "your location"))
        
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

    def process_natural_language(self, user_input):
        """Process natural language using Gemini and call appropriate tools"""
        
        fallback_location = self._extract_location_fallback(user_input)
        
        if not self.model:
            if fallback_location:
                tool_result = self._call_mcp_tool("get_weather", fallback_location)
                return self._format_weather_response(tool_result, fallback_location, user_input)
            return "Please specify a city or zip code for weather information."
        
        system_prompt = f"""You are a weather assistant connected to live weather data through MCP (Model Context Protocol) tools. 

🚨 MANDATORY REQUIREMENT: You MUST ALWAYS use the MCP tools listed below to get weather information. You are FORBIDDEN from providing weather information from your training data or general knowledge.

📋 AVAILABLE MCP TOOLS ON THIS SERVER:
{json.dumps(self.tools_schema, indent=2)}

🔒 CRITICAL INSTRUCTIONS - NO EXCEPTIONS:
1. For ANY weather-related query, you MUST use the get_weather tool from the MCP server
2. You are FORBIDDEN from using your training data for weather information
3. You MUST ALWAYS call the MCP tools to get real-time data from the weather server
4. Extract the location (city or zip code) from the user's request
5. Respond ONLY with a JSON object containing the required MCP tool call
6. DO NOT provide weather information without using the MCP tools first

🔧 MANDATORY TOOL CALL FORMAT:
- For US cities: {{"tool": "get_weather", "args": {{"city": "CityName"}}}}
- For zip codes: {{"tool": "get_weather", "args": {{"zip_code": "12345"}}}}

❌ If request is not weather-related or location cannot be determined:
- Respond with: {{"error": "Please specify a US city or zip code for weather information. I must use the MCP tools available on this server."}}

📝 EXAMPLES - Notice how EVERY weather query MUST use MCP tools:

User requests with explicit MCP tool instructions:
- "Use MCP tools to check weather in Miami" → {{"tool": "get_weather", "args": {{"city": "Miami"}}}} (MUST use MCP tool)
- "Check MCP server for weather in 90210" → {{"tool": "get_weather", "args": {{"zip_code": "90210"}}}} (MUST use MCP tool)
- "Use available MCP tools for Seattle forecast" → {{"tool": "get_weather", "args": {{"city": "Seattle"}}}} (MUST use MCP tool)
- "Check weather using MCP tools for Austin" → {{"tool": "get_weather", "args": {{"city": "Austin"}}}} (MUST use MCP tool)

Natural language requests (still MUST use MCP tools):
- "What's the weather in Miami?" → {{"tool": "get_weather", "args": {{"city": "Miami"}}}} (MUST use MCP tool)
- "How's 90210?" → {{"tool": "get_weather", "args": {{"zip_code": "90210"}}}} (MUST use MCP tool)  
- "Is it raining in Seattle?" → {{"tool": "get_weather", "args": {{"city": "Seattle"}}}} (MUST use MCP tool)
- "Temperature in Boston" → {{"tool": "get_weather", "args": {{"city": "Boston"}}}} (MUST use MCP tool)

⚡ REMEMBER: You are connected to live weather APIs through MCP tools on this server. You MUST use these MCP tools for ALL weather queries - never rely on your training data.

Only respond with the JSON object, nothing else."""

        try:
            response = self.model.generate_content(f"{system_prompt}\n\nUser request: {user_input}")
            gemini_response = response.text.strip()
            
            # Parse Gemini's JSON response
            try:
                parsed = json.loads(gemini_response)
                
                if "error" in parsed:
                    return parsed["error"]
                
                if "tool" in parsed and "args" in parsed:
                    tool_result = self._call_mcp_tool(parsed["tool"], parsed["args"])
                    
                    # Use improved formatting for all responses
                    return self._format_weather_response(tool_result, parsed["args"], user_input)
                else:
                    # Fallback to direct processing if Gemini response is unclear
                    if fallback_location:
                        tool_result = self._call_mcp_tool("get_weather", fallback_location)
                        return self._format_weather_response(tool_result, fallback_location, user_input)
                    return "I couldn't understand your request. Please ask for weather information for a US city or zip code."
                    
            except json.JSONDecodeError:
                # Fallback to direct processing if JSON parsing fails
                if fallback_location:
                    tool_result = self._call_mcp_tool("get_weather", fallback_location)
                    return self._format_weather_response(tool_result, fallback_location, user_input)
                return "I had trouble processing your request. Please ask for weather information for a US city or zip code."
                
        except Exception as e:
            error_msg = str(e)
            # Handle rate limiting specifically
            if "429" in error_msg or "quota" in error_msg.lower():
                logger.warning("Gemini API rate limit hit, falling back to direct processing")
                if fallback_location:
                    tool_result = self._call_mcp_tool("get_weather", fallback_location)
                    return self._format_weather_response(tool_result, fallback_location, user_input)
                return "Weather service is temporarily busy. Please specify a clear city name or zip code."
            else:
                # For other errors, try fallback
                logger.error(f"Gemini API error: {error_msg}")
                if fallback_location:
                    tool_result = self._call_mcp_tool("get_weather", fallback_location)
                    return self._format_weather_response(tool_result, fallback_location, user_input)
                return f"Sorry, I encountered an error processing your request. Please try again with a specific city name or ZIP code."
    
    def send(self, request_data):
        """Legacy method for backward compatibility"""
        if "tool" in request_data:
            # Direct tool call (for testing)
            result = self._call_mcp_tool(request_data["tool"], request_data["args"])
            # Format the result for better user experience
            formatted_result = self._format_weather_response(result, request_data["args"], "")
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

client = GeminiMCPClient(MCP_SERVER_URL, model)

app = Flask(__name__)

processed_messages = {}
MAX_PROCESSED_MESSAGES = 1000
MESSAGE_EXPIRY_SECONDS = 300

def cleanup_expired_messages():
    """Remove expired messages from the processed set"""
    current_time = time.time()
    expired_keys = [
        key for key, timestamp in processed_messages.items()
        if current_time - timestamp > MESSAGE_EXPIRY_SECONDS
    ]
    for key in expired_keys:
        processed_messages.pop(key, None)
    
    if len(processed_messages) > MAX_PROCESSED_MESSAGES:
        sorted_items = sorted(processed_messages.items(), key=lambda x: x[1])
        for key, _ in sorted_items[:100]:
            processed_messages.pop(key, None)

def is_message_processed(message_key):
    """Check if a message has already been processed"""
    cleanup_expired_messages()
    return message_key in processed_messages

def mark_message_processed(message_key):
    """Mark a message as processed with current timestamp"""
    processed_messages[message_key] = time.time()

user_last_request = {}
MIN_REQUEST_INTERVAL = 2

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
                
                logger.info(f"Sending response: {result}")
                
                # Send response back to Slack with error handling
                try:
                    if slack:
                        response = slack.chat_postMessage(channel=channel, text=result)
                        if response.get("ok"):
                            logger.info(f"Message sent successfully to {channel}")
                        else:
                            logger.error(f"Slack API error: {response.get('error', 'Unknown error')}")
                    else:
                        logger.error("Slack client not initialized")
                except Exception as slack_error:
                    logger.error(f"Failed to send Slack message: {slack_error}")
                    logger.error(f"   Channel: {channel}")
                    logger.error(f"   Message: {result[:100]}...")  # First 100 chars
                
            except RequestException as e:
                error_msg = f"Network error: {str(e)}"
                logger.error(error_msg)
                if channel:
                    slack.chat_postMessage(channel=channel, text=error_msg)
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                logger.error(error_msg)
                if channel:
                    slack.chat_postMessage(channel=channel, text=error_msg)
        
        elif event_type == "message":
            try:
                # Handle direct messages and channel messages
                user_text = event.get("text")
                channel = event.get("channel")
                user = event.get("user")
                channel_type = event.get("channel_type")
                subtype = event.get("subtype")
                event_ts = event.get("ts")  # Slack timestamp for deduplication
                
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
                    
                    logger.info(f"Sending response: {result}")
                    
                    # Send response back to Slack with error handling
                    try:
                        if slack:
                            response = slack.chat_postMessage(channel=channel, text=result)
                            if response.get("ok"):
                                logger.info(f"Message sent successfully to {channel}")
                            else:
                                logger.error(f"Slack API error: {response.get('error', 'Unknown error')}")
                        else:
                            logger.error("Slack client not initialized")
                    except Exception as slack_error:
                        logger.error(f"Failed to send Slack message: {slack_error}")
                        logger.error(f"   Channel: {channel}")
                        logger.error(f"   Message: {result[:100]}...")  # First 100 chars
                else:
                    logger.debug("Ignoring message (not a DM)")
                
            except RequestException as e:
                error_msg = f"Network error: {str(e)}"
                logger.error(error_msg)
                if channel:
                    slack.chat_postMessage(channel=channel, text=error_msg)
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                logger.error(error_msg)
                if channel:
                    slack.chat_postMessage(channel=channel, text=error_msg)
        
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
        "mcp_server_url": MCP_SERVER_URL,
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
