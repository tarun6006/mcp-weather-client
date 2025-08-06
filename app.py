import os
import json
import requests
import time
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from requests.exceptions import RequestException
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_print_token(token_name, token_value):
    """Safely print token information without exposing the actual value"""
    if not token_value:
        print(f"❌ {token_name} not configured")
        return False
    
    # Show only first 4 and last 4 characters for debugging
    masked_token = f"{token_value[:4]}...{token_value[-4:]}" if len(token_value) > 8 else "***"
    print(f"✅ {token_name} configured: {masked_token}")
    return True

def safe_debug_print(message, sensitive_data=None):
    """Print debug information safely, masking sensitive data"""
    if sensitive_data:
        # Mask sensitive data if provided
        if isinstance(sensitive_data, str) and len(sensitive_data) > 8:
            masked = f"{sensitive_data[:4]}...{sensitive_data[-4:]}"
        else:
            masked = "***"
        print(f"🔍 {message}: {masked}")
    else:
        print(f"🔍 {message}")

load_dotenv()

# Slack configuration
SLACK_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# MCP Server configuration - use individual components
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "5001"))
MCP_SERVER_PROTOCOL = os.getenv("MCP_SERVER_PROTOCOL", "http")
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "/mcp")

# Construct MCP server URL from components
MCP_SERVER_URL = f"{MCP_SERVER_PROTOCOL}://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}{MCP_SERVER_PATH}"

# Gemini AI configuration - Define variables first
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")  # Use Gemini 2.5 Pro only

# Debug: Print configuration on startup
print(f"MCP Server Configuration:")
print(f"  Host: {MCP_SERVER_HOST}")
print(f"  Port: {MCP_SERVER_PORT}")
print(f"  Protocol: {MCP_SERVER_PROTOCOL}")
print(f"  Path: {MCP_SERVER_PATH}")
print(f"  Full URL: {MCP_SERVER_URL}")

print(f"\nGemini Configuration:")
print(f"  Model: {GEMINI_MODEL}")
safe_print_token("GOOGLE_API_KEY", GOOGLE_API_KEY)

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Use optimized configuration for Gemini 2.5 Pro
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
        "max_output_tokens": 300,
    }
    model = genai.GenerativeModel(GEMINI_MODEL, generation_config=generation_config)
else:
    model = None

# Initialize Slack clients with safe logging
print(f"\nSlack Configuration:")
safe_print_token("SLACK_BOT_TOKEN", SLACK_TOKEN)
safe_print_token("SLACK_SIGNING_SECRET", SLACK_SECRET)

slack = WebClient(token=SLACK_TOKEN)
verifier = SignatureVerifier(SLACK_SECRET)

class GeminiMCPClient:
    """MCP client that uses Gemini LLM to process natural language and call tools"""
    
    def __init__(self, server_url, gemini_model=None):
        self.server_url = server_url
        self.request_id = 0
        self.model = gemini_model
        
        # Available MCP tools schema for Gemini - THESE MUST BE USED
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
        """Enhanced fallback method to extract location from user input"""
        import re
        
        # Clean the input
        text = user_input.lower().strip()
        
        # Remove common weather-related words and bot mentions
        stop_words = ['weather', 'forecast', 'temperature', 'what', 'is', 'the', 'in', 'for', 'at', 'how', 'whats', "what's"]
        words = text.split()
        
        # Remove @mentions and stop words
        location_words = []
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
            if not word.startswith('@') and clean_word not in stop_words and clean_word:
                location_words.append(clean_word)
        
        if location_words:
            location = " ".join(location_words).strip()
            
            # Check for ZIP code (5 digits)
            zip_match = re.search(r'\b\d{5}\b', location)
            if zip_match:
                return {"zip_code": zip_match.group()}
            
            # Otherwise treat as city
            if location:
                return {"city": location.title()}  # Capitalize for better API results
        
        return None

    def _format_weather_response(self, tool_result, location_args, user_input):
        """Format weather responses to be user-friendly and include requested location"""
        
        # Extract the requested location for display
        requested_location = location_args.get("city", location_args.get("zip_code", "your location"))
        
        # Check if it's an error response
        if "error" in tool_result.lower() or "not found" in tool_result.lower():
            # Handle different types of errors with user-friendly messages
            if "location not found" in tool_result.lower() or "not found" in tool_result.lower():
                return f"I couldn't find weather information for '{requested_location}'. Please check the spelling or try a different city name or ZIP code."
            elif "could not resolve" in tool_result.lower():
                return f"I'm having trouble finding '{requested_location}'. Could you try with a different city name or a 5-digit ZIP code?"
            elif "timeout" in tool_result.lower() or "network" in tool_result.lower():
                return f"I'm having trouble getting weather data for '{requested_location}' right now. Please try again in a moment."
            else:
                # Generic error fallback
                return f"I'm unable to get weather information for '{requested_location}' at the moment. Please try a different location or try again later."
        
        # For successful responses, make them more conversational
        if ":" in tool_result and any(weather_word in tool_result.lower() for weather_word in ["sunny", "cloudy", "rain", "snow", "clear", "partly", "mostly", "°f", "°c"]):
            # It's a successful weather response - make it more polite
            return f"Here's the weather for {requested_location}: {tool_result}"
        
        # Fallback for other successful responses
        return f"Weather update for {requested_location}: {tool_result}"

    def process_natural_language(self, user_input):
        """Use Gemini to process natural language and determine tool calls"""
        
        # First try enhanced fallback method
        fallback_location = self._extract_location_fallback(user_input)
        
        if not self.model:
            # No Gemini model available - use fallback only
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
                print(f"⚠️ Gemini API rate limit hit, falling back to direct processing")
                if fallback_location:
                    tool_result = self._call_mcp_tool("get_weather", fallback_location)
                    return self._format_weather_response(tool_result, fallback_location, user_input)
                return "Weather service is temporarily busy. Please specify a clear city name or zip code."
            else:
                # For other errors, try fallback
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

# Add global error handler for better debugging
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
    
    print(f"🚨 UNHANDLED EXCEPTION:")
    print(f"   Type: {error_details['error_type']}")
    print(f"   Message: {error_details['error_message']}")
    print(f"   Method: {error_details['request_method']}")
    print(f"   Path: {error_details['request_path']}")
    print(f"   Request Data: {error_details['request_data']}")
    print(f"   Full Traceback:")
    print(error_details['traceback'])
    
    # Return appropriate error response
    if request and request.path.startswith('/slack'):
        # For Slack endpoints, always return 200 to avoid retries
        return jsonify({
            "error": "Internal server error",
            "message": "The bot encountered an unexpected error. Please try again.",
            "error_id": error_details['error_type']
        }), 200
    else:
        # For other endpoints, return 500
        return jsonify({
            "error": "Internal server error", 
            "message": str(e),
            "error_type": error_details['error_type']
        }), 500

@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events with improved signature validation and challenge response"""
    
    # Get raw payload and headers
    payload = request.get_data()
    headers = request.headers
    
    print(f"🔍 Slack events endpoint called")
    print(f"📥 Raw payload length: {len(payload)}")
    print(f"📋 Headers: {dict(headers)}")
    
    # Parse JSON data first to check request type
    try:
        data = request.get_json(force=True)
        if not data:
            print("❌ No JSON data received")
            return "No JSON data", 400
            
        print(f"📄 Parsed JSON: {data}")
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        return "Invalid JSON", 400
    
    # Validate mandatory Slack parameters
    if not data.get("type"):
        print("❌ Missing 'type' parameter in request")
        return "Missing type parameter", 400
    
    request_type = data.get("type")
    print(f"🔍 Request type: {request_type}")
    
    # Handle URL verification challenge FIRST (before signature verification)
    if request_type == "url_verification":
        challenge = data.get("challenge")
        token = data.get("token")
        
        print(f"🔍 URL verification challenge received")
        safe_debug_print("Challenge", challenge)
        safe_debug_print("Verification Token", token)
        
        # Validate challenge parameter exists
        if not challenge:
            print("❌ Missing challenge parameter")
            return "Missing challenge parameter", 400
        
        # For URL verification, Slack doesn't require signature validation
        # Return the challenge exactly as received
        response_data = {"challenge": challenge}
        print(f"🔍 Responding with: {response_data}")
        
        return jsonify(response_data), 200
    
    # For all other events, verify Slack signature
    print(f"🔐 Checking Slack configuration...")
    
    # Check if Slack integration is properly configured
    if not safe_print_token("SLACK_SIGNING_SECRET", SLACK_SECRET):
        return "Slack signing secret not configured", 500
    
    if not safe_print_token("SLACK_BOT_TOKEN", SLACK_TOKEN):
        return "Slack bot token not configured", 500
    
    if not verifier:
        print("❌ SignatureVerifier not initialized")
        return "Signature verifier not initialized", 500
    
    # Validate required headers for signature verification
    if 'X-Slack-Request-Timestamp' not in headers:
        print("❌ Missing X-Slack-Request-Timestamp header")
        return "Missing timestamp header", 400
    
    if 'X-Slack-Signature' not in headers:
        print("❌ Missing X-Slack-Signature header")
        return "Missing signature header", 400
    
    # Get timestamp and signature
    timestamp = headers.get('X-Slack-Request-Timestamp')
    signature = headers.get('X-Slack-Signature')
    
    print(f"🔍 Timestamp: {timestamp}")
    safe_debug_print("Signature", signature)
    print(f"🔍 Payload length: {len(payload)} bytes")
    
    # Check timestamp freshness (Slack requires within 5 minutes)
    try:
        request_time = int(timestamp)
        current_time = int(time.time())
        time_diff = abs(current_time - request_time)
        
        print(f"🕐 Request time: {request_time}")
        print(f"🕐 Current time: {current_time}")
        print(f"🕐 Time difference: {time_diff} seconds")
        
        if time_diff > 300:  # 5 minutes
            print(f"❌ Request too old: {time_diff} seconds")
            return "Request timestamp too old", 400
            
    except ValueError as e:
        print(f"❌ Invalid timestamp format: {e}")
        return "Invalid timestamp", 400
    
    # Verify signature
    try:
        print(f"🔐 Verifying signature...")
        is_valid = verifier.is_valid_request(payload, headers)
        
        if not is_valid:
            print("❌ Invalid Slack signature")
            print(f"   Expected signature calculated from:")
            print(f"   - Timestamp: {timestamp}")
            print(f"   - Payload: {payload.decode('utf-8', errors='replace')}")
            return "Invalid signature", 403
            
        print("✅ Slack signature verified successfully")
        
    except Exception as e:
        print(f"❌ Signature verification error: {e}")
        print(f"   Error type: {type(e).__name__}")
        safe_debug_print("SLACK_SIGNING_SECRET length", len(SLACK_SECRET) if SLACK_SECRET else 'None')
        return "Signature verification failed", 403
    
    # Handle event callbacks
    if request_type == "event_callback":
        event = data.get("event", {})
        
        if not event:
            print("❌ Missing event data in event_callback")
            return "Missing event data", 400
        
        event_type = event.get("type")
        print(f"🔍 Event type: {event_type}")
        
        if event_type == "app_mention":
            try:
                # Validate required event parameters
                user_text = event.get("text")
                channel = event.get("channel")
                user = event.get("user")
                
                if not user_text:
                    print("❌ Missing text in app_mention event")
                    return "", 200  # Return 200 to acknowledge receipt
                
                if not channel:
                    print("❌ Missing channel in app_mention event")
                    return "", 200
                
                print(f"📩 App mention from user {user} in channel {channel}")
                print(f"📝 Message: {user_text}")
                
                # Process the natural language request with Gemini + MCP
                resp = client.send({"text": user_text})
                content = resp.get("tool_result", {}).get("content", [])
                
                # Extract the response text
                if content and len(content) > 0:
                    result = content[0].get("text", "No weather data received")
                else:
                    result = "Error from MCP client - no content received"
                
                print(f"🤖 Sending response: {result}")
                
                # Send response back to Slack with error handling
                try:
                    if slack:
                        response = slack.chat_postMessage(channel=channel, text=result)
                        if response.get("ok"):
                            print(f"✅ Message sent successfully to {channel}")
                        else:
                            print(f"❌ Slack API error: {response.get('error', 'Unknown error')}")
                    else:
                        print(f"❌ Slack client not initialized")
                except Exception as slack_error:
                    print(f"❌ Failed to send Slack message: {slack_error}")
                    print(f"   Channel: {channel}")
                    print(f"   Message: {result[:100]}...")  # First 100 chars
                
            except RequestException as e:
                error_msg = f"Network error: {str(e)}"
                print(f"❌ {error_msg}")
                if channel:
                    slack.chat_postMessage(channel=channel, text=error_msg)
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(f"❌ {error_msg}")
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
                
                # Skip bot messages and system messages
                if subtype == "bot_message" or not user or not user_text:
                    print(f"⏭️ Skipping message: subtype={subtype}, user={user}, text_present={bool(user_text)}")
                    return "", 200
                
                # Skip messages from the bot itself
                bot_user_id = None
                try:
                    if slack:  # Ensure slack client exists
                        auth_response = slack.auth_test()
                        if auth_response and auth_response.get("ok"):
                            bot_user_id = auth_response.get("user_id")
                        else:
                            print(f"⚠️ Slack auth_test failed: {auth_response}")
                except Exception as auth_error:
                    print(f"⚠️ Failed to get bot user ID: {auth_error}")
                    # Continue without bot user ID - will still process messages
                
                if bot_user_id and user == bot_user_id:
                    print(f"⏭️ Skipping message from bot itself")
                    return "", 200
                
                if not channel:
                    print("❌ Missing channel in message event")
                    return "", 200
                
                print(f"💬 Message from user {user} in channel {channel} (type: {channel_type})")
                print(f"📝 Message: {user_text}")
                
                # For direct messages (channel_type = "im") or if bot is mentioned in text
                is_direct_message = channel_type == "im"
                is_bot_mentioned = bot_user_id and f"<@{bot_user_id}>" in user_text
                
                # Only respond to direct messages or when explicitly mentioned
                if is_direct_message or is_bot_mentioned:
                    print(f"🎯 Processing message (DM: {is_direct_message}, Mentioned: {is_bot_mentioned})")
                    
                    # Process the natural language request with Gemini + MCP
                    resp = client.send({"text": user_text})
                    content = resp.get("tool_result", {}).get("content", [])
                    
                    # Extract the response text
                    if content and len(content) > 0:
                        result = content[0].get("text", "No weather data received")
                    else:
                        result = "Error from MCP client - no content received"
                    
                    print(f"🤖 Sending response: {result}")
                    
                    # Send response back to Slack with error handling
                    try:
                        if slack:
                            response = slack.chat_postMessage(channel=channel, text=result)
                            if response.get("ok"):
                                print(f"✅ Message sent successfully to {channel}")
                            else:
                                print(f"❌ Slack API error: {response.get('error', 'Unknown error')}")
                        else:
                            print(f"❌ Slack client not initialized")
                    except Exception as slack_error:
                        print(f"❌ Failed to send Slack message: {slack_error}")
                        print(f"   Channel: {channel}")
                        print(f"   Message: {result[:100]}...")  # First 100 chars
                else:
                    print(f"⏭️ Ignoring message (not DM and bot not mentioned)")
                
            except RequestException as e:
                error_msg = f"Network error: {str(e)}"
                print(f"❌ {error_msg}")
                if channel:
                    slack.chat_postMessage(channel=channel, text=error_msg)
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(f"❌ {error_msg}")
                if channel:
                    slack.chat_postMessage(channel=channel, text=error_msg)
        
        else:
            print(f"⚠️ Unhandled event type: {event_type}")
    
    else:
        print(f"⚠️ Unhandled request type: {request_type}")
    
    # Always return 200 to acknowledge receipt
    return "", 200

if __name__ == "__main__":
    # Use port 5002 for local development, PORT environment variable for Cloud Run
    port = int(os.environ.get("PORT", 5002))
    
    # Determine if running locally or on Cloud Run
    is_cloud_run = os.environ.get("K_SERVICE") is not None
    environment = "Cloud Run" if is_cloud_run else "Local"
    
    print(f"Starting MCP client on port {port} ({environment})")
    app.run(host="0.0.0.0", port=port, debug=not is_cloud_run)
