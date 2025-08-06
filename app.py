import os
import json
import requests
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from requests.exceptions import RequestException
import google.generativeai as genai

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
print(f"  API Key Available: {'Yes' if GOOGLE_API_KEY else 'No'}")

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

    def process_natural_language(self, user_input):
        """Use Gemini to process natural language and determine tool calls"""
        
        # First try enhanced fallback method
        fallback_location = self._extract_location_fallback(user_input)
        
        if not self.model:
            # No Gemini model available - use fallback only
            if fallback_location:
                return self._call_mcp_tool("get_weather", fallback_location)
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
                    
                    # Use simple formatting instead of another Gemini call to avoid rate limits
                    if "error" in tool_result.lower():
                        return tool_result
                    else:
                        # Simple formatting without additional API calls
                        location_info = parsed["args"].get("city", parsed["args"].get("zip_code", "your location"))
                        return f"The weather in {location_info}: {tool_result}"
                else:
                    # Fallback to direct processing if Gemini response is unclear
                    if fallback_location:
                        return self._call_mcp_tool("get_weather", fallback_location)
                    return "I couldn't understand your request. Please ask for weather information for a US city or zip code."
                    
            except json.JSONDecodeError:
                # Fallback to direct processing if JSON parsing fails
                if fallback_location:
                    return self._call_mcp_tool("get_weather", fallback_location)
                return "I had trouble processing your request. Please ask for weather information for a US city or zip code."
                
        except Exception as e:
            error_msg = str(e)
            # Handle rate limiting specifically
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"⚠️ Gemini API rate limit hit, falling back to direct processing")
                if fallback_location:
                    return self._call_mcp_tool("get_weather", fallback_location)
                return "Weather service is temporarily busy. Please specify a clear city name or zip code."
            else:
                # For other errors, try fallback
                if fallback_location:
                    return self._call_mcp_tool("get_weather", fallback_location)
                return f"Sorry, I encountered an error: {error_msg}"
    
    def send(self, request_data):
        """Legacy method for backward compatibility"""
        if "tool" in request_data:
            # Direct tool call (for testing)
            result = self._call_mcp_tool(request_data["tool"], request_data["args"])
            return {
                "tool_result": {
                    "content": [{"type": "text", "text": result}]
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

@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events with proper signature validation and challenge response"""
    
    # Get raw payload and headers
    payload = request.get_data()
    headers = request.headers
    
    print(f"🔍 Slack events endpoint called")
    print(f"📥 Raw payload length: {len(payload)}")
    print(f"📋 Headers: {dict(headers)}")
    
    # Validate required headers
    if 'X-Slack-Request-Timestamp' not in headers:
        print("❌ Missing X-Slack-Request-Timestamp header")
        return "Missing timestamp header", 400
    
    if 'X-Slack-Signature' not in headers:
        print("❌ Missing X-Slack-Signature header")
        return "Missing signature header", 400
    
    # Parse JSON data
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
        print(f"🔍 Challenge: {challenge}")
        print(f"🔍 Token: {token}")
        
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
    if not verifier or not slack:
        print("❌ Slack integration not configured")
        return "Slack integration not configured", 500
    
    try:
        if not verifier.is_valid_request(payload, headers):
            print("❌ Invalid Slack signature")
            return "Invalid signature", 403
        print("✅ Slack signature verified")
    except Exception as e:
        print(f"❌ Signature verification error: {e}")
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
                
                # Send response back to Slack
                slack.chat_postMessage(channel=channel, text=result)
                
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
        
        else:
            print(f"⚠️ Unhandled event type: {event_type}")
    
    else:
        print(f"⚠️ Unhandled request type: {request_type}")
    
    # Always return 200 to acknowledge receipt
    return "", 200

@app.route("/test", methods=["POST", "GET"])
def test_weather():
    """
    Test endpoint that supports both natural language and direct tool calls
    
    POST /test with JSON:
    - {"text": "What's the weather in Miami?"} - Natural language
    - {"city": "Miami"} - Direct city call
    - {"zip_code": "33101"} - Direct ZIP code call
    
    GET /test - Returns endpoint info
    """
    
    if request.method == "GET":
        return jsonify({
            "endpoint": "/test",
            "methods": ["GET", "POST"],
            "description": "Test MCP Weather Bot functionality",
            "examples": {
                "natural_language": {"text": "What's the weather in Miami?"},
                "direct_city": {"city": "Miami"},
                "direct_zip": {"zip_code": "33101"}
            },
            "mcp_server_url": MCP_SERVER_URL,
            "gemini_model": GEMINI_MODEL,
            "status": "ready"
        }), 200
    
    # Handle POST requests
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "usage": "Send JSON with 'text', 'city', or 'zip_code' parameter"
            }), 400
        
        print(f"🧪 Test endpoint called with: {data}")
        
        # Support multiple input formats
        if "text" in data:
            # Natural language processing via Gemini + MCP
            user_text = data["text"]
            print(f"📝 Processing natural language: '{user_text}'")
            resp = client.send({"text": user_text})
            
        elif "city" in data:
            # Direct city tool call (bypass Gemini)
            city = data["city"]
            print(f"🏙️ Direct city call: '{city}'")
            resp = client.send({"tool": "get_weather", "args": {"city": city}})
            
        elif "zip_code" in data:
            # Direct ZIP code tool call (bypass Gemini)
            zip_code = data["zip_code"]
            print(f"📮 Direct ZIP call: '{zip_code}'")
            resp = client.send({"tool": "get_weather", "args": {"zip_code": zip_code}})
            
        else:
            return jsonify({
                "error": "Invalid request format",
                "required": "One of: 'text', 'city', or 'zip_code'",
                "examples": {
                    "natural_language": {"text": "weather in Phoenix"},
                    "direct_city": {"city": "Phoenix"},
                    "direct_zip": {"zip_code": "85044"}
                }
            }), 400
        
        # Process response
        if not resp:
            return jsonify({"error": "No response from MCP client"}), 500
            
        content = resp.get("tool_result", {}).get("content", [])
        
        # Extract text from the first content item
        if content and len(content) > 0:
            result = content[0].get("text", "No weather data received")
            print(f"✅ Response: {result}")
            
            return jsonify({
                "success": True,
                "response": result,
                "request": data,
                "timestamp": time.time()
            }), 200
        else:
            print("❌ No content in MCP response")
            return jsonify({
                "error": "No content received from MCP client",
                "response_data": resp
            }), 500
            
    except RequestException as e:
        error_msg = f"Network error connecting to MCP server: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 503
        
    except Exception as e:
        error_msg = f"Internal error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    # Use port 5002 for local development, PORT environment variable for Cloud Run
    port = int(os.environ.get("PORT", 5002))
    
    # Determine if running locally or on Cloud Run
    is_cloud_run = os.environ.get("K_SERVICE") is not None
    environment = "Cloud Run" if is_cloud_run else "Local"
    
    print(f"Starting MCP client on port {port} ({environment})")
    app.run(host="0.0.0.0", port=port, debug=not is_cloud_run)
