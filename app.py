import os
import json
import requests
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

# Debug: Print MCP server configuration on startup
print(f"MCP Server Configuration:")
print(f"  Host: {MCP_SERVER_HOST}")
print(f"  Port: {MCP_SERVER_PORT}")
print(f"  Protocol: {MCP_SERVER_PROTOCOL}")
print(f"  Path: {MCP_SERVER_PATH}")
print(f"  Full URL: {MCP_SERVER_URL}")

# Gemini AI configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
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
    
    def process_natural_language(self, user_input):
        """Use Gemini to process natural language and determine tool calls"""
        
        if not self.model:
            # Fallback: try to extract city from user input
            words = user_input.split()
            if len(words) >= 1:
                # Try to find location in the text
                # Skip bot mentions (@bot, @weatherbot, etc.)
                location_words = []
                for word in words:
                    if not word.startswith('@'):
                        location_words.append(word)
                
                if location_words:
                    location = " ".join(location_words).strip()
                    if location.isdigit() and len(location) == 5:
                        return self._call_mcp_tool("get_weather", {"zip_code": location})
                    else:
                        return self._call_mcp_tool("get_weather", {"city": location})
            
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
                    
                    # Use Gemini to format the response naturally
                    format_prompt = f"""Format this real-time weather information in a natural, conversational way:

Live weather data from MCP tools: {tool_result}
Original request: {user_input}

Instructions:
1. Respond in a friendly, natural way as if you're personally telling someone about the weather
2. Keep it concise and conversational
3. You may mention that this is current/live data if appropriate
4. Don't add information not provided in the weather data

Example formats:
- "It's currently sunny and 75°F in Miami!"
- "Right now in Seattle, it's mostly cloudy at 68°F"
- "The current conditions in 90210 show partly sunny skies with a temperature of 82°F"""
                    
                    format_response = self.model.generate_content(format_prompt)
                    return format_response.text.strip()
                else:
                    return "I couldn't understand your request. Please ask for weather information for a US city or zip code."
                    
            except json.JSONDecodeError:
                return "I had trouble processing your request. Please ask for weather information for a US city or zip code."
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
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
    payload = request.get_data()
    headers = request.headers
    if not verifier.is_valid_request(payload, headers):
        return "Invalid signature", 403
    data = request.json
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})
    event = data.get("event", {})
    if event.get("type") == "app_mention":
        try:
            # Get the full user message
            user_text = event["text"]
            
            # Use Gemini to process the natural language request
            resp = client.send({"text": user_text})
            content = resp.get("tool_result", {}).get("content", [])
            
            # Extract text from the first content item
            if content and len(content) > 0:
                result = content[0].get("text", "No weather data received")
            else:
                result = "Error from MCP client - no content received"
                
            slack.chat_postMessage(channel=event["channel"], text=result)
        except RequestException as e:
            slack.chat_postMessage(channel=event["channel"], text=f"Network error: {e}")
        except Exception as e:
            slack.chat_postMessage(channel=event["channel"], text=f"Error: {e}")
    return "", 200

@app.route("/test", methods=["POST"])
def test_weather():
    """Test endpoint that supports both natural language and direct tool calls"""
    try:
        data = request.get_json()
        
        # Support both natural language and direct tool calls
        if "text" in data:
            # Natural language processing
            resp = client.send({"text": data["text"]})
        elif "city" in data:
            # Direct tool call (backward compatibility)
            resp = client.send({"tool": "get_weather", "args": {"city": data["city"]}})
        elif "zip_code" in data:
            # Direct tool call with zip code
            resp = client.send({"tool": "get_weather", "args": {"zip_code": data["zip_code"]}})
        else:
            return jsonify({"error": "Provide 'text' for natural language or 'city'/'zip_code' for direct call"}), 400
        
        content = resp.get("tool_result", {}).get("content", [])
        
        # Extract text from the first content item
        if content and len(content) > 0:
            result = content[0].get("text", "No weather data received")
        else:
            result = "Error from MCP client - no content received"
            
        return jsonify({"response": result}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use port 5002 for local development, PORT environment variable for Cloud Run
    port = int(os.environ.get("PORT", 5002))
    
    # Determine if running locally or on Cloud Run
    is_cloud_run = os.environ.get("K_SERVICE") is not None
    environment = "Cloud Run" if is_cloud_run else "Local"
    
    print(f"Starting MCP client on port {port} ({environment})")
    app.run(host="0.0.0.0", port=port, debug=not is_cloud_run)
