"""Gemini AI client with MCP tool integration"""
import json
import logging
import requests
import google.generativeai as genai
from typing import Dict, Any, Optional
from services.sse_client import SSECalculatorClient
from config.settings import CONFIG

logger = logging.getLogger(__name__)

class GeminiMCPClient:
    """MCP client that uses Gemini LLM to process natural language and call tools"""
    
    def __init__(self, weather_server_url: str, calc_server_host: str, calc_server_port: int, 
                 calc_server_protocol: str, gemini_model=None):
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

    def _generate_tool_call_formats(self, tool_call_formats: Dict[str, Any]) -> str:
        """Generate tool call format section from configuration"""
        formats_section = ""
        
        # Weather formats
        weather_formats = tool_call_formats.get('weather_queries', {})
        if weather_formats:
            formats_section += "Weather queries:\n"
            if 'city_format' in weather_formats:
                formats_section += f"- For US cities: {weather_formats['city_format']}\n"
            if 'zip_format' in weather_formats:
                formats_section += f"- For zip codes: {weather_formats['zip_format']}\n"
        
        # Math formats  
        math_formats = tool_call_formats.get('math_calculations', {})
        if math_formats:
            formats_section += "\nMath calculations:\n"
            for format_name, format_example in math_formats.items():
                if format_name != 'non_supported_query':
                    format_display_name = format_name.replace('_', ' ').title()
                    formats_section += f"- For {format_display_name.lower()}: {format_example}\n"
        
        # Non-supported query format
        non_supported = tool_call_formats.get('non_supported_query', '')
        if non_supported:
            formats_section += f"\nIf request is not weather or math-related:\n- Respond with: {non_supported}\n"
        
        return formats_section.rstrip()

    def _generate_examples(self, examples: Dict[str, Any]) -> str:
        """Generate examples section from configuration"""
        examples_section = ""
        
        # Weather examples
        weather_examples = examples.get('weather', [])
        if weather_examples:
            examples_section += "\nWeather examples:\n"
            for example in weather_examples:
                examples_section += f"- \"{example['input']}\" = {example['output']}\n"
        
        # Math examples
        math_examples = examples.get('math', [])
        if math_examples:
            examples_section += "\nMath examples:\n"
            for example in math_examples:
                examples_section += f"- \"{example['input']}\" = {example['output']}\n"
        
        return examples_section.rstrip()

    def _format_response(self, tool_result: Dict[str, Any], tool_name: str, arguments: Dict[str, Any], user_input: str) -> str:
        """Format the tool response into a natural language response"""
        if "error" in tool_result:
            return f"Sorry, I encountered an error: {tool_result['error']}"
        
        if tool_name == "get_weather":
            if "result" in tool_result:
                weather_data = tool_result["result"]
                if isinstance(weather_data, dict) and "properties" in weather_data:
                    location = weather_data.get("properties", {}).get("location", "Unknown location")
                    temp = weather_data.get("properties", {}).get("temperature")
                    conditions = weather_data.get("properties", {}).get("shortForecast", "No conditions available")
                    
                    weather_prefix = self.response_messages.get('weather_prefix', 'Weather for')
                    temp_label = self.response_messages.get('temperature_label', 'Temperature:')
                    conditions_label = self.response_messages.get('conditions_label', 'Conditions:')
                    
                    response = f"{weather_prefix} {location}:\n"
                    if temp:
                        response += f"{temp_label} {temp}\n"
                    response += f"{conditions_label} {conditions}"
                    return response
                else:
                    return f"{self.response_messages.get('weather_data_fallback', 'Weather data')}: {weather_data}"
            else:
                return self.error_messages.get('no_weather_data', 'No weather data received from the server.')
        
        elif tool_name in self.calculator_tools:
            if "result" in tool_result:
                result = tool_result["result"]
                return f"{self.response_messages.get('result_prefix', 'The result is')}: {result}"
            else:
                return self.error_messages.get('no_calculation_result', 'No calculation result received from the server.')
        
        return f"{self.response_messages.get('generic_result', 'Result')}: {tool_result}"

    def process_natural_language(self, user_input: str) -> Dict[str, Any]:
        """Process natural language input using Gemini and call appropriate MCP tools"""
        if not self.model:
            return {
                "tool_result": {
                    "content": [{
                        "text": self.error_messages.get('gemini_not_configured', 'Gemini model is not configured. Please check your GOOGLE_API_KEY.')
                    }]
                }
            }
        
        if not user_input.strip():
            return self.error_messages.get('empty_input', 'Please specify a city or zip code for weather information or a mathematical calculation.')
        
        # Build system prompt from configuration
        gemini_config = CONFIG.get('gemini_instructions', {})
        system_prompt_template = gemini_config.get('system_prompt_template', 'You are an AI assistant connected to specialized MCP servers.')
        
        system_prompt = f"""{system_prompt_template}

AVAILABLE MCP TOOLS:
{json.dumps(self.tools_schema, indent=2)}

CRITICAL INSTRUCTIONS - NO EXCEPTIONS:
{chr(10).join(f"{i}. {instruction}" for i, instruction in enumerate(gemini_config.get('critical_instructions', []), 1))}

MANDATORY TOOL CALL FORMAT:
{self._generate_tool_call_formats(gemini_config.get('tool_call_formats', {}))}

EXAMPLES - Notice how EVERY query MUST use MCP tools:
{self._generate_examples(gemini_config.get('examples', {}))}

REMEMBER: You are connected to specialized MCP servers. You MUST use these MCP tools for ALL weather and math queries - never rely on your training data or do calculations yourself.

Only respond with the JSON object, nothing else."""

        try:
            response = self.model.generate_content(f"{system_prompt}\n\nUser request: {user_input}")
            
            # Check if response was blocked by safety filters
            if not response.candidates or not response.candidates[0].content.parts:
                logger.error(f"Gemini response blocked by safety filters. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                return {
                    "tool_result": {
                        "content": [{
                            "text": self.error_messages.get('safety_restriction', 'I apologize, but I cannot process that request due to content safety restrictions. Please try rephrasing your question about weather or calculations.')
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
                    tool_name = parsed["tool"]
                    arguments = parsed["args"]
                    
                    # Call the appropriate MCP server
                    if tool_name == "get_weather":
                        result = self._call_weather_server(tool_name, arguments)
                    elif tool_name in self.calculator_tools:
                        result = self._call_calculator_server(tool_name, arguments)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    
                    # Format the response
                    formatted_response = self._format_response(result, tool_name, arguments, user_input)
                    
                    return {
                        "tool_result": {
                            "content": [{
                                "text": formatted_response
                            }]
                        }
                    }
                else:
                    logger.error(f"Invalid Gemini response format: {parsed}")
                    return {
                        "tool_result": {
                            "content": [{
                                "text": self.error_messages.get('understanding_error', 'I apologize, but I couldn\'t understand how to help with that request. Please ask about weather information (US cities/zip codes) or mathematical calculations.')
                            }]
                        }
                    }
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini JSON response: {gemini_response}, error: {e}")
                return {
                    "tool_result": {
                        "content": [{
                            "text": self.error_messages.get('processing_error', 'I apologize, but I had trouble processing that request. Please ask about weather information (US cities/zip codes) or mathematical calculations.')
                        }]
                    }
                }
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "tool_result": {
                    "content": [{
                        "text": "I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists."
                    }]
                }
            }

    def _call_weather_server(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call the weather MCP server"""
        self.request_id += 1
        
        mcp_request = {
            "jsonrpc": "2.0",
            "id": str(self.request_id),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            response = requests.post(
                self.weather_server_url,
                json=mcp_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Weather server error: {e}")
            return {"error": f"Weather server unavailable: {e}"}

    def _call_calculator_server(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call the calculator MCP server via SSE"""
        self.request_id += 1
        
        mcp_request = {
            "jsonrpc": "2.0",
            "id": str(self.request_id),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            result = self.calc_sse_client.send_request(mcp_request)
            if result:
                return result
            else:
                return {"error": "Calculator server timeout or unavailable"}
        except Exception as e:
            logger.error(f"Calculator server error: {e}")
            return {"error": f"Calculator server error: {e}"}

    def send(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return the result"""
        user_text = request_data.get("text", "").strip()
        return self.process_natural_language(user_text)
