#!/usr/bin/env python3
"""
Calculator MCP Service

Provides SSE (Server-Sent Events) client for calculator MCP server
with event-driven architecture and real-time communication.
"""

import json
import threading
import uuid
import logging
import requests
import sseclient
from typing import Dict, Any, Optional
import os
import yaml

from utils.http_utility import http_client

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}
    except yaml.YAMLError:
        return {}

CONFIG = load_config()
TIMEOUT_CONFIG = CONFIG.get('timeout_config', {})
SSE_CONNECTION_TIMEOUT = TIMEOUT_CONFIG.get('sse_connection_timeout', 30)
SSE_REQUEST_TIMEOUT = TIMEOUT_CONFIG.get('sse_request_timeout', 30)
SSE_RESPONSE_TIMEOUT = TIMEOUT_CONFIG.get('sse_response_timeout', 30)

class CalculatorMCPService:
    """MCP service for calculator server with SSE event-driven architecture"""
    
    def __init__(self, calc_server_host: str, calc_server_port: int, calc_server_protocol: str):
        self.calc_server_host = calc_server_host
        self.calc_server_port = calc_server_port
        self.calc_server_protocol = calc_server_protocol
        self.client_id = str(uuid.uuid4())
        self.sse_client = None
        self.response_events = {}  # request_id -> threading.Event
        self.response_data = {}    # request_id -> response_data
        self.connected = self.initial_connected_state
        self.lock = threading.Lock()
        
        # Load error and logging messages from config
        config = load_config()
        self.error_messages = config.get('error_messages', {})
        self.logging_messages = config.get('logging_messages', {})
        
        # Get standard ports from config
        utilities_config = config.get('utilities', {})
        standard_ports = utilities_config.get('standard_ports', ["80", "443"])
        
        # Get HTTP client config for SSE connections
        http_config = config.get('http_client_config', {})
        sse_config = http_config.get('sse_connection', {})
        self.sse_stream = sse_config.get('stream', True)
        
        # Get connection defaults from config
        connection_defaults = utilities_config.get('connection_defaults', {})
        self.initial_connected_state = connection_defaults.get('initial_connected_state', False)
        
        # Build SSE connection URL
        port_str = f":{calc_server_port}" if str(calc_server_port) not in standard_ports else ""
        self.sse_url = f"{calc_server_protocol}://{calc_server_host}{port_str}/sse/connect?client_id={self.client_id}"
        self.mcp_url = f"{calc_server_protocol}://{calc_server_host}{port_str}/sse/mcp"
        
        self._connect()
    
    def _connect(self):
        """Establish SSE connection"""
        try:
            logger.info(self.logging_messages.get('sse_connecting', "Connecting to calculator SSE server: {url}").format(url=self.sse_url))
            response = http_client.get(self.sse_url, stream=self.sse_stream, timeout=SSE_CONNECTION_TIMEOUT)
            response.raise_for_status()
            
            self.sse_client = sseclient.SSEClient(response)
            self.connected = True
            
            # Start background thread to handle SSE messages
            self.sse_thread = threading.Thread(target=self._handle_sse_messages, daemon=True)
            self.sse_thread.start()
            
            logger.info(self.logging_messages.get('sse_connection_established', "SSE connection established with calculator server (client_id: {client_id})").format(client_id=self.client_id))
            
        except Exception as e:
            error_msg = self.error_messages.get('calculator_sse_connect_failed', "Failed to connect to calculator SSE server: {error}").format(error=str(e))
            logger.error(error_msg)
            self.connected = self.initial_connected_state
    
    def _handle_sse_messages(self):
        """Handle incoming SSE messages with event-driven architecture"""
        try:
            for event in self.sse_client.events():
                if not self.connected:
                    break
                
                logger.debug(self.logging_messages.get('debug_messages', {}).get('sse_event_debug', "SSE event: {event}, data: {data}").format(event=event.event, data=event.data))
                
                if event.event == "connected":
                    try:
                        data = json.loads(event.data)
                        logger.debug(self.logging_messages.get('debug_messages', {}).get('sse_connection_confirmed_debug', "SSE connection confirmed: {data}").format(data=data))
                    except json.JSONDecodeError:
                        logger.debug(self.logging_messages.get('sse_connection_confirmed', "SSE connection confirmed (non-JSON data)"))
                
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
                                    logger.debug(self.logging_messages.get('debug_messages', {}).get('sse_signaled_event_debug', "Signaled event for request {request_id}").format(request_id=request_id))
                    except json.JSONDecodeError as e:
                        logger.warning(self.logging_messages.get('warning_messages', {}).get('sse_parse_failed', "Failed to parse SSE message: {data}, error: {error}").format(data=event.data, error=str(e)))
                
                elif event.event == "heartbeat":
                    try:
                        data = json.loads(event.data)
                        logger.debug(self.logging_messages.get('debug_messages', {}).get('sse_heartbeat_debug', "SSE heartbeat received: {timestamp}").format(timestamp=data.get('timestamp')))
                    except json.JSONDecodeError:
                        logger.debug(self.logging_messages.get('sse_heartbeat', "SSE heartbeat received"))
                
                else:
                    logger.debug(self.logging_messages.get('sse_unknown_event', "Unknown SSE event type: {event}").format(event=event.event))
                    
        except Exception as e:
            logger.error(f"SSE message handler error: {e}")
            self.connected = self.initial_connected_state
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Make MCP tool call via SSE with event-driven waiting"""
        if not self.connected:
            return self.error_messages.get('calculator_not_connected', "Error: Not connected to calculator SSE server")
        
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
                        return self.error_messages.get('calculator_unknown_error', "Unknown error occurred")
                else:
                    return self.error_messages.get('calculator_no_response', "Error: No response data received")
            else:
                # Cleanup on timeout
                with self.lock:
                    self.response_events.pop(request_id, None)
                    self.response_data.pop(request_id, None)
                return self.error_messages.get('calculator_timeout_response', "Error: Timeout waiting for SSE response ({timeout}s)").format(timeout=SSE_RESPONSE_TIMEOUT)
            
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
        self.connected = self.initial_connected_state
        if self.sse_client:
            try:
                self.sse_client.close()
            except:
                pass
