"""SSE client for calculator MCP server communication"""
import requests
import json
import time
import uuid
import threading
import logging
import sseclient
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SSECalculatorClient:
    """SSE client for calculator MCP server"""
    
    def __init__(self, calc_server_host: str, calc_server_port: int, calc_server_protocol: str):
        self.calc_server_host = calc_server_host
        self.calc_server_port = calc_server_port
        self.calc_server_protocol = calc_server_protocol
        self.client_id = str(uuid.uuid4())
        self.sse_client = None
        self.response_queue: Dict[str, Any] = {}
        self.connected = False
        self.lock = threading.Lock()
        
        # Build SSE connection URL
        port_str = f":{calc_server_port}" if calc_server_port not in ["80", "443"] else ""
        self.sse_url = f"{calc_server_protocol}://{calc_server_host}{port_str}/sse/connect?client_id={self.client_id}"
        self.mcp_url = f"{calc_server_protocol}://{calc_server_host}{port_str}/sse/mcp"
        
        self._connect()
    
    def _connect(self):
        """Establish SSE connection"""
        try:
            logger.info(f"Connecting to calculator SSE server: {self.sse_url}")
            response = requests.get(self.sse_url, stream=True, timeout=10)
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
        """Handle incoming SSE messages"""
        try:
            for event in self.sse_client.events():
                if event.event == "message":
                    try:
                        data = json.loads(event.data)
                        request_id = data.get("id")
                        
                        if request_id:
                            with self.lock:
                                self.response_queue[request_id] = data
                            logger.debug(f"Received SSE response for request {request_id}")
                        else:
                            logger.warning(f"Received SSE message without request ID: {data}")
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse SSE message: {event.data}, error: {e}")
                
                elif event.event == "heartbeat":
                    logger.debug("Received SSE heartbeat")
                
                elif event.event == "error":
                    logger.error(f"SSE server error: {event.data}")
                    
        except Exception as e:
            logger.error(f"SSE message handling error: {e}")
            self.connected = False
    
    def send_request(self, request_data: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Send request to calculator server via SSE"""
        if not self.connected:
            logger.error("SSE client not connected")
            return None
        
        request_id = str(uuid.uuid4())
        request_data["id"] = request_id
        
        try:
            # Send request to SSE MCP endpoint
            response = requests.post(
                self.mcp_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            # Wait for response via SSE
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self.lock:
                    if request_id in self.response_queue:
                        result = self.response_queue.pop(request_id)
                        logger.debug(f"Got SSE response for {request_id}")
                        return result
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            logger.error(f"SSE request {request_id} timed out after {timeout}s")
            return None
            
        except Exception as e:
            logger.error(f"Failed to send SSE request: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from SSE server"""
        self.connected = False
        if self.sse_client:
            try:
                self.sse_client.close()
                logger.info("SSE calculator client disconnected")
            except:
                pass
