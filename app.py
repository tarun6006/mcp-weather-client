import os
import requests
from flask import Flask, request, jsonify
from mcp_utils.core import MCPClient
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier

load_dotenv()
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")  # weather server endpoint
USER_AGENT = os.getenv("NWS_USER_AGENT")

slack = WebClient(token=SLACK_BOT_TOKEN)
verifier = SignatureVerifier(SLACK_SIGNING_SECRET)
client = MCPClient("weather-client", "1.0", server_url=MCP_SERVER_URL)

app = Flask(__name__)

@app.route("/slack/command", methods=["POST"])
def slash():
    if not verifier.is_valid_request(request.get_data(), request.headers):
        return "Invalid", 403
    city = request.form.get("text")
    inp = {"tool": "get_weather", "args": {"city": city}}
    resp = client.send(inp)
    text = resp.get("tool_result", {}).get("content", "Sorry, error.")
    slack.chat_postMessage(channel=request.form["channel_id"], text=text)
    return "", 200

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)))
