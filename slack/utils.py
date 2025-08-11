"""Slack utility functions for bot operations"""
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from config.settings import GREEN_TICK_EMOJI

logger = logging.getLogger(__name__)

def get_bot_user_id(slack_client: WebClient) -> str:
    """Get the bot's user ID"""
    try:
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        logger.error(f"Failed to get bot user ID: {e}")
        return ""

def check_message_already_processed(slack_client: WebClient, channel: str, message_ts: str) -> bool:
    """Check if a message already has a bot reply (thread or reaction)"""
    try:
        bot_user_id = get_bot_user_id(slack_client)
        if not bot_user_id:
            return False
        
        # Check for reactions
        try:
            reactions_response = slack_client.reactions_get(channel=channel, timestamp=message_ts)
            if reactions_response.get("ok"):
                reactions = reactions_response.get("message", {}).get("reactions", [])
                for reaction in reactions:
                    if reaction.get("name") == GREEN_TICK_EMOJI:
                        if bot_user_id in reaction.get("users", []):
                            logger.info(f"Message {message_ts} already has bot reaction")
                            return True
        except SlackApiError:
            # Reactions might not exist, continue checking for replies
            pass
        
        # Check for thread replies from bot
        try:
            replies_response = slack_client.conversations_replies(channel=channel, ts=message_ts)
            if replies_response.get("ok"):
                messages = replies_response.get("messages", [])
                for message in messages[1:]:  # Skip the original message
                    if message.get("user") == bot_user_id:
                        logger.info(f"Message {message_ts} already has bot thread reply")
                        return True
        except SlackApiError:
            # Thread might not exist, that's fine
            pass
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking if message already processed: {e}")
        return False

def send_threaded_response_with_reaction(slack_client: WebClient, channel: str, thread_ts: str, 
                                       response_text: str, user_id: str = None) -> bool:
    """Send response in thread and add reaction to original message"""
    try:
        # Tag the user in the response if user_id provided
        if user_id:
            response_text = f"<@{user_id}> {response_text}"
        
        # Send threaded response
        response = slack_client.chat_postMessage(
            channel=channel,
            text=response_text,
            thread_ts=thread_ts
        )
        
        if response["ok"]:
            # Add green tick reaction to original message
            try:
                slack_client.reactions_add(
                    channel=channel,
                    timestamp=thread_ts,
                    name=GREEN_TICK_EMOJI
                )
                logger.info(f"Sent threaded response and added reaction for message {thread_ts}")
                return True
            except SlackApiError as e:
                logger.warning(f"Failed to add reaction but message sent: {e}")
                return True  # Message was sent successfully, reaction failure is not critical
        else:
            logger.error(f"Failed to send threaded response: {response}")
            return False
            
    except SlackApiError as e:
        logger.error(f"Slack API error in threaded response: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in threaded response: {e}")
        return False

def handle_thread_spam_request(slack_client: WebClient, channel: str, thread_ts: str, user: str):
    """Handle requests made in existing threads (spam prevention)"""
    try:
        spam_message = f"<@{user}> Please send new questions as fresh messages rather than replying in threads. This helps keep conversations organized and ensures I can respond properly to your request."
        
        slack_client.chat_postMessage(
            channel=channel,
            text=spam_message,
            thread_ts=thread_ts
        )
        logger.info(f"Sent thread spam prevention message to user {user}")
        
    except SlackApiError as e:
        logger.error(f"Failed to send thread spam message: {e}")

def is_request_in_existing_thread(event: dict) -> bool:
    """Check if the request is part of an existing thread"""
    return "thread_ts" in event and event.get("ts") != event.get("thread_ts")
