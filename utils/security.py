"""Security utilities for message processing and rate limiting"""
import time
import logging
from typing import Dict, Set
from config.settings import MESSAGE_EXPIRY_SECONDS, MIN_REQUEST_INTERVAL, MAX_PROCESSED_MESSAGES

logger = logging.getLogger(__name__)

# Global state for message processing and rate limiting
processed_messages: Dict[str, float] = {}
user_last_request: Dict[str, float] = {}

def cleanup_old_messages():
    """Remove old processed messages to prevent memory bloat"""
    current_time = time.time()
    cutoff_time = current_time - MESSAGE_EXPIRY_SECONDS
    
    # Clean up processed messages
    keys_to_remove = [
        key for key, timestamp in processed_messages.items() 
        if timestamp < cutoff_time
    ]
    for key in keys_to_remove:
        del processed_messages[key]
    
    # Clean up rate limiting data
    rate_keys_to_remove = [
        user for user, timestamp in user_last_request.items() 
        if timestamp < cutoff_time
    ]
    for user in rate_keys_to_remove:
        del user_last_request[user]
    
    # Limit memory usage
    if len(processed_messages) > MAX_PROCESSED_MESSAGES:
        # Remove oldest entries
        sorted_messages = sorted(processed_messages.items(), key=lambda x: x[1])
        excess_count = len(processed_messages) - MAX_PROCESSED_MESSAGES
        for key, _ in sorted_messages[:excess_count]:
            del processed_messages[key]
    
    if keys_to_remove or rate_keys_to_remove:
        logger.debug(f"Cleaned up {len(keys_to_remove)} old messages and {len(rate_keys_to_remove)} rate limit entries")

def is_message_processed(message_key: str) -> bool:
    """Check if a message has already been processed"""
    cleanup_old_messages()
    return message_key in processed_messages

def mark_message_processed(message_key: str):
    """Mark a message as processed"""
    processed_messages[message_key] = time.time()

def is_rate_limited(user: str) -> bool:
    """Check if a user is rate limited"""
    current_time = time.time()
    last_request = user_last_request.get(user, 0)
    
    if current_time - last_request < MIN_REQUEST_INTERVAL:
        return True
    
    user_last_request[user] = current_time
    return False

def validate_message_age(event_ts: str) -> bool:
    """Validate that a message is not too old"""
    try:
        event_time = float(event_ts)
        current_time = time.time()
        message_age = current_time - event_time
        
        if message_age > MESSAGE_EXPIRY_SECONDS:
            logger.info(f"Message too old ({message_age:.1f}s), ignoring")
            return False
        return True
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid event timestamp: {event_ts}, error: {e}")
        return False
