"""
Black box tests for security utilities
Tests message processing, rate limiting, and validation behavior
"""
import pytest
import time
from unittest.mock import patch
from utils.security import (
    is_message_processed, mark_message_processed, is_rate_limited,
    validate_message_age, cleanup_old_messages
)

class TestMessageProcessing:
    """Test message processing and deduplication"""
    
    def setup_method(self):
        """Clear state before each test"""
        from utils.security import processed_messages, user_last_request
        processed_messages.clear()
        user_last_request.clear()
    
    def test_new_message_not_processed(self):
        """Test that new messages are not marked as processed"""
        message_key = "test_message_123"
        assert not is_message_processed(message_key)
    
    def test_mark_and_check_message_processed(self):
        """Test marking and checking processed messages"""
        message_key = "test_message_456"
        
        # Initially not processed
        assert not is_message_processed(message_key)
        
        # Mark as processed
        mark_message_processed(message_key)
        
        # Now should be processed
        assert is_message_processed(message_key)
    
    def test_different_messages_independent(self):
        """Test that different messages are tracked independently"""
        message1 = "message_1"
        message2 = "message_2"
        
        mark_message_processed(message1)
        
        assert is_message_processed(message1)
        assert not is_message_processed(message2)

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def setup_method(self):
        """Clear state before each test"""
        from utils.security import processed_messages, user_last_request
        processed_messages.clear()
        user_last_request.clear()
    
    def test_first_request_not_rate_limited(self):
        """Test that first request from user is not rate limited"""
        user = "user123"
        assert not is_rate_limited(user)
    
    def test_rapid_requests_rate_limited(self):
        """Test that rapid successive requests are rate limited"""
        user = "user456"
        
        # First request should pass
        assert not is_rate_limited(user)
        
        # Immediate second request should be rate limited
        assert is_rate_limited(user)
    
    def test_different_users_independent_rate_limits(self):
        """Test that different users have independent rate limits"""
        user1 = "user1"
        user2 = "user2"
        
        # Both users' first requests should pass
        assert not is_rate_limited(user1)
        assert not is_rate_limited(user2)
        
        # Both users' rapid second requests should be rate limited
        assert is_rate_limited(user1)
        assert is_rate_limited(user2)
    
    def test_rate_limit_expires(self):
        """Test that rate limits expire after the interval"""
        import time
        user = "user789"
        
        # First request should not be rate limited
        assert not is_rate_limited(user)
        
        # Wait for rate limit to expire
        time.sleep(2.1)  # Slightly more than MIN_REQUEST_INTERVAL
        assert not is_rate_limited(user)
        


class TestMessageValidation:
    """Test message age validation"""
    
    @patch('time.time')
    def test_recent_message_valid(self, mock_time):
        """Test that recent messages are valid"""
        current_time = 1000
        message_time = 950  # 50 seconds ago
        mock_time.return_value = current_time
        
        assert validate_message_age(str(message_time))
    
    @patch('time.time')
    def test_old_message_invalid(self, mock_time):
        """Test that old messages are invalid"""
        current_time = 1000
        message_time = 500  # 500 seconds ago (> MESSAGE_EXPIRY_SECONDS)
        mock_time.return_value = current_time
        
        assert not validate_message_age(str(message_time))
    
    def test_invalid_timestamp_format(self):
        """Test handling of invalid timestamp formats"""
        invalid_timestamps = ["not_a_number", "", "abc123", None]
        
        for invalid_ts in invalid_timestamps:
            assert not validate_message_age(str(invalid_ts) if invalid_ts else "")

class TestCleanupFunctionality:
    """Test cleanup of old data"""
    
    def setup_method(self):
        """Clear state before each test"""
        from utils.security import processed_messages, user_last_request
        processed_messages.clear()
        user_last_request.clear()
    
    @patch('time.time')
    def test_cleanup_old_messages(self, mock_time):
        """Test that old messages are cleaned up"""
        from utils.security import processed_messages
        
        current_time = 1000
        old_time = 500  # Very old
        recent_time = 950  # Recent
        
        # Add old and recent messages
        processed_messages["old_message"] = old_time
        processed_messages["recent_message"] = recent_time
        
        mock_time.return_value = current_time
        cleanup_old_messages()
        
        # Old message should be removed, recent should remain
        assert "old_message" not in processed_messages
        assert "recent_message" in processed_messages
    
    @patch('time.time')
    def test_cleanup_memory_limit(self, mock_time):
        """Test that cleanup respects memory limits"""
        from utils.security import processed_messages, MAX_PROCESSED_MESSAGES
        
        mock_time.return_value = 1000
        
        # Add more messages than the limit
        for i in range(MAX_PROCESSED_MESSAGES + 100):
            processed_messages[f"message_{i}"] = 999 - i  # Older messages have lower timestamps
        
        cleanup_old_messages()
        
        # Should not exceed the maximum
        assert len(processed_messages) <= MAX_PROCESSED_MESSAGES
    
    def test_cleanup_handles_empty_state(self):
        """Test that cleanup handles empty state gracefully"""
        # Should not raise any exceptions
        cleanup_old_messages()
        
        # State should remain empty
        from utils.security import processed_messages, user_last_request
        assert len(processed_messages) == 0
        assert len(user_last_request) == 0
