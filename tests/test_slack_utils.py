"""
Black box tests for Slack utilities
Tests Slack bot functionality without knowing internal implementation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from slack_sdk.errors import SlackApiError
from slack.utils import (
    get_bot_user_id, check_message_already_processed,
    send_threaded_response_with_reaction, handle_thread_spam_request,
    is_request_in_existing_thread
)

class TestBotUserID:
    """Test bot user ID retrieval"""
    
    def test_get_bot_user_id_success(self):
        """Test successful bot user ID retrieval"""
        mock_client = Mock()
        mock_client.auth_test.return_value = {"user_id": "U123456789"}
        
        user_id = get_bot_user_id(mock_client)
        
        assert user_id == "U123456789"
        mock_client.auth_test.assert_called_once()
    
    def test_get_bot_user_id_api_error(self):
        """Test bot user ID retrieval with API error"""
        mock_client = Mock()
        mock_client.auth_test.side_effect = SlackApiError("Auth failed", response={"error": "invalid_auth"})
        
        user_id = get_bot_user_id(mock_client)
        
        assert user_id == ""
        mock_client.auth_test.assert_called_once()

class TestMessageProcessingCheck:
    """Test message processing status checking"""
    
    @patch('slack.utils.get_bot_user_id')
    def test_check_message_no_bot_user_id(self, mock_get_bot_user_id):
        """Test message check when bot user ID cannot be retrieved"""
        mock_get_bot_user_id.return_value = ""
        mock_client = Mock()
        
        result = check_message_already_processed(mock_client, "C123", "1234567890.123")
        
        assert result is False
    
    @patch('slack.utils.get_bot_user_id')
    def test_check_message_with_bot_reaction(self, mock_get_bot_user_id):
        """Test message check when bot has already reacted"""
        mock_get_bot_user_id.return_value = "U_BOT123"
        mock_client = Mock()
        
        # Mock reactions response
        mock_client.reactions_get.return_value = {
            "ok": True,
            "message": {
                "reactions": [
                    {
                        "name": "white_check_mark",
                        "users": ["U_BOT123", "U_OTHER"]
                    }
                ]
            }
        }
        
        result = check_message_already_processed(mock_client, "C123", "1234567890.123")
        
        assert result is True
        mock_client.reactions_get.assert_called_once()
    
    @patch('slack.utils.get_bot_user_id')
    def test_check_message_with_bot_thread_reply(self, mock_get_bot_user_id):
        """Test message check when bot has already replied in thread"""
        mock_get_bot_user_id.return_value = "U_BOT123"
        mock_client = Mock()
        
        # Mock no reactions but thread reply exists
        mock_client.reactions_get.side_effect = SlackApiError("No reactions", response={})
        mock_client.conversations_replies.return_value = {
            "ok": True,
            "messages": [
                {"user": "U_HUMAN", "ts": "1234567890.123"},  # Original message
                {"user": "U_BOT123", "ts": "1234567890.124"}   # Bot reply
            ]
        }
        
        result = check_message_already_processed(mock_client, "C123", "1234567890.123")
        
        assert result is True
        mock_client.conversations_replies.assert_called_once()
    
    @patch('slack.utils.get_bot_user_id')
    def test_check_message_not_processed(self, mock_get_bot_user_id):
        """Test message check when not processed"""
        mock_get_bot_user_id.return_value = "U_BOT123"
        mock_client = Mock()
        
        # No reactions and no thread replies
        mock_client.reactions_get.side_effect = SlackApiError("No reactions", response={})
        mock_client.conversations_replies.return_value = {
            "ok": True,
            "messages": [
                {"user": "U_HUMAN", "ts": "1234567890.123"}  # Only original message
            ]
        }
        
        result = check_message_already_processed(mock_client, "C123", "1234567890.123")
        
        assert result is False

class TestThreadedResponse:
    """Test threaded response functionality"""
    
    def test_send_threaded_response_success(self):
        """Test successful threaded response"""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ok": True}
        mock_client.reactions_add.return_value = {"ok": True}
        
        result = send_threaded_response_with_reaction(
            mock_client, "C123", "1234567890.123", "Test response", "U_USER123"
        )
        
        assert result is True
        mock_client.chat_postMessage.assert_called_once()
        mock_client.reactions_add.assert_called_once()
        
        # Check that user is tagged in response
        call_args = mock_client.chat_postMessage.call_args
        assert "<@U_USER123>" in call_args[1]["text"]
    
    def test_send_threaded_response_without_user_tag(self):
        """Test threaded response without user tagging"""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ok": True}
        mock_client.reactions_add.return_value = {"ok": True}
        
        response_text = "Test response"
        result = send_threaded_response_with_reaction(
            mock_client, "C123", "1234567890.123", response_text
        )
        
        assert result is True
        call_args = mock_client.chat_postMessage.call_args
        assert call_args[1]["text"] == response_text
    
    def test_send_threaded_response_message_fails(self):
        """Test threaded response when message sending fails"""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ok": False}
        
        result = send_threaded_response_with_reaction(
            mock_client, "C123", "1234567890.123", "Test response"
        )
        
        assert result is False
        mock_client.reactions_add.assert_not_called()
    
    def test_send_threaded_response_reaction_fails(self):
        """Test threaded response when reaction fails but message succeeds"""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ok": True}
        mock_client.reactions_add.side_effect = SlackApiError("Reaction failed", response={})
        
        result = send_threaded_response_with_reaction(
            mock_client, "C123", "1234567890.123", "Test response"
        )
        
        assert result is True  # Should still return True if message sent successfully
    
    def test_send_threaded_response_exception(self):
        """Test threaded response with unexpected exception"""
        mock_client = Mock()
        mock_client.chat_postMessage.side_effect = Exception("Unexpected error")
        
        result = send_threaded_response_with_reaction(
            mock_client, "C123", "1234567890.123", "Test response"
        )
        
        assert result is False

class TestThreadSpamHandling:
    """Test thread spam prevention"""
    
    def test_handle_thread_spam_request_success(self):
        """Test successful thread spam handling"""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ok": True}
        
        # Should not raise exception
        handle_thread_spam_request(mock_client, "C123", "1234567890.123", "U_USER123")
        
        mock_client.chat_postMessage.assert_called_once()
        call_args = mock_client.chat_postMessage.call_args
        assert "<@U_USER123>" in call_args[1]["text"]
        assert "fresh messages" in call_args[1]["text"]
    
    def test_handle_thread_spam_request_api_error(self):
        """Test thread spam handling with API error"""
        mock_client = Mock()
        mock_client.chat_postMessage.side_effect = SlackApiError("Send failed", response={})
        
        # Should not raise exception even with API error
        handle_thread_spam_request(mock_client, "C123", "1234567890.123", "U_USER123")
        
        mock_client.chat_postMessage.assert_called_once()

class TestThreadDetection:
    """Test thread detection functionality"""
    
    def test_is_request_in_existing_thread_true(self):
        """Test detection of request in existing thread"""
        event = {
            "thread_ts": "1234567890.123",
            "ts": "1234567890.124"  # Different from thread_ts
        }
        
        result = is_request_in_existing_thread(event)
        
        assert result is True
    
    def test_is_request_in_existing_thread_false_no_thread(self):
        """Test detection when not in thread"""
        event = {
            "ts": "1234567890.123"
            # No thread_ts
        }
        
        result = is_request_in_existing_thread(event)
        
        assert result is False
    
    def test_is_request_in_existing_thread_false_original_message(self):
        """Test detection for original thread message"""
        event = {
            "thread_ts": "1234567890.123",
            "ts": "1234567890.123"  # Same as thread_ts (original message)
        }
        
        result = is_request_in_existing_thread(event)
        
        assert result is False
