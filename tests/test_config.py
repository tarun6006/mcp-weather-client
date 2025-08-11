"""
Black box tests for configuration module
Tests external behavior without knowing internal implementation
"""
import pytest
import os
from unittest.mock import patch, mock_open
from config.settings import (
    load_config, log_configuration, SLACK_SECRET, SLACK_TOKEN,
    WEATHER_MCP_URL, CALC_MCP_URL, GOOGLE_API_KEY, GEMINI_MODEL
)
from config.logging_config import setup_logging, safe_log_token, safe_debug_log

class TestConfigSettings:
    """Test configuration loading and environment variables"""
    
    def test_load_config_success(self):
        """Test successful config loading"""
        mock_yaml_content = """
        test_key: test_value
        nested:
          key: nested_value
        """
        
        with patch('builtins.open', mock_open(read_data=mock_yaml_content)):
            with patch('yaml.safe_load') as mock_yaml:
                mock_yaml.return_value = {"test_key": "test_value"}
                config = load_config()
                assert config is not None
                mock_yaml.assert_called_once()
    
    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                load_config()
    
    def test_environment_variables_loaded(self):
        """Test that environment variables are properly loaded"""
        # These should be strings or None, not raise exceptions
        assert isinstance(SLACK_SECRET, (str, type(None)))
        assert isinstance(SLACK_TOKEN, (str, type(None)))
        assert isinstance(GOOGLE_API_KEY, (str, type(None)))
        assert isinstance(GEMINI_MODEL, str)
        assert isinstance(WEATHER_MCP_URL, str)
        assert isinstance(CALC_MCP_URL, str)
    
    def test_default_values(self):
        """Test default values are set correctly"""
        assert GEMINI_MODEL == "gemini-2.5-pro"
        assert WEATHER_MCP_URL is not None
        assert "localhost" in CALC_MCP_URL
    
    @patch('config.settings.logger')
    def test_log_configuration(self, mock_logger):
        """Test configuration logging"""
        log_configuration()
        # Should log multiple configuration lines
        assert mock_logger.info.call_count >= 5
    
    def test_config_structure_validation(self):
        """Test that configuration has expected structure"""
        from config.settings import CONFIG
        
        # Test that main configuration sections exist
        assert 'mcp_tools' in CONFIG
        assert 'gemini_instructions' in CONFIG
        assert 'client_messages' in CONFIG
        assert 'location_extraction' in CONFIG
        
        # Test MCP tools structure
        mcp_tools = CONFIG['mcp_tools']
        assert isinstance(mcp_tools, dict)
        assert 'get_weather' in mcp_tools
        assert len(mcp_tools) >= 10  # Should have weather + calculator tools
        
        # Test client messages structure
        client_messages = CONFIG['client_messages']
        assert 'error_messages' in client_messages
        assert 'response_messages' in client_messages
        assert 'tool_categories' in client_messages
        
        # Test error messages
        error_messages = client_messages['error_messages']
        expected_error_keys = [
            'no_weather_data', 'no_calculation_result', 'empty_input',
            'safety_restriction', 'understanding_error', 'processing_error',
            'gemini_not_configured'
        ]
        for key in expected_error_keys:
            assert key in error_messages
            assert isinstance(error_messages[key], str)
            assert len(error_messages[key]) > 0
        
        # Test response messages
        response_messages = client_messages['response_messages']
        expected_response_keys = [
            'weather_prefix', 'temperature_label', 'conditions_label',
            'result_prefix', 'generic_result', 'weather_data_fallback'
        ]
        for key in expected_response_keys:
            assert key in response_messages
            assert isinstance(response_messages[key], str)
            assert len(response_messages[key]) > 0
        
        # Test calculator tools array
        tool_categories = client_messages['tool_categories']
        assert 'all_calculator_tools' in tool_categories
        calculator_tools = tool_categories['all_calculator_tools']
        assert isinstance(calculator_tools, list)
        assert len(calculator_tools) == 10  # Should have exactly 10 calculator tools
        
        expected_calc_tools = [
            'add', 'subtract', 'multiply', 'divide', 'power',
            'sqrt', 'factorial', 'modulo', 'absolute', 'parse_expression'
        ]
        for tool in expected_calc_tools:
            assert tool in calculator_tools
    
    def test_gemini_instructions_structure(self):
        """Test Gemini instructions configuration structure"""
        from config.settings import CONFIG
        
        gemini_instructions = CONFIG['gemini_instructions']
        
        # Test required sections
        assert 'system_prompt_template' in gemini_instructions
        assert 'critical_instructions' in gemini_instructions
        assert 'tool_call_formats' in gemini_instructions
        assert 'examples' in gemini_instructions
        
        # Test critical instructions
        critical_instructions = gemini_instructions['critical_instructions']
        assert isinstance(critical_instructions, list)
        assert len(critical_instructions) >= 6
        
        # Test tool call formats
        tool_call_formats = gemini_instructions['tool_call_formats']
        assert 'weather_queries' in tool_call_formats
        assert 'math_calculations' in tool_call_formats
        assert 'non_supported_query' in tool_call_formats
        
        # Test examples
        examples = gemini_instructions['examples']
        assert 'weather' in examples
        assert 'math' in examples
        assert isinstance(examples['weather'], list)
        assert isinstance(examples['math'], list)
        assert len(examples['weather']) >= 3
        assert len(examples['math']) >= 3

class TestLoggingConfig:
    """Test logging configuration functionality"""
    
    @patch('logging.basicConfig')
    @patch('logging.getLogger')
    def test_setup_logging(self, mock_get_logger, mock_basic_config):
        """Test logging setup"""
        mock_logger = mock_get_logger.return_value
        
        setup_logging()
        
        mock_basic_config.assert_called_once()
        # Should set log levels for external libraries
        assert mock_get_logger.call_count >= 3
        mock_logger.setLevel.assert_called()
    
    @patch('app.logger')
    def test_safe_log_token_with_long_token(self, mock_logger):
        """Test token logging with long token"""
        token = "xoxb-1234567890-abcdefghijklmnop"
        safe_log_token("TEST_TOKEN", token)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "xoxb" in call_args
        assert "mnop" in call_args
        assert "1234567890" not in call_args  # Middle should be masked
    
    @patch('app.logger')
    def test_safe_log_token_with_short_token(self, mock_logger):
        """Test token logging with short token"""
        token = "short"
        safe_log_token("TEST_TOKEN", token)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "***" in call_args
    
    @patch('app.logger')
    def test_safe_log_token_with_none(self, mock_logger):
        """Test token logging with None value"""
        safe_log_token("TEST_TOKEN", None)
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "NOT SET" in call_args
    
    @patch('app.logger')
    def test_safe_debug_log_with_sensitive_data(self, mock_logger):
        """Test debug logging with sensitive data"""
        safe_debug_log("Test message", sensitive_data={"key": "value"})
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "SENSITIVE DATA MASKED" in call_args
        assert "value" not in call_args
    
    @patch('app.logger')
    def test_safe_debug_log_without_sensitive_data(self, mock_logger):
        """Test debug logging without sensitive data"""
        message = "Regular debug message"
        safe_debug_log(message)
        
        mock_logger.debug.assert_called_once_with(message)
