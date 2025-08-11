"""
Tests for Gemini client configuration loading and message handling
"""
import pytest
from unittest.mock import patch, MagicMock
from services.gemini_client import GeminiMCPClient


class TestGeminiClientConfiguration:
    """Test Gemini client configuration loading"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration data"""
        return {
            'mcp_tools': {
                'get_weather': {
                    'description': 'Weather tool',
                    'server': 'weather'
                },
                'add': {
                    'description': 'Addition tool',
                    'server': 'calculator'
                }
            },
            'client_messages': {
                'error_messages': {
                    'no_weather_data': 'Test: No weather data received',
                    'no_calculation_result': 'Test: No calculation result received',
                    'empty_input': 'Test: Please specify input',
                    'safety_restriction': 'Test: Safety restriction',
                    'understanding_error': 'Test: Understanding error',
                    'processing_error': 'Test: Processing error',
                    'gemini_not_configured': 'Test: Gemini not configured'
                },
                'response_messages': {
                    'weather_prefix': 'Test Weather for',
                    'temperature_label': 'Test Temperature:',
                    'conditions_label': 'Test Conditions:',
                    'result_prefix': 'Test result is',
                    'generic_result': 'Test Result',
                    'weather_data_fallback': 'Test Weather data'
                },
                'tool_categories': {
                    'all_calculator_tools': [
                        'add', 'subtract', 'multiply', 'divide',
                        'power', 'sqrt', 'factorial', 'modulo',
                        'absolute', 'parse_expression'
                    ]
                }
            },
            'gemini_instructions': {
                'system_prompt_template': 'Test system prompt template',
                'critical_instructions': [
                    'Test instruction 1',
                    'Test instruction 2'
                ],
                'tool_call_formats': {
                    'weather_queries': {
                        'city_format': 'Test city format',
                        'zip_format': 'Test zip format'
                    },
                    'math_calculations': {
                        'addition': 'Test addition format'
                    },
                    'non_supported_query': 'Test error format'
                },
                'examples': {
                    'weather': [
                        {'input': 'Test weather input', 'output': 'Test weather output'}
                    ],
                    'math': [
                        {'input': 'Test math input', 'output': 'Test math output'}
                    ]
                }
            }
        }
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_configuration_loading(self, mock_sse_client, mock_config_import, mock_config):
        """Test that configuration is loaded correctly during initialization"""
        mock_config_import.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        # Test tools schema loading
        assert client.tools_schema == mock_config['mcp_tools']
        
        # Test client messages loading
        assert client.error_messages == mock_config['client_messages']['error_messages']
        assert client.response_messages == mock_config['client_messages']['response_messages']
        
        # Test calculator tools array loading
        expected_tools = mock_config['client_messages']['tool_categories']['all_calculator_tools']
        assert client.calculator_tools == expected_tools
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_error_message_usage(self, mock_sse_client, mock_config_import, mock_config):
        """Test that error messages from config are used correctly"""
        mock_config_import.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        # Test weather error message
        result = client._format_response({}, 'get_weather', {}, 'test')
        assert result == 'Test: No weather data received'
        
        # Test calculation error message
        result = client._format_response({}, 'add', {}, 'test')
        assert result == 'Test: No calculation result received'
        
        # Test empty input message
        result = client.process_natural_language('')
        assert result == 'Test: Please specify input'
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_response_message_usage(self, mock_sse_client, mock_config_import, mock_config):
        """Test that response messages from config are used correctly"""
        mock_config_import.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        # Test weather response formatting
        weather_data = {
            'result': {
                'properties': {
                    'location': 'Test City',
                    'temperature': '72°F',
                    'shortForecast': 'Sunny'
                }
            }
        }
        result = client._format_response(weather_data, 'get_weather', {}, 'test')
        
        assert 'Test Weather for Test City' in result
        assert 'Test Temperature: 72°F' in result
        assert 'Test Conditions: Sunny' in result
        
        # Test calculation result formatting
        calc_data = {'result': 42}
        result = client._format_response(calc_data, 'add', {}, 'test')
        assert result == 'Test result is: 42'
        
        # Test generic result formatting
        generic_data = {'some': 'data'}
        result = client._format_response(generic_data, 'unknown_tool', {}, 'test')
        assert result == 'Test Result: {\'some\': \'data\'}'
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_calculator_tools_array_usage(self, mock_sse_client, mock_config_import, mock_config):
        """Test that calculator tools array from config is used correctly"""
        mock_config_import.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        # Test that calculator tools are recognized
        calc_data = {'result': 100}
        
        for tool_name in client.calculator_tools:
            result = client._format_response(calc_data, tool_name, {}, 'test')
            assert 'Test result is: 100' == result
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_tool_call_format_generation(self, mock_sse_client, mock_config_import, mock_config):
        """Test dynamic tool call format generation"""
        mock_config_import.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        tool_call_formats = mock_config['gemini_instructions']['tool_call_formats']
        result = client._generate_tool_call_formats(tool_call_formats)
        
        assert 'Weather queries:' in result
        assert 'Test city format' in result
        assert 'Test zip format' in result
        assert 'Math calculations:' in result
        assert 'Test addition format' in result
        assert 'Test error format' in result
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_examples_generation(self, mock_sse_client, mock_config_import, mock_config):
        """Test dynamic examples generation"""
        mock_config_import.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        examples = mock_config['gemini_instructions']['examples']
        result = client._generate_examples(examples)
        
        assert 'Weather examples:' in result
        assert 'Test weather input' in result
        assert 'Test weather output' in result
        assert 'Math examples:' in result
        assert 'Test math input' in result
        assert 'Test math output' in result
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_fallback_to_defaults(self, mock_sse_client, mock_config_import):
        """Test that fallback defaults work when config is missing"""
        # Mock empty configuration
        mock_config_import.get.return_value = {}
        
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        # Test that defaults are used when config is missing
        assert client.tools_schema == {}
        assert client.error_messages == {}
        assert client.response_messages == {}
        assert client.calculator_tools == []
        
        # Test that fallback strings are used in methods
        result = client._format_response({}, 'get_weather', {}, 'test')
        assert 'No weather data received from the server.' in result
        
        result = client._format_response({}, 'add', {}, 'test')
        assert 'No calculation result received from the server.' in result


class TestConfigurationValidation:
    """Test configuration validation and error handling"""
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_missing_config_sections(self, mock_sse_client, mock_config_import):
        """Test behavior when configuration sections are missing"""
        # Mock partially missing configuration
        partial_config = {
            'mcp_tools': {'get_weather': {'description': 'Weather tool'}},
            # Missing client_messages and gemini_instructions
        }
        mock_config_import.get.side_effect = lambda key, default=None: partial_config.get(key, default)
        
        # Should not raise exceptions during initialization
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        # Should use empty defaults
        assert client.error_messages == {}
        assert client.response_messages == {}
        assert client.calculator_tools == []
    
    @patch('services.gemini_client.CONFIG')
    @patch('services.sse_client.SSECalculatorClient')
    def test_malformed_config_handling(self, mock_sse_client, mock_config_import):
        """Test handling of malformed configuration data"""
        # Mock malformed configuration
        malformed_config = {
            'client_messages': {
                'error_messages': 'not_a_dict',  # Should be a dict
                'tool_categories': {
                    'all_calculator_tools': 'not_a_list'  # Should be a list
                }
            }
        }
        mock_config_import.get.side_effect = lambda key, default=None: malformed_config.get(key, default)
        
        # Should not raise exceptions during initialization
        client = GeminiMCPClient('http://weather', 'calc', 8080, 'http')
        
        # Should handle malformed data gracefully
        assert client.error_messages == 'not_a_dict'  # Will be handled by .get() calls
        assert client.calculator_tools == 'not_a_list'  # Will be handled by tool checking
