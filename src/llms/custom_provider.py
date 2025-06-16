# src/llms/custom_provider.py
# Copyright (c) 2025 - Custom LLM Provider Implementation for DeerFlow

import requests
import json
import os
import logging
from typing import Dict, Any, List, Iterator, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Configure logging for provider debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Custom exception for provider-specific errors with detailed context"""
    def __init__(self, provider: str, message: str, status_code: int = None, response_text: str = None):
        self.provider = provider
        self.status_code = status_code
        self.response_text = response_text
        error_msg = f"[{provider}] {message}"
        if status_code:
            error_msg += f" (HTTP {status_code})"
        if response_text:
            error_msg += f" - Response: {response_text[:200]}"
        super().__init__(error_msg)


class BaseLLMProvider:
    """Base class for all LLM providers with enhanced error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.model = config.get("model", "").replace(self._get_provider_prefix(), "")
        self.provider_name = self.__class__.__name__.replace("Provider", "")
        
        # Enhanced configuration validation
        if not self.api_key:
            raise ProviderError(self.provider_name, "API key is required - check your .env file")
        if not self.base_url:
            raise ProviderError(self.provider_name, "Base URL is required - check your conf.yaml")
        if not self.model:
            raise ProviderError(self.provider_name, "Model name is required")
            
        logger.info(f"Initialized {self.provider_name} provider - Model: {self.model}")
    
    def _get_provider_prefix(self) -> str:
        """Override in subclasses to define model prefix (e.g., 'gemini/', 'deepseek/')"""
        return ""
    
    def _format_messages(self, messages: Union[str, List, BaseMessage]) -> List[Dict]:
        """Convert various message formats to standardized format"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        
        if isinstance(messages, list):
            formatted = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    formatted.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    formatted.append({"role": "system", "content": msg.content})
                elif isinstance(msg, dict):
                    formatted.append(msg)
                elif isinstance(msg, str):
                    formatted.append({"role": "user", "content": msg})
            return formatted
        
        # Single message object
        if hasattr(messages, 'content'):
            return [{"role": "user", "content": messages.content}]
        
        return [{"role": "user", "content": str(messages)}]
    
    def _handle_api_error(self, response, operation: str = "API call"):
        """Enhanced error handling with response debugging"""
        try:
            error_detail = response.json()
            error_msg = f"{operation} failed"
            
            # Extract specific error information
            if "error" in error_detail:
                error_info = error_detail["error"]
                if isinstance(error_info, dict):
                    error_msg += f" - {error_info.get('message', 'Unknown error')}"
                    if "code" in error_info:
                        error_msg += f" (Code: {error_info['code']})"
                else:
                    error_msg += f" - {error_info}"
            else:
                error_msg += f" - {response.text[:200]}"
                
        except json.JSONDecodeError:
            error_msg = f"{operation} failed - Invalid JSON response: {response.text[:200]}"
        
        logger.error(f"[{self.provider_name}] {error_msg}")
        raise ProviderError(self.provider_name, error_msg, response.status_code, response.text)
    
    def invoke(self, messages) -> str:
        """Synchronous message completion - override in subclasses"""
        raise NotImplementedError(f"{self.provider_name} must implement invoke method")
    
    def stream(self, messages) -> Iterator[str]:
        """Streaming completion - fallback to invoke if not implemented"""
        response = self.invoke(messages)
        yield response


class GeminiProvider(BaseLLMProvider):
    """Google Gemini Provider - Optimized for large context B2B research"""
    
    def _get_provider_prefix(self) -> str:
        return "gemini/"
    
    def _convert_to_gemini_format(self, messages: List[Dict]) -> List[Dict]:
        """Convert standard messages to Gemini's specific format"""
        gemini_contents = []
        
        for msg in messages:
            if msg["role"] in ["user", "system"]:
                # Gemini treats system messages as user messages
                gemini_contents.append({
                    "parts": [{"text": msg["content"]}]
                })
        
        return gemini_contents
    
    def invoke(self, messages) -> str:
        """Gemini-specific API call with enhanced error handling"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Convert messages to Gemini format
        standard_messages = self._format_messages(messages)
        gemini_contents = self._convert_to_gemini_format(standard_messages)
        
        # Gemini-specific payload structure
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": self.config.get("temperature", 0.7),
                "maxOutputTokens": self.config.get("max_tokens", 32768),
                "candidateCount": 1,
                "stopSequences": []
            }
        }
        
        # Gemini URL format: /v1beta/models/{model}:generateContent?key={api_key}
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        logger.info(f"[Gemini] Making request to: {url}")
        logger.debug(f"[Gemini] Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=60  # Longer timeout for large context
            )
            
            logger.info(f"[Gemini] Response status: {response.status_code}")
            
            if response.status_code != 200:
                self._handle_api_error(response, "Gemini content generation")
            
            # Parse Gemini response format
            result = response.json()
            logger.debug(f"[Gemini] Response: {json.dumps(result, indent=2)}")
            
            # Validate response structure
            if "candidates" not in result:
                raise ProviderError("Gemini", f"Invalid response format - missing 'candidates': {result}")
            
            if not result["candidates"]:
                raise ProviderError("Gemini", f"No candidates in response: {result}")
            
            candidate = result["candidates"][0]
            
            if "content" not in candidate:
                raise ProviderError("Gemini", f"No content in candidate: {candidate}")
            
            if "parts" not in candidate["content"]:
                raise ProviderError("Gemini", f"No parts in content: {candidate['content']}")
            
            if not candidate["content"]["parts"]:
                raise ProviderError("Gemini", f"Empty parts array: {candidate['content']}")
            
            response_text = candidate["content"]["parts"][0]["text"]
            logger.info(f"[Gemini] Successfully generated {len(response_text)} characters")
            
            return response_text
            
        except requests.exceptions.Timeout:
            error_msg = "Request timeout - try reducing input size or increasing timeout"
            logger.error(f"[Gemini] {error_msg}")
            raise ProviderError("Gemini", error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            logger.error(f"[Gemini] {error_msg}")
            raise ProviderError("Gemini", error_msg)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek Provider - Cost-effective fallback"""
    
    def _get_provider_prefix(self) -> str:
        return "deepseek/"
    
    def invoke(self, messages) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 8192)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                self._handle_api_error(response, "DeepSeek completion")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[DeepSeek] Network error: {e}")
            raise ProviderError("DeepSeek", f"Network error: {e}")


def create_custom_provider(config: Dict[str, Any]) -> BaseLLMProvider:
    """Factory function to create appropriate provider with enhanced detection"""
    model = config.get("model", "").lower()
    
    # Provider detection logic
    if "gemini" in model:
        logger.info(f"Detected Gemini provider for model: {model}")
        return GeminiProvider(config)
    elif "deepseek" in model:
        logger.info(f"Detected DeepSeek provider for model: {model}")
        return DeepSeekProvider(config)
    else:
        # Default to Gemini if model name is unclear
        logger.warning(f"Unknown model '{model}', defaulting to Gemini provider")
        return GeminiProvider(config)


class CustomLLMWrapper:
    """Wrapper to make custom providers compatible with existing DeerFlow code"""
    
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        logger.info(f"Initialized CustomLLMWrapper with {provider.provider_name} provider")
    
    def invoke(self, messages):
        """Main invoke method with comprehensive error handling"""
        try:
            logger.info(f"[{self.provider.provider_name}] Processing request...")
            
            # Input validation
            if not messages:
                raise ValueError("Messages cannot be empty")
            
            result = self.provider.invoke(messages)
            
            if not result:
                raise ProviderError(self.provider.provider_name, "Empty response received")
            
            logger.info(f"[{self.provider.provider_name}] Successfully processed request")
            return result
            
        except ProviderError as e:
            # Re-raise provider errors with context
            logger.error(f"Provider error: {e}")
            raise Exception(f"LLM Provider Error: {str(e)}")
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error in {self.provider.provider_name}: {str(e)}"
            logger.error(error_msg)
            raise Exception(f"LLM Provider Unexpected Error: {error_msg}")
    
    def stream(self, messages):
        """Streaming interface"""
        try:
            return self.provider.stream(messages)
        except Exception as e:
            logger.error(f"[{self.provider.provider_name}] Streaming error: {e}")
            raise Exception(f"LLM Provider Streaming Error: {str(e)}")


def test_gemini_provider(api_key: str) -> Dict[str, Any]:
    """Test function to validate Gemini setup"""
    config = {
        "model": "gemini-2.0-flash-001",
        "api_key": api_key,
        "base_url": "https://generativelanguage.googleapis.com",
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        provider = GeminiProvider(config)
        wrapper = CustomLLMWrapper(provider)
        
        test_response = wrapper.invoke("Hello! Please respond with 'Gemini is working correctly.'")
        
        return {
            "status": "success",
            "provider": "Gemini",
            "model": config["model"],
            "response_length": len(test_response),
            "response_preview": test_response[:100] + "..." if len(test_response) > 100 else test_response
        }
        
    except Exception as e:
        return {
            "status": "error",
            "provider": "Gemini",
            "error": str(e)
        }


if __name__ == "__main__":
    # Quick test if run directly
    import os
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        result = test_gemini_provider(api_key)
        print(f"Test result: {json.dumps(result, indent=2)}")
    else:
        print("GOOGLE_API_KEY not found in environment")
