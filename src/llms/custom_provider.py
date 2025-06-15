# src/llms/custom_provider.py
# Copyright (c) 2025 - Custom LLM Provider Implementation
# Future-proof provider switching for DeerFlow

import requests
import json
import os
import logging
from typing import Dict, Any, List, Iterator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Configure logging for provider issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Custom exception for provider-specific errors"""
    def __init__(self, provider: str, message: str, status_code: int = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class BaseLLMProvider:
    """Base class for all LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.model = config.get("model", "").replace(self._get_provider_prefix(), "")
        self.provider_name = self.__class__.__name__.replace("Provider", "")
        
        # Validate configuration
        if not self.api_key:
            raise ProviderError(self.provider_name, "API key is required")
        if not self.base_url:
            raise ProviderError(self.provider_name, "Base URL is required")
    
    def _get_provider_prefix(self) -> str:
        return ""
    
    def _format_messages(self, messages) -> List[Dict]:
        """Convert LangChain messages to API format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, str):
                formatted.append({"role": "user", "content": msg})
        return formatted
    
    def _handle_api_error(self, response, operation: str = "API call"):
        """Centralized error handling with detailed logging"""
        error_msg = f"{operation} failed - Status: {response.status_code}"
        try:
            error_detail = response.json().get("error", {})
            if isinstance(error_detail, dict):
                error_msg += f" - {error_detail.get('message', response.text)}"
            else:
                error_msg += f" - {response.text}"
        except:
            error_msg += f" - {response.text}"
        
        logger.error(f"[{self.provider_name}] {error_msg}")
        raise ProviderError(self.provider_name, error_msg, response.status_code)
    
    def invoke(self, messages) -> str:
        """Synchronous message completion"""
        raise NotImplementedError
    
    def stream(self, messages) -> Iterator[str]:
        """Streaming message completion"""
        # Fallback to invoke if streaming not implemented
        response = self.invoke(messages)
        yield response


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API Provider - Cost effective testing"""
    
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
            "temperature": 0.7
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
            
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[DeepSeek] Network error: {e}")
            raise ProviderError("DeepSeek", f"Network error: {e}")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini Provider - 1M token research volume"""
    
    def _get_provider_prefix(self) -> str:
        return "gemini/"
    
    def invoke(self, messages) -> str:
        headers = {"Content-Type": "application/json"}
        
        # Convert messages to Gemini format
        gemini_messages = []
        for msg_dict in self._format_messages(messages):
            if msg_dict["role"] == "user":
                gemini_messages.append({
                    "parts": [{"text": msg_dict["content"]}]
                })
        
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 8192
            }
        }
        
        try:
            url = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                self._handle_api_error(response, "Gemini completion")
            
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[Gemini] Network error: {e}")
            raise ProviderError("Gemini", f"Network error: {e}")


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude Provider - Premium reasoning"""
    
    def _get_provider_prefix(self) -> str:
        return "claude/"
    
    def invoke(self, messages) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        claude_messages = self._format_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": 8192,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                self._handle_api_error(response, "Claude completion")
            
            return response.json()["content"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[Claude] Network error: {e}")
            raise ProviderError("Claude", f"Network error: {e}")


class QwenProvider(BaseLLMProvider):
    """Alibaba Qwen Provider via DashScope API"""
    
    def _get_provider_prefix(self) -> str:
        return "qwen/"
    
    def invoke(self, messages) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": {
                "messages": self._format_messages(messages)
            },
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 8192
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/services/aigc/text-generation/generation",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                self._handle_api_error(response, "Qwen completion")
            
            result = response.json()
            return result["output"]["text"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[Qwen] Network error: {e}")
            raise ProviderError("Qwen", f"Network error: {e}")


class GroqProvider(BaseLLMProvider):
    """Groq Provider - Ultra-fast inference"""
    
    def _get_provider_prefix(self) -> str:
        return "groq/"
    
    def invoke(self, messages) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": 0.7,
            "max_tokens": 8192
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                self._handle_api_error(response, "Groq completion")
            
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[Groq] Network error: {e}")
            raise ProviderError("Groq", f"Network error: {e}")


class XAIProvider(BaseLLMProvider):
    """xAI (Elon Musk) Provider - Grok models"""
    
    def _get_provider_prefix(self) -> str:
        return "xai/"
    
    def invoke(self, messages) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": 0.7,
            "max_tokens": 8192
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                self._handle_api_error(response, "xAI completion")
            
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[xAI] Network error: {e}")
            raise ProviderError("xAI", f"Network error: {e}")


class PerplexityProvider(BaseLLMProvider):
    """Perplexity Provider - Research-focused AI"""
    
    def _get_provider_prefix(self) -> str:
        return "perplexity/"
    
    def invoke(self, messages) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": 0.7,
            "max_tokens": 8192
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                self._handle_api_error(response, "Perplexity completion")
            
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[Perplexity] Network error: {e}")
            raise ProviderError("Perplexity", f"Network error: {e}")


def create_custom_provider(config: Dict[str, Any]) -> BaseLLMProvider:
    """Factory function to create the appropriate provider with explicit error handling"""
    model = config.get("model", "").lower()
    
    # Provider mapping with clear error messages
    provider_map = {
        "deepseek": DeepSeekProvider,
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
        "qwen": QwenProvider,
        "groq": GroqProvider,
        "xai": XAIProvider,
        "perplexity": PerplexityProvider
    }
    
    # Detect provider
    detected_provider = None
    for provider_name, provider_class in provider_map.items():
        if provider_name in model:
            detected_provider = provider_class
            break
    
    if not detected_provider:
        available_providers = list(provider_map.keys())
        error_msg = f"Unknown provider in model '{model}'. Available providers: {available_providers}"
        logger.error(error_msg)
        raise ProviderError("ProviderFactory", error_msg)
    
    logger.info(f"Creating provider: {detected_provider.__name__} for model: {model}")
    return detected_provider(config)


# Compatibility wrapper to work with existing DeerFlow code
class CustomLLMWrapper:
    """Wrapper to make custom providers compatible with ChatOpenAI interface"""
    
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        logger.info(f"Initialized {provider.provider_name} provider with model: {provider.model}")
    
    def invoke(self, messages):
        """
        Invoke with explicit error handling - NO SILENT FALLBACKS
        """
        try:
            logger.info(f"[{self.provider.provider_name}] Processing request...")
            result = self.provider.invoke(messages)
            logger.info(f"[{self.provider.provider_name}] Request successful")
            return result
            
        except ProviderError as e:
            # Log the specific provider error
            logger.error(f"Provider {e.provider} failed: {e}")
            # Re-raise the error instead of falling back silently
            raise Exception(f"LLM Provider Error - {e.provider}: {str(e)}. Please check your API configuration and credits.")
            
        except Exception as e:
            logger.error(f"[{self.provider.provider_name}] Unexpected error: {e}")
            raise Exception(f"LLM Provider Unexpected Error - {self.provider.provider_name}: {str(e)}")
    
    def stream(self, messages):
        try:
            return self.provider.stream(messages)
        except Exception as e:
            logger.error(f"[{self.provider.provider_name}] Streaming error: {e}")
            raise Exception(f"LLM Provider Streaming Error - {self.provider.provider_name}: {str(e)}")


# Health check function for monitoring
def check_provider_health(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test provider connectivity and return status
    """
    try:
        provider = create_custom_provider(config)
        test_response = provider.invoke("Hello")
        return {
            "status": "healthy",
            "provider": provider.provider_name,
            "model": provider.model,
            "test_response_length": len(test_response)
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "provider": config.get("model", "unknown"),
            "error": str(e)
        }
