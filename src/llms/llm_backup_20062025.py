# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Dict, Union
import os
import logging

from langchain_openai import ChatOpenAI
from src.config import load_yaml_config
from src.config.agents import LLMType

# Import custom provider system
try:
    from src.llms.custom_provider import create_custom_provider, CustomLLMWrapper, ProviderError
    CUSTOM_PROVIDERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Custom providers not available: {e}")
    CUSTOM_PROVIDERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for LLM instances
_llm_cache: dict[LLMType, Union[ChatOpenAI, CustomLLMWrapper]] = {}


def _get_env_llm_conf(llm_type: str) -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.
    Environment variables should follow the format: {LLM_TYPE}_MODEL__{KEY}
    e.g., BASIC_MODEL__api_key, BASIC_MODEL__base_url
    """
    prefix = f"{llm_type.upper()}_MODEL__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix):].lower()
            conf[conf_key] = value
    return conf


def _is_custom_provider(model_name: str) -> bool:
    """
    Check if model requires custom provider implementation.
    Returns True for providers that need special API format handling.
    """
    if not CUSTOM_PROVIDERS_AVAILABLE:
        return False
    
    custom_providers = [
        "gemini",      # Google Gemini (different API format)
        "deepseek",    # DeepSeek (for consistency)
        "claude",      # Anthropic Claude
        "perplexity",  # Perplexity AI
        "qwen",        # Alibaba Qwen
        "groq",        # Groq
        "xai",         # xAI (Grok)
    ]
    
    model_lower = model_name.lower()
    is_custom = any(provider in model_lower for provider in custom_providers)
    
    if is_custom:
        logger.info(f"Detected custom provider needed for model: {model_name}")
    
    return is_custom


def _create_custom_llm(llm_conf: Dict[str, Any]) -> CustomLLMWrapper:
    """Create custom provider LLM instance with validation."""
    try:
        # Validate required configuration
        required_fields = ["model", "api_key", "base_url"]
        missing_fields = [field for field in required_fields if not llm_conf.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        # Create provider
        provider = create_custom_provider(llm_conf)
        wrapper = CustomLLMWrapper(provider)
        
        logger.info(f"Successfully created custom provider: {provider.provider_name}")
        return wrapper
        
    except Exception as e:
        logger.error(f"Failed to create custom provider: {e}")
        raise ValueError(f"Custom provider creation failed: {e}")


def _create_standard_llm(llm_conf: Dict[str, Any]) -> ChatOpenAI:
    """Create standard ChatOpenAI instance for OpenAI-compatible APIs."""
    try:
        # Remove custom provider specific configurations that ChatOpenAI doesn't understand
        openai_conf = {k: v for k, v in llm_conf.items() 
                      if k in ["model", "api_key", "base_url", "temperature", "max_tokens"]}
        
        # Map max_tokens to max_completion_tokens for ChatOpenAI
        if "max_tokens" in openai_conf:
            openai_conf["max_completion_tokens"] = openai_conf.pop("max_tokens")
        
        logger.info(f"Creating standard ChatOpenAI for model: {openai_conf.get('model')}")
        return ChatOpenAI(**openai_conf)
        
    except Exception as e:
        logger.error(f"Failed to create standard LLM: {e}")
        raise ValueError(f"Standard LLM creation failed: {e}")


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> Union[ChatOpenAI, CustomLLMWrapper]:
    """
    Create LLM instance based on configuration with intelligent provider routing.
    """
    # Get type-specific configuration
    llm_type_map = {
        "reasoning": conf.get("REASONING_MODEL", {}),
        "basic": conf.get("BASIC_MODEL", {}),
        "vision": conf.get("VISION_MODEL", {}),
    }
    
    llm_conf = llm_type_map.get(llm_type)
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM configuration for type '{llm_type}': {llm_conf}")
    
    # Get configuration from environment variables (higher priority)
    env_conf = _get_env_llm_conf(llm_type)
    
    # Merge configurations with environment variables taking precedence
    merged_conf = {**llm_conf, **env_conf}
    
    if not merged_conf:
        raise ValueError(f"No configuration found for LLM type: {llm_type}")
    
    # Validate essential fields
    model_name = merged_conf.get("model", "")
    if not model_name:
        raise ValueError(f"Model name is required for LLM type: {llm_type}")
    
    # Route to appropriate provider
    if _is_custom_provider(model_name):
        logger.info(f"Using custom provider for {llm_type}: {model_name}")
        return _create_custom_llm(merged_conf)
    else:
        logger.info(f"Using standard ChatOpenAI for {llm_type}: {model_name}")
        return _create_standard_llm(merged_conf)


def get_llm_by_type(llm_type: LLMType) -> Union[ChatOpenAI, CustomLLMWrapper]:
    """
    Get LLM instance by type with caching.
    Returns either ChatOpenAI or CustomLLMWrapper depending on the provider.
    """
    # Return cached instance if available
    if llm_type in _llm_cache:
        logger.debug(f"Returning cached LLM for type: {llm_type}")
        return _llm_cache[llm_type]
    
    try:
        # Load configuration
        conf_path = Path(__file__).parent.parent.parent / "conf.yaml"
        conf = load_yaml_config(str(conf_path.resolve()))
        
        # Create LLM instance
        llm = _create_llm_use_conf(llm_type, conf)
        
        # Cache the instance
        _llm_cache[llm_type] = llm
        logger.info(f"Created and cached LLM for type: {llm_type}")
        
        return llm
        
    except Exception as e:
        logger.error(f"Failed to get LLM for type '{llm_type}': {e}")
        raise


def clear_llm_cache():
    """Clear the LLM cache - useful for testing or configuration changes."""
    global _llm_cache
    _llm_cache.clear()
    logger.info("LLM cache cleared")


def get_cached_llm_info() -> Dict[str, Any]:
    """Get information about currently cached LLM instances."""
    info = {}
    for llm_type, llm_instance in _llm_cache.items():
        if isinstance(llm_instance, CustomLLMWrapper):
            info[llm_type] = {
                "type": "CustomLLMWrapper",
                "provider": llm_instance.provider.provider_name,
                "model": llm_instance.provider.model
            }
        elif isinstance(llm_instance, ChatOpenAI):
            info[llm_type] = {
                "type": "ChatOpenAI",
                "model": getattr(llm_instance, 'model_name', 'unknown')
            }
        else:
            info[llm_type] = {
                "type": type(llm_instance).__name__,
                "details": "unknown"
            }
    return info


def test_llm_configuration() -> Dict[str, Any]:
    """
    Test all configured LLM types to ensure they're working properly.
    Returns status information for each type.
    """
    results = {}
    
    for llm_type in ["basic", "reasoning", "vision"]:
        try:
            llm = get_llm_by_type(llm_type)
            
            # Test with a simple prompt
            test_response = llm.invoke("Hello! Please respond with 'OK' if you're working.")
            
            results[llm_type] = {
                "status": "success",
                "response_length": len(test_response),
                "llm_type": type(llm).__name__
            }
            
        except Exception as e:
            results[llm_type] = {
                "status": "error", 
                "error": str(e)
            }
    
    return results


# Convenience functions for specific research tasks
def get_research_llm(context_size: str = "large") -> Union[ChatOpenAI, CustomLLMWrapper]:
    """
    Get LLM optimized for research tasks based on required context size.
    
    Args:
        context_size: "small" (8K), "large" (32K+), or "huge" (1M+)
    """
    if context_size == "small":
        return get_llm_by_type("basic")      # Fast, cost-effective
    elif context_size == "large":
        return get_llm_by_type("reasoning")  # Balanced performance
    else:  # huge
        return get_llm_by_type("reasoning")  # Maximum context


def get_vision_llm() -> Union[ChatOpenAI, CustomLLMWrapper]:
    """Get LLM capable of processing images and vision tasks."""
    return get_llm_by_type("vision")


# Legacy compatibility
def _create_deepseek_client(conf: Dict[str, Any]):
    """
    Legacy DeepSeek client - kept for backward compatibility.
    New implementations should use the custom provider system.
    """
    logger.warning("Using legacy DeepSeek client - consider upgrading to custom provider")
    
    import requests
    
    class SimpleDeepSeek:
        def __init__(self, config):
            self.api_key = config["api_key"]
            self.base_url = config["base_url"]
            self.model = config["model"]
            
        def invoke(self, messages):
            headers = {
                "Authorization": f"Bearer {self.api_key}", 
                "Content-Type": "application/json"
            }
            
            # Convert message to simple format
            if isinstance(messages, str):
                formatted_messages = [{"role": "user", "content": messages}]
            else:
                formatted_messages = [{"role": "user", "content": str(messages)}]
            
            payload = {
                "model": self.model.replace("deepseek/", ""),
                "messages": formatted_messages,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions", 
                headers=headers, 
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek API Error: {response.status_code} - {response.text}")
            
            return response.json()["choices"][0]["message"]["content"]
    
    return SimpleDeepSeek(conf)


if __name__ == "__main__":
    # Test the LLM system
    try:
        logger.info("Testing DeerFlow LLM system...")
        
        # Test basic functionality
        basic_llm = get_llm_by_type("basic")
        print(f"✅ Basic LLM initialized: {type(basic_llm).__name__}")
        
        # Test configuration
        test_results = test_llm_configuration()
        print(f"✅ Configuration test completed: {len(test_results)} types tested")
        
        # Show cache info
        cache_info = get_cached_llm_info()
        print(f"✅ Cache info: {cache_info}")
        
    except Exception as e:
        print(f"❌ LLM system test failed: {e}")
        logger.error(f"LLM system test failed: {e}")
