# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# UPDATED: Now compatible with Vertex AI and your exact .env configuration

from pydantic import Field
from pathlib import Path
from typing import Any, Dict, Union
import os
import logging

from langchain_openai import ChatOpenAI
from src.config import load_yaml_config
from src.config.agents import LLMType
from src.config.agents import AGENT_LLM_MAP

# --- Avery's Diagnostic Imports: START ---
from src.prompts.planner_model import Plan
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.messages import HumanMessage
# --- Avery's Diagnostic Imports: END ---

# Import Vertex AI compatible custom provider system
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
    Updated to match your .env file structure
    """
    conf = {}
    
    # Map LLM types to your exact environment variable names
    if llm_type.lower() == "basic":
        conf.update({
            "model": os.getenv("GEMINI_BASIC_MODEL", "gemini-2.0-flash-001"),
            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
            "location": os.getenv("VERTEX_AI_LOCATION", "us-central1"),
            "api_key": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),  # Service account path
            "provider_type": "vertex_ai",
            "temperature": 0.7,
            "max_tokens": 8192
        })
    elif llm_type.lower() == "reasoning":
        conf.update({
            "model": os.getenv("GEMINI_REASONING_MODEL", "gemini-2.5-pro-preview-06-05"),
            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
            "location": os.getenv("VERTEX_AI_LOCATION", "us-central1"),
            "api_key": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),  # Service account path
            "provider_type": "vertex_ai",
            "temperature": 0.8,
            "max_tokens": 32768
        })
    elif llm_type.lower() == "vision":
        conf.update({
            "model": os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-pro-preview-06-05"),
            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
            "location": os.getenv("VERTEX_AI_LOCATION", "us-central1"),
            "api_key": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),  # Service account path
            "provider_type": "vertex_ai",
            "temperature": 0.7,
            "max_tokens": 32768,
            "supports_vision": True
        })
    
    # Add base URL for Vertex AI if not set
    if conf.get("provider_type") == "vertex_ai" and not conf.get("base_url"):
        project_id = conf.get("project_id")
        location = conf.get("location")
        if project_id and location:
            conf["base_url"] = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models"
    
    return conf


def _is_vertex_ai_model(model_name: str) -> bool:
    """
    Check if model requires Vertex AI provider.
    Updated to match your configuration approach.
    """
    if not CUSTOM_PROVIDERS_AVAILABLE:
        return False
    
    # Check if it's a Gemini model (should use Vertex AI)
    vertex_ai_models = [
        "gemini-2.0-flash-001",
        "gemini-2.5-pro-preview-06-05",
        "gemini",  # Generic gemini models
    ]
    
    model_lower = model_name.lower()
    is_vertex = any(model in model_lower for model in vertex_ai_models)
    
    if is_vertex:
        logger.info(f"Detected Vertex AI model: {model_name}")
    
    return is_vertex


def _create_vertex_ai_llm(llm_conf: Dict[str, Any]) -> CustomLLMWrapper:
    """Create Vertex AI LLM instance with validation."""
    try:
        # Validate required configuration for Vertex AI
        required_fields = ["model", "project_id", "location", "api_key"]
        missing_fields = [field for field in required_fields if not llm_conf.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required Vertex AI configuration fields: {missing_fields}")
        
        # Verify credentials file exists
        credentials_path = llm_conf.get("api_key")
        if credentials_path and not os.path.exists(credentials_path):
            raise ValueError(f"Vertex AI credentials file not found: {credentials_path}")
        
        # Create provider
        provider = create_custom_provider(llm_conf)
        wrapper = CustomLLMWrapper(provider=provider)
        
        logger.info(f"Successfully created Vertex AI provider: {llm_conf.get('model')}")
        return wrapper
        
    except Exception as e:
        logger.error(f"Failed to create Vertex AI provider: {e}")
        raise ValueError(f"Vertex AI provider creation failed: {e}")


def _create_standard_llm(llm_conf: Dict[str, Any]) -> ChatOpenAI:
    """Create standard ChatOpenAI instance for non-Vertex AI models."""
    try:
        # Remove Vertex AI specific configurations that ChatOpenAI doesn't understand
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
    Create LLM instance based on configuration with Vertex AI routing.
    Updated to match your conf.yaml structure.
    """
    # Get type-specific configuration from conf.yaml
    llm_type_map = {
        "reasoning": conf.get("REASONING_MODEL", {}),
        "basic": conf.get("BASIC_MODEL", {}),
        "vision": conf.get("VISION_MODEL", {}),
    }
    
    llm_conf = llm_type_map.get(llm_type)
    if not isinstance(llm_conf, dict):
        logger.warning(f"No conf.yaml configuration for '{llm_type}', falling back to environment variables")
        llm_conf = {}
    
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
    if _is_vertex_ai_model(model_name) or merged_conf.get("provider_type") == "vertex_ai":
        logger.info(f"Using Vertex AI for {llm_type}: {model_name}")
        return _create_vertex_ai_llm(merged_conf)
    else:
        logger.info(f"Using standard ChatOpenAI for {llm_type}: {model_name}")
        return _create_standard_llm(merged_conf)


def get_llm_by_type(llm_type: LLMType) -> Union[ChatOpenAI, CustomLLMWrapper]:
    """
    Get LLM instance by type with caching.
    Updated for your Vertex AI configuration.
    """
    # Return cached instance if available
    if llm_type in _llm_cache:
        logger.debug(f"Returning cached LLM for type: {llm_type}")
        return _llm_cache[llm_type]
    
    try:
        # Load configuration from conf.yaml
        conf_path = Path(__file__).parent.parent.parent / "conf.yaml"
        
        if conf_path.exists():
            conf = load_yaml_config(str(conf_path.resolve()))
            logger.info(f"Loaded configuration from conf.yaml")
        else:
            logger.warning(f"conf.yaml not found at {conf_path}, using environment variables only")
            conf = {}
        
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
                "type": "CustomLLMWrapper (Vertex AI)",
                "provider": llm_instance.provider.provider_name,
                "model": llm_instance.provider.model,
                "project_id": getattr(llm_instance.provider, 'project_id', 'unknown'),
                "location": getattr(llm_instance.provider, 'location', 'unknown')
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


def test_vertex_ai_configuration() -> Dict[str, Any]:
    """
    Test Vertex AI configuration using your exact setup.
    """
    results = {}
    
    # Test each LLM type with your configuration
    for llm_type in ["basic", "reasoning", "vision"]:
        try:
            logger.info(f"Testing {llm_type} LLM configuration...")
            
            llm = get_llm_by_type(llm_type)
            
            # Test with a simple prompt
            test_message = f"Hello! Please respond with 'OK - {llm_type} model working correctly.'"
            test_response = llm.invoke(test_message)
            
            results[llm_type] = {
                "status": "success",
                "response_length": len(test_response.content if hasattr(test_response, 'content') else str(test_response)),
                "llm_type": type(llm).__name__,
                "model": getattr(llm.provider, 'model', 'unknown') if hasattr(llm, 'provider') else 'unknown',
                "response_preview": str(test_response)[:100] + "..." if len(str(test_response)) > 100 else str(test_response)
            }
            
        except Exception as e:
            results[llm_type] = {
                "status": "error", 
                "error": str(e)
            }
    
    return results


def verify_environment_variables() -> Dict[str, Any]:
    """Verify that all required environment variables are set correctly."""
    required_vars = {
        "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "VERTEX_AI_LOCATION": os.getenv("VERTEX_AI_LOCATION"),
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "GEMINI_BASIC_MODEL": os.getenv("GEMINI_BASIC_MODEL"),
        "GEMINI_REASONING_MODEL": os.getenv("GEMINI_REASONING_MODEL"),
        "GEMINI_VISION_MODEL": os.getenv("GEMINI_VISION_MODEL"),
    }
    
    verification = {}
    for var_name, var_value in required_vars.items():
        if var_value:
            if var_name == "GOOGLE_APPLICATION_CREDENTIALS":
                # Check if file exists
                file_exists = os.path.exists(var_value) if var_value else False
                verification[var_name] = {
                    "status": "✅ SET" if file_exists else "❌ FILE NOT FOUND",
                    "value": var_value,
                    "file_exists": file_exists
                }
            else:
                verification[var_name] = {
                    "status": "✅ SET",
                    "value": var_value
                }
        else:
            verification[var_name] = {
                "status": "❌ NOT SET",
                "value": None
            }
    
    return verification


# Convenience functions for specific research tasks (updated for Vertex AI)
def get_research_llm(context_size: str = "large") -> Union[ChatOpenAI, CustomLLMWrapper]:
    """
    Get LLM optimized for research tasks based on required context size.
    Updated to use your Vertex AI models.
    """
    if context_size == "small":
        return get_llm_by_type("basic")      # gemini-2.0-flash-001 - fast, cost-effective
    elif context_size == "large":
        return get_llm_by_type("reasoning")  # gemini-2.5-pro-preview-06-05 - 1M context
    else:  # huge
        return get_llm_by_type("reasoning")  # Maximum context capability


def get_vision_llm() -> Union[ChatOpenAI, CustomLLMWrapper]:
    """Get LLM capable of processing images and vision tasks."""
    return get_llm_by_type("vision")  # gemini-2.5-pro-preview-06-05 with vision


from src.prompts.planner_model import Plan # Add this import at the top
from langchain_core.output_parsers import PydanticToolsParser # Add this import
from langchain_core.messages import HumanMessage # Add this import

def run_isolated_planner_test():
    """
    This is a self-contained test to run at startup.
    It completely bypasses the agent graph to test the core LLM functionality.
    """
    logger.info("="*80)
    logger.info("--- RUNNING AVERY'S ISOLATED PLANNER DIAGNOSTIC ---")
    
    try:
        # Step 1: Get the exact LLM type for the planner
        planner_llm_type = AGENT_LLM_MAP.get("planner", "basic")
        logger.info(f"TEST: Getting base LLM for type: '{planner_llm_type}'")
        base_llm = get_llm_by_type(planner_llm_type)
        
        # Step 2: Build the manual chain, as per the intelligence document
        logger.info("TEST: Binding the 'Plan' tool to the LLM.")
        llm_with_tools = base_llm.bind_tools([Plan], tool_choice="Plan")
        
        logger.info("TEST: Creating PydanticToolsParser for the 'Plan' tool.")
        parser = PydanticToolsParser(tools=[Plan], first_tool_only=True)
        
        logger.info("TEST: Constructing the chain: llm_with_tools | parser")
        chain = llm_with_tools | parser
        
        # Step 3: Invoke the chain with a simple, direct prompt
        test_prompt = "Create a research plan to find out the height of the Eiffel Tower."
        logger.info(f"TEST: Invoking chain with prompt: '{test_prompt}'")
        
        result = chain.invoke([HumanMessage(content=test_prompt)])
        
        # Step 4: Validate the result
        logger.info(f"TEST: Chain invocation complete. Result type: {type(result)}")
        
        if isinstance(result, Plan):
            logger.info("✅✅✅ SUCCESS: The isolated chain successfully produced a 'Plan' object.")
            logger.info(f"Resulting Plan Title: {result.title}")
            return True
        else:
            logger.error(f"❌❌❌ FAILURE: The isolated chain DID NOT produce a 'Plan' object.")
            logger.error(f"Result was: {result}")
            return False

    except Exception as e:
        logger.error(f"❌❌❌ CATASTROPHIC FAILURE: The isolated test threw an exception.")
        logger.error(f"Exception details: {e}", exc_info=True) # exc_info=True gives a full stack trace
        return False
    finally:
        logger.info("--- ISOLATED PLANNER DIAGNOSTIC COMPLETE ---")
        logger.info("="*80)


# Run the diagnostic test as soon as the module is loaded
run_isolated_planner_test()


# ============================================================================***
# *********** MULTI-LLM STRUCTURED OUTPUT COMPATIBILITY LAYER ************
# ============================================================================***

def get_structured_output_llm(agent_type: str, schema_class, fallback_method: str = None):
    """
    Get LLM with structured output capability detection.
    
    This function wraps get_llm_by_type() and automatically determines
    whether to use json_mode or native structured output based on the provider.
    
    Args:
        agent_type: The agent type from AGENT_LLM_MAP ("planner", "coordinator", etc.)
        schema_class: The Pydantic schema class (e.g., Plan)
        fallback_method: Override method if auto-detection fails
    
    Returns:
        LLM instance configured with appropriate structured output method
    """
    # Get the base LLM instance using existing system
    llm_type = AGENT_LLM_MAP.get(agent_type, agent_type)  # ← ADD THIS LINE
    base_llm = get_llm_by_type(llm_type)
    # base_llm = get_llm_by_type(agent_type)
    
    # Detect provider capabilities
    capabilities = _detect_provider_capabilities(base_llm)
    
    # Determine structured output method
    if fallback_method:
        # Use explicit override
        method = fallback_method
        logger.info(f"Using fallback method '{method}' for {agent_type}")
    elif capabilities["supports_json_mode"]:
        # Provider supports json_mode (OpenAI, Perplexity, Groq)
        method = "json_mode"
        logger.info(f"Using json_mode for {capabilities['provider']} ({agent_type})")
    else:
        # Provider uses native function calling (Gemini, Anthropic)
        method = None  # Default structured output
        logger.info(f"Using native structured output for {capabilities['provider']} ({agent_type})")
    
    # Create structured output LLM
    try:
        if method:
            return base_llm.with_structured_output(schema_class, method=method)
        else:
            return base_llm.with_structured_output(schema_class)
    except Exception as e:
        logger.error(f"Failed to create structured output LLM for {agent_type}: {e}")
        
        # Fallback: try without method parameter
        logger.warning(f"Attempting fallback for {agent_type}")
        try:
            return base_llm.with_structured_output(schema_class)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed for {agent_type}: {fallback_error}")
            raise ValueError(f"Could not create structured output LLM: {e}")


def _detect_provider_capabilities(llm_instance) -> dict:
    """
    Detect LLM provider capabilities based on instance type.
    
    This uses static mapping for reliability - no runtime testing needed.
    """
    llm_class_name = llm_instance.__class__.__name__
    
    # Static capability mapping
    capability_map = {
        # OpenAI-compatible providers (support json_mode)
        "ChatOpenAI": {
            "provider": "openai",
            "supports_json_mode": True,
            "supports_function_calling": True
        },
        "ChatPerplexity": {
            "provider": "perplexity", 
            "supports_json_mode": True,
            "supports_function_calling": True
        },
        "ChatGroq": {
            "provider": "groq",
            "supports_json_mode": True,
            "supports_function_calling": True
        },
        
        # Google providers (native function calling only)
        "ChatVertexAI": {
            "provider": "google_vertex",
            "supports_json_mode": False,
            "supports_function_calling": True
        },
        "ChatGoogleGenerativeAI": {
            "provider": "google_ai",
            "supports_json_mode": False,
            "supports_function_calling": True
        },
        
        # Anthropic (native function calling only)
        "ChatAnthropic": {
            "provider": "anthropic",
            "supports_json_mode": False,
            "supports_function_calling": True
        },
        
        # Custom wrapper for your Vertex AI system
        "CustomLLMWrapper": {
            "provider": "vertex_ai_custom",
            "supports_json_mode": False,
            "supports_function_calling": True
        }
    }
    
    # Look up capabilities
    capabilities = capability_map.get(llm_class_name)
    
    if capabilities:
        logger.debug(f"Detected provider: {capabilities['provider']} (class: {llm_class_name})")
        return capabilities
    else:
        # Unknown provider - use safe defaults
        logger.warning(f"Unknown LLM class: {llm_class_name}, using safe defaults")
        return {
            "provider": "unknown",
            "supports_json_mode": False,  # Safe default
            "supports_function_calling": True
        }


def get_provider_info(agent_type: str) -> dict:
    """
    Get provider information for debugging purposes.
    
    Returns:
        Dictionary with provider details for the specified agent type
    """
    try:
        llm = get_llm_by_type(agent_type)
        capabilities = _detect_provider_capabilities(llm)
        
        return {
            "agent_type": agent_type,
            "llm_class": llm.__class__.__name__,
            "provider": capabilities["provider"],
            "supports_json_mode": capabilities["supports_json_mode"],
            "supports_function_calling": capabilities["supports_function_calling"],
            "model_info": getattr(llm, 'model_name', 'unknown') if hasattr(llm, 'model_name') else 'unknown'
        }
    except Exception as e:
        return {
            "agent_type": agent_type,
            "error": str(e)
        }
