# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# UPDATED: Now compatible with Vertex AI and your exact .env configuration

from pydantic import Field
from pathlib import Path
from typing import Any, Dict, Union
import os
import logging
import sys
import traceback
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticToolsParser

from langchain_openai import ChatOpenAI
from src.config import load_yaml_config
from src.config.agents import LLMType
from src.config.agents import AGENT_LLM_MAP

# Add these imports for the diagnostics
try:
    from src.prompts.planner_model import Plan
    PLAN_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Plan model not available for testing: {e}")
    PLAN_MODEL_AVAILABLE = False

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

# Cache for LLM instances - FIXED THE TYPO
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


# ============================================================================
# ENHANCED DIAGNOSTIC FUNCTIONS FOR DEERFLOW PLANNER_NODE DEBUGGING
# ============================================================================

# Global flag to prevent multiple diagnostic runs
_DIAGNOSTICS_RUN = False

def print_diagnostic_header(title: str):
    """Print a clearly visible diagnostic header."""
    border = "=" * 80
    print(f"\n{border}")
    print(f"🦌 DEERFLOW DIAGNOSTIC: {title}")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{border}")

def print_diagnostic_footer():
    """Print diagnostic footer."""
    print("=" * 80 + "\n")

def test_basic_llm_connectivity():
    """Test basic LLM connectivity for all types."""
    print_diagnostic_header("BASIC LLM CONNECTIVITY TEST")
    
    results = {}
    llm_types = ["basic", "reasoning", "vision"]
    
    for llm_type in llm_types:
        print(f"\n🧠 Testing {llm_type.upper()} LLM...")
        try:
            llm = get_llm_by_type(llm_type)
            print(f"   ✅ LLM created: {type(llm).__name__}")
            
            # Test simple invocation
            test_message = "Say 'OK' to confirm connectivity."
            response = llm.invoke(test_message)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            print(f"   ✅ Response received: {response_content[:50]}...")
            results[llm_type] = "SUCCESS"
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)[:100]}...")
            results[llm_type] = f"FAILED: {e}"
    
    print(f"\n📊 CONNECTIVITY SUMMARY:")
    for llm_type, result in results.items():
        status = "✅" if result == "SUCCESS" else "❌"
        print(f"   {status} {llm_type.upper()}: {result}")
    
    print_diagnostic_footer()
    return results

def test_structured_output_compatibility():
    """Test structured output for the exact planner_node scenario."""
    print_diagnostic_header("STRUCTURED OUTPUT COMPATIBILITY TEST")
    
    if not PLAN_MODEL_AVAILABLE:
        print("❌ Cannot run structured output test - Plan model not available")
        print_diagnostic_footer()
        return None
    
    results = {}
    
    # Test the exact planner scenario
    try:
        print("🎯 Testing PLANNER agent structured output (the failing scenario)...")
        
        # Get the exact LLM type used by planner
        planner_llm_type = AGENT_LLM_MAP.get("planner", "basic")
        print(f"   📍 Planner uses LLM type: {planner_llm_type}")
        
        base_llm = get_llm_by_type(planner_llm_type)
        print(f"   ✅ Base LLM created: {type(base_llm).__name__}")
        
        # Test Method 1: Direct .with_structured_output() (current failing method)
        print("\n   🔬 Method 1: Direct .with_structured_output()")
        try:
            structured_llm = base_llm.with_structured_output(Plan)
            test_prompt = "Create a research plan to find the height of the Eiffel Tower."
            result = structured_llm.invoke(test_prompt)
            
            if isinstance(result, Plan):
                print(f"   ✅ SUCCESS: Returned Plan object with title: {result.title}")
                results["direct_structured"] = "SUCCESS"
            else:
                print(f"   ❌ FAILED: Returned {type(result)} instead of Plan")
                print(f"   📝 Result content: {str(result)[:200]}...")
                results["direct_structured"] = f"WRONG_TYPE: {type(result)}"
                
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)[:150]}...")
            results["direct_structured"] = f"EXCEPTION: {e}"
        
        # Test Method 2: Manual bind_tools + parser (Avery's suggested fix)
        print("\n   🔬 Method 2: Manual bind_tools + PydanticToolsParser")
        try:
            llm_with_tools = base_llm.bind_tools([Plan], tool_choice="Plan")
            parser = PydanticToolsParser(tools=[Plan], first_tool_only=True)
            chain = llm_with_tools | parser
            
            test_prompt = [HumanMessage(content="Create a research plan to find the height of the Eiffel Tower.")]
            result = chain.invoke(test_prompt)
            
            if isinstance(result, Plan):
                print(f"   ✅ SUCCESS: Returned Plan object with title: {result.title}")
                results["manual_tools"] = "SUCCESS"
            else:
                print(f"   ❌ FAILED: Returned {type(result)} instead of Plan")
                print(f"   📝 Result content: {str(result)[:200]}...")
                results["manual_tools"] = f"WRONG_TYPE: {type(result)}"
                
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)[:150]}...")
            results["manual_tools"] = f"EXCEPTION: {e}"
        
        # Test Method 3: Provider capability detection
        print("\n   🔬 Method 3: Provider Capability Detection")
        try:
            capabilities = _detect_provider_capabilities(base_llm)
            print(f"   📊 Provider: {capabilities['provider']}")
            print(f"   📊 Supports JSON mode: {capabilities['supports_json_mode']}")
            print(f"   📊 Supports function calling: {capabilities['supports_function_calling']}")
            
            if capabilities['supports_json_mode']:
                structured_llm = base_llm.with_structured_output(Plan, method="json_mode")
            else:
                structured_llm = base_llm.with_structured_output(Plan)
            
            result = structured_llm.invoke("Create a research plan to find the height of the Eiffel Tower.")
            
            if isinstance(result, Plan):
                print(f"   ✅ SUCCESS: Returned Plan object with title: {result.title}")
                results["capability_detection"] = "SUCCESS"
            else:
                print(f"   ❌ FAILED: Returned {type(result)} instead of Plan")
                results["capability_detection"] = f"WRONG_TYPE: {type(result)}"
                
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)[:150]}...")
            results["capability_detection"] = f"EXCEPTION: {e}"
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR in structured output test: {e}")
        results["critical_error"] = str(e)
    
    print(f"\n📊 STRUCTURED OUTPUT SUMMARY:")
    for method, result in results.items():
        status = "✅" if result == "SUCCESS" else "❌"
        print(f"   {status} {method}: {result}")
    
    print_diagnostic_footer()
    return results

def test_vertex_ai_specific_configuration():
    """Test Vertex AI specific configuration and authentication."""
    print_diagnostic_header("VERTEX AI CONFIGURATION TEST")
    
    # Test environment variables
    print("🔑 Environment Variables Check:")
    required_vars = {
        "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "VERTEX_AI_LOCATION": os.getenv("VERTEX_AI_LOCATION"), 
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "GEMINI_BASIC_MODEL": os.getenv("GEMINI_BASIC_MODEL"),
        "GEMINI_REASONING_MODEL": os.getenv("GEMINI_REASONING_MODEL"),
    }
    
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if var_value:
            if var_name == "GOOGLE_APPLICATION_CREDENTIALS":
                file_exists = os.path.exists(var_value)
                status = "✅" if file_exists else "❌ FILE NOT FOUND"
                print(f"   {status} {var_name}: {var_value}")
                if not file_exists:
                    missing_vars.append(f"{var_name} (file not found)")
            else:
                print(f"   ✅ {var_name}: {var_value}")
        else:
            print(f"   ❌ {var_name}: NOT SET")
            missing_vars.append(var_name)
    
    if missing_vars:
        print(f"\n⚠️  Missing configuration: {', '.join(missing_vars)}")
    
    # Test Vertex AI provider creation
    print(f"\n🏭 Vertex AI Provider Creation:")
    try:
        if CUSTOM_PROVIDERS_AVAILABLE:
            basic_conf = _get_env_llm_conf("basic")
            print(f"   📋 Basic LLM config: {basic_conf}")
            
            vertex_llm = _create_vertex_ai_llm(basic_conf)
            print(f"   ✅ Vertex AI LLM created: {type(vertex_llm).__name__}")
            
            # Test simple call
            response = vertex_llm.invoke("Say 'Vertex AI working' to confirm.")
            print(f"   ✅ Vertex AI response: {response.content[:50]}...")
            
        else:
            print("   ❌ Custom providers not available")
            
    except Exception as e:
        print(f"   ❌ Vertex AI creation failed: {e}")
        print(f"   📝 Full error: {traceback.format_exc()}")
    
    print_diagnostic_footer()

def run_comprehensive_diagnostics():
    """Run all diagnostic tests in sequence."""
    global _DIAGNOSTICS_RUN
    
    if _DIAGNOSTICS_RUN:
        print("🔄 Diagnostics already run, skipping...")
        return
    
    print("\n" + "🦌" * 25 + " DEERFLOW COMPREHENSIVE DIAGNOSTICS " + "🦌" * 25)
    print(f"🚀 Starting comprehensive diagnostic suite...")
    print(f"📍 Python version: {sys.version}")
    print(f"📍 Working directory: {os.getcwd()}")
    
    try:
        # Test 1: Basic connectivity
        connectivity_results = test_basic_llm_connectivity()
        
        # Test 2: Structured output (the main issue)
        structured_results = test_structured_output_compatibility()
        
        # Test 3: Vertex AI specific
        test_vertex_ai_specific_configuration()
        
        # Summary
        print_diagnostic_header("FINAL DIAGNOSTIC SUMMARY")
        
        if connectivity_results:
            working_llms = [k for k, v in connectivity_results.items() if v == "SUCCESS"]
            print(f"🧠 Working LLM types: {working_llms}")
        
        if structured_results:
            working_methods = [k for k, v in structured_results.items() if v == "SUCCESS"]
            if working_methods:
                print(f"✅ Working structured output methods: {working_methods}")
                print(f"🎯 RECOMMENDATION: Use {working_methods[0]} for planner_node")
            else:
                print("❌ NO working structured output methods found!")
                print("🔧 URGENT: Planner will continue to fail until this is fixed")
        
        print_diagnostic_footer()
        
    except Exception as e:
        print(f"❌ DIAGNOSTIC SUITE CRASHED: {e}")
        traceback.print_exc()
    
    _DIAGNOSTICS_RUN = True

def force_run_diagnostics():
    """Force run diagnostics even if already run (for manual testing)."""
    global _DIAGNOSTICS_RUN
    _DIAGNOSTICS_RUN = False
    run_comprehensive_diagnostics()

# ============================================================================
# MULTI-LLM STRUCTURED OUTPUT COMPATIBILITY LAYER  
# ============================================================================

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
    base_llm = get_llm_by_type(agent_type)
    
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

# Automatic diagnostic trigger
def trigger_startup_diagnostics():
    """Trigger diagnostics on module import if environment variable is set."""
    if os.getenv("DEERFLOW_RUN_DIAGNOSTICS", "false").lower() in ["true", "1", "yes"]:
        print("🔧 DEERFLOW_RUN_DIAGNOSTICS=true detected, running diagnostics...")
        run_comprehensive_diagnostics()
    elif os.getenv("DEERFLOW_AUTO_DIAGNOSE", "false").lower() in ["true", "1", "yes"]:
        print("🔧 DEERFLOW_AUTO_DIAGNOSE=true detected, running diagnostics...")
        run_comprehensive_diagnostics()

# Trigger diagnostics automatically when module is imported during startup
trigger_startup_diagnostics()

if __name__ == "__main__":
    # Test the Vertex AI system with your configuration
    try:
        logger.info("Testing DeerFlow Vertex AI system...")
        
        # Verify environment variables
        env_verification = verify_environment_variables()
        print("🔍 Environment Variable Verification:")
        for var_name, var_info in env_verification.items():
            print(f"  {var_name}: {var_info['status']}")
            if var_info.get('value'):
                print(f"    Value: {var_info['value']}")
        
        print("\n" + "="*50)
        
        # Run enhanced diagnostics
        print("🦌 Running enhanced DeerFlow diagnostics...")
        force_run_diagnostics()
        
        print("\n" + "="*50)
        
        # Show cache info
        cache_info = get_cached_llm_info()
        print("📦 Cache Information:")
        for llm_type, info in cache_info.items():
            print(f"  {llm_type}: {info}")
        
    except Exception as e:
        print(f"❌ Vertex AI system test failed: {e}")
        logger.error(f"Vertex AI system test failed: {e}")
