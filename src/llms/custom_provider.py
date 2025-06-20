# src/llms/custom_provider.py
# Copyright (c) 2025 - Vertex AI Compatible Provider Implementation for DeerFlow
# CORRECTED: Now uses Vertex AI API instead of Google AI Studio API

import os
import json
import logging
from typing import Dict, Any, List, Iterator, Union, Optional, Callable
from pydantic import Field  # ← ADD THIS LINE - This was missing!
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun

# Google Cloud AI Platform imports for Vertex AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    from google.auth import default
    from google.auth.exceptions import DefaultCredentialsError
    VERTEX_AI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Vertex AI dependencies not available: {e}")
    VERTEX_AI_AVAILABLE = False

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
        self.project_id = config.get("project_id") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = config.get("location") or os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.credentials_path = config.get("api_key") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.model = config.get("model", "").replace(self._get_provider_prefix(), "")
        self.provider_name = self.__class__.__name__.replace("Provider", "")
        
        # Enhanced configuration validation
        if not self.project_id:
            raise ProviderError(self.provider_name, "Google Cloud Project ID is required - check GOOGLE_CLOUD_PROJECT env var")
        if not self.credentials_path:
            raise ProviderError(self.provider_name, "Service account credentials required - check GOOGLE_APPLICATION_CREDENTIALS env var")
        if not self.model:
            raise ProviderError(self.provider_name, "Model name is required")
            
        logger.info(f"Initialized {self.provider_name} provider - Project: {self.project_id}, Model: {self.model}, Location: {self.location}")
    
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
    
    def invoke(self, messages) -> str:
        """Synchronous message completion - override in subclasses"""
        raise NotImplementedError(f"{self.provider_name} must implement invoke method")
    
    def stream(self, messages) -> Iterator[str]:
        """Streaming completion - fallback to invoke if not implemented"""
        response = self.invoke(messages)
        yield response


class VertexAIGeminiProvider(BaseLLMProvider):
    """Google Vertex AI Gemini Provider - Uses Vertex AI API (aiplatform.googleapis.com)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not VERTEX_AI_AVAILABLE:
            raise ProviderError("VertexAI", "Vertex AI dependencies not installed. Run: pip install google-cloud-aiplatform")
        
        # Initialize Vertex AI with project and location
        try:
            # Set credentials if path is provided
            if self.credentials_path and os.path.exists(self.credentials_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
            
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            logger.info(f"[VertexAI] Initialized successfully - Project: {self.project_id}, Location: {self.location}")
            
            # Test credentials
            try:
                credentials, project = default()
                logger.info(f"[VertexAI] Credentials verified for project: {project}")
            except DefaultCredentialsError as e:
                raise ProviderError("VertexAI", f"Credential error: {e}")
                
        except Exception as e:
            raise ProviderError("VertexAI", f"Initialization failed: {e}")
    
    def _get_provider_prefix(self) -> str:
        return "gemini/"
    
    def _convert_messages_to_vertex_format(self, messages: List[Dict]) -> str:
        """Convert messages to simple text prompt for Vertex AI"""
        # For now, we'll combine all messages into a single prompt
        # Vertex AI Gemini models can handle complex conversations but this is simpler for initial implementation
        combined_prompt = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                combined_prompt += f"System: {content}\n\n"
            elif role == "user":
                combined_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                combined_prompt += f"Assistant: {content}\n\n"
        
        # Remove trailing newlines
        return combined_prompt.strip()
    
    def invoke(self, messages) -> str:
        """Vertex AI Gemini-specific API call"""
        try:
            # Convert messages to format
            standard_messages = self._format_messages(messages)
            prompt = self._convert_messages_to_vertex_format(standard_messages)
            
            logger.info(f"[VertexAI] Making request with model: {self.model}")
            logger.debug(f"[VertexAI] Prompt: {prompt[:500]}...")
            
            # Initialize the model
            model = GenerativeModel(self.model)
            
            # Generate content
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.get("temperature", 0.7),
                    "max_output_tokens": self.config.get("max_tokens", 32768),
                    "top_p": self.config.get("top_p", 0.95),
                    "top_k": self.config.get("top_k", 40),
                }
            )
            
            # Extract response text
            if response.text:
                logger.info(f"[VertexAI] Successfully generated {len(response.text)} characters")
                return response.text
            else:
                # Check if there were any safety issues or other problems
                if response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        raise ProviderError("VertexAI", f"Generation stopped: {candidate.finish_reason}")
                
                raise ProviderError("VertexAI", "Empty response received")
                
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            else:
                error_msg = f"Vertex AI API error: {str(e)}"
                logger.error(f"[VertexAI] {error_msg}")
                raise ProviderError("VertexAI", error_msg)


# Legacy provider for non-Vertex AI models (DeepSeek, etc.)
class StandardAPIProvider(BaseLLMProvider):
    """Standard API Provider for OpenAI-compatible APIs (DeepSeek, Perplexity, etc.)"""
    
    def _get_provider_prefix(self) -> str:
        return ""
    
    def invoke(self, messages) -> str:
        """Standard OpenAI-compatible API call"""
        import requests
        
        # Use api_key for standard APIs (not service account path)
        api_key = self.config.get("api_key")
        if not api_key:
            raise ProviderError(self.provider_name, "API key is required for standard providers")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 8192)
        }
        
        base_url = self.config.get("base_url", "https://api.openai.com/v1")
        
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                raise ProviderError(self.provider_name, f"API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise ProviderError(self.provider_name, f"Network error: {e}")


def create_custom_provider(config: Dict[str, Any]) -> BaseLLMProvider:
    """Factory function to create appropriate provider with enhanced detection"""
    model = config.get("model", "").lower()
    provider_type = config.get("provider_type", "").lower()
    
    # Provider detection logic
    if "gemini" in model or provider_type == "vertex_ai":
        logger.info(f"Creating Vertex AI provider for model: {model}")
        return VertexAIGeminiProvider(config)
    else:
        logger.info(f"Creating standard API provider for model: {model}")
        return StandardAPIProvider(config)


class CustomLLMWrapper(BaseChatModel):
    """
    Full LangChain-compatible wrapper that supports all ChatModel methods
    Updated for Vertex AI compatibility
    """
    # Declare provider as a Pydantic field (REQUIRED for LangChain BaseChatModel)
    provider: BaseLLMProvider = Field(...)
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, provider: BaseLLMProvider, **kwargs):
        super().__init__(provider=provider, **kwargs)
        # self.provider = provider
        self._bound_tools = []
        self._bound_config = {}
        logger.info(f"Initialized CustomLLMWrapper with {provider.provider_name} provider")
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for the LLM type."""
        return f"custom_{self.provider.provider_name.lower()}"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Core generation method required by LangChain"""
        try:
            logger.info(f"[{self.provider.provider_name}] Processing request...")
            
            # Input validation
            if not messages:
                raise ValueError("Messages cannot be empty")
            
            result = self.provider.invoke(messages)
            
            if not result:
                raise ProviderError(self.provider.provider_name, "Empty response received")
            
            logger.info(f"[{self.provider.provider_name}] Successfully processed request")
            
            # Create LangChain ChatGeneration
            generation = ChatGeneration(message=AIMessage(content=result))
            return ChatResult(generations=[generation])
            
        except ProviderError as e:
            # Re-raise provider errors with context
            logger.error(f"Provider error: {e}")
            raise Exception(f"LLM Provider Error: {str(e)}")
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error in {self.provider.provider_name}: {str(e)}"
            logger.error(error_msg)
            raise Exception(f"LLM Provider Unexpected Error: {error_msg}")
    
    def invoke(self, input, config: Optional[RunnableConfig] = None, **kwargs) -> AIMessage:
        """LangChain invoke interface"""
        # Handle different input types
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, list):
            messages = input
        else:
            messages = [HumanMessage(content=str(input))]
        
        result = self._generate(messages, **kwargs)
        return result.generations[0].message
    
    def bind_tools(self, tools: List[Union[Dict[str, Any], BaseTool, Callable]], **kwargs) -> "CustomLLMWrapper":
        """Bind tools to the model"""
        logger.info(f"[{self.provider.provider_name}] Binding {len(tools)} tools")
        
        # Create a new instance with bound tools
        new_wrapper = CustomLLMWrapper(self.provider)
        new_wrapper._bound_tools = tools
        new_wrapper._bound_config = {**self._bound_config, **kwargs}
        
        return new_wrapper
    
    def with_config(self, config: RunnableConfig) -> "CustomLLMWrapper":
        """Create a new instance with updated configuration"""
        new_wrapper = CustomLLMWrapper(self.provider)
        new_wrapper._bound_tools = self._bound_tools
        new_wrapper._bound_config = {**self._bound_config, **config}
        return new_wrapper
    
    def stream(self, input, config: Optional[RunnableConfig] = None, **kwargs) -> Iterator[AIMessage]:
        """Streaming interface"""
        try:
            # Handle different input types
            if isinstance(input, str):
                messages = [HumanMessage(content=input)]
            elif isinstance(input, list):
                messages = input
            else:
                messages = [HumanMessage(content=str(input))]
            
            for chunk in self.provider.stream(messages):
                yield AIMessage(content=chunk)
                
        except Exception as e:
            logger.error(f"[{self.provider.provider_name}] Streaming error: {e}")
            raise Exception(f"LLM Provider Streaming Error: {str(e)}")


def test_vertex_ai_setup(project_id: str, location: str = "us-central1") -> Dict[str, Any]:
    """Test function to validate Vertex AI setup"""
    try:
        # Test configuration
        config = {
            "model": "gemini-2.0-flash-001",
            "project_id": project_id,
            "location": location,
            "provider_type": "vertex_ai",
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        provider = VertexAIGeminiProvider(config)
        wrapper = CustomLLMWrapper(provider)
        
        test_response = wrapper.invoke("Hello! Please respond with 'Vertex AI is working correctly.'")
        
        return {
            "status": "success",
            "provider": "Vertex AI Gemini",
            "project_id": project_id,
            "location": location,
            "model": config["model"],
            "response_length": len(test_response.content),
            "response_preview": test_response.content[:100] + "..." if len(test_response.content) > 100 else test_response.content
        }
        
    except Exception as e:
        return {
            "status": "error",
            "provider": "Vertex AI Gemini",
            "project_id": project_id,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test Vertex AI setup
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "deerflow-goa8z")
    
    result = test_vertex_ai_setup(project_id)
    print(f"Vertex AI test result: {json.dumps(result, indent=2)}")
