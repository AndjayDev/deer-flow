# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, List, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage, BaseMessage
from langgraph.types import Command

from src.config.report_style import ReportStyle
from src.config.tools import SELECTED_RAG_PROVIDER
from src.graph.builder import build_graph_with_memory
from src.podcast.graph.builder import build_graph as build_podcast_graph
from src.ppt.graph.builder import build_graph as build_ppt_graph
from src.prose.graph.builder import build_graph as build_prose_graph
from src.prompt_enhancer.graph.builder import build_graph as build_prompt_enhancer_graph
from src.rag.builder import build_retriever
from src.rag.retriever import Resource
from src.server.chat_request import (
    ChatMessage,
    ChatRequest,
    EnhancePromptRequest,
    GeneratePodcastRequest,
    GeneratePPTRequest,
    GenerateProseRequest,
    TTSRequest,
)
from src.server.mcp_request import MCPServerMetadataRequest, MCPServerMetadataResponse
from src.server.mcp_utils import load_mcp_tools
from src.server.rag_request import (
    RAGConfigResponse,
    RAGResourceRequest,
    RAGResourcesResponse,
)
from src.tools import VolcengineTTS

logger = logging.getLogger(__name__)

INTERNAL_SERVER_ERROR_DETAIL = "Internal Server Error"

# ============================================================================
# ENHANCED FASTAPI LIFESPAN WITH DEERFLOW DIAGNOSTICS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager - runs on startup and shutdown with DeerFlow diagnostics."""
    
    # === STARTUP EVENTS ===
    logger.info("🦌 DeerFlow FastAPI application starting up...")
    
    # Force LLM system initialization and diagnostics
    if os.getenv("DEERFLOW_RUN_DIAGNOSTICS", "false").lower() in ["true", "1", "yes"]:
        logger.info("🔬 DEERFLOW_RUN_DIAGNOSTICS=true detected, running startup diagnostics...")
        try:
            # Import and trigger diagnostics
            from src.llms.llm import force_run_diagnostics, get_llm_by_type
            
            # Force import and cache creation by getting a basic LLM
            logger.info("🧠 Initializing LLM system during startup...")
            basic_llm = get_llm_by_type("basic")
            logger.info(f"✅ LLM system initialized: {type(basic_llm).__name__}")
            
            # Run comprehensive diagnostics
            logger.info("🔬 Running comprehensive startup diagnostics...")
            force_run_diagnostics()
            logger.info("✅ Startup diagnostics completed!")
            
        except Exception as e:
            logger.error(f"❌ Startup diagnostic failed: {e}")
            import traceback
            logger.error(f"Full startup diagnostic traceback: {traceback.format_exc()}")
    else:
        logger.info("🔇 Startup diagnostics disabled (set DEERFLOW_RUN_DIAGNOSTICS=true to enable)")
    
    logger.info("🚀 DeerFlow application startup complete!")
    
    yield  # Application runs here
    
    # === SHUTDOWN EVENTS ===
    logger.info("🛑 DeerFlow application shutting down...")

# ============================================================================
# FASTAPI APP CREATION WITH LIFESPAN
# ============================================================================

app = FastAPI(
    title="DeerFlow API",
    description="Deep Research and Automation Framework with Enhanced Diagnostics",
    version="1.0.0",
    lifespan=lifespan  # This enables the startup/shutdown diagnostics
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

graph = build_graph_with_memory()

# ============================================================================
# DEERFLOW DIAGNOSTIC ENDPOINTS
# ============================================================================

@app.get("/api/test/enhanced")
async def test_enhanced_app():
    """Simple test to verify enhanced app.py is deployed."""
    return {
        "status": "success",
        "message": "Enhanced app.py is working!",
        "version": "enhanced_with_diagnostics",
        "timestamp": "2025-01-24"
    }

@app.get("/api/diagnostics/status")
async def diagnostics_status():
    """Simple status check for diagnostics system."""
    try:
        # Try to import the diagnostic module
        from src.llms import llm
        
        return {
            "status": "available",
            "message": "DeerFlow diagnostic system is available",
            "endpoints": [
                "/api/diagnostics/status - This status endpoint",
                "/api/diagnostics/run - Run comprehensive diagnostics", 
                "/api/diagnostics/llm-info - Get LLM configuration info",
                "/api/diagnostics/test-planner - Test planner structured output specifically"
            ],
            "environment": {
                "DEERFLOW_RUN_DIAGNOSTICS": os.getenv("DEERFLOW_RUN_DIAGNOSTICS", "false"),
                "DEERFLOW_AUTO_DIAGNOSE": os.getenv("DEERFLOW_AUTO_DIAGNOSE", "false")
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Diagnostic system not available: {e}"
        }

@app.get("/api/diagnostics/run")
async def manual_diagnostics():
    """Manual endpoint to trigger comprehensive LLM diagnostics."""
    try:
        logger.info("🔬 Manual diagnostics triggered via API")
        
        # Import the diagnostic functions
        from src.llms.llm import force_run_diagnostics, get_llm_by_type, get_cached_llm_info
        
        # Force LLM initialization first
        logger.info("🧠 Initializing LLM system...")
        try:
            basic_llm = get_llm_by_type("basic")
            logger.info(f"✅ Basic LLM initialized: {type(basic_llm).__name__}")
        except Exception as llm_error:
            logger.error(f"❌ LLM initialization failed: {llm_error}")
            raise HTTPException(status_code=500, detail=f"LLM initialization failed: {llm_error}")
        
        # Run comprehensive diagnostics
        logger.info("🔬 Running comprehensive diagnostics...")
        force_run_diagnostics()
        
        # Get summary info
        cache_info = get_cached_llm_info()
        
        return {
            "status": "success", 
            "message": "Diagnostics completed successfully - check application logs for detailed output",
            "cache_info": cache_info,
            "note": "Detailed diagnostic output is available in the application logs"
        }
        
    except Exception as e:
        error_msg = f"Manual diagnostics failed: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/diagnostics/llm-info")
async def get_llm_info():
    """Get current LLM configuration and provider information."""
    try:
        from src.llms.llm import get_cached_llm_info, get_provider_info, verify_environment_variables
        
        # Get cache info
        cache_info = get_cached_llm_info()
        
        # Get provider info for each agent type
        provider_info = {}
        for agent_type in ["basic", "reasoning", "vision"]:
            try:
                provider_info[agent_type] = get_provider_info(agent_type)
            except Exception as e:
                provider_info[agent_type] = {"error": str(e)}
        
        # Get environment variable status
        env_status = verify_environment_variables()
        
        return {
            "status": "success",
            "cache_info": cache_info,
            "provider_info": provider_info,
            "environment_variables": env_status,
            "diagnostics_enabled": os.getenv("DEERFLOW_RUN_DIAGNOSTICS", "false"),
            "auto_diagnostics_enabled": os.getenv("DEERFLOW_AUTO_DIAGNOSE", "false")
        }
        
    except Exception as e:
        error_msg = f"Failed to get LLM info: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/diagnostics/test-planner")
async def test_planner_specifically():
    """Test the specific planner_node structured output issue that's causing NoneType errors."""
    try:
        from src.llms.llm import get_llm_by_type
        from src.config.agents import AGENT_LLM_MAP
        
        # Check if Plan model is available
        try:
            from src.prompts.planner_model import Plan
            plan_available = True
        except ImportError as e:
            plan_available = False
            return {
                "status": "error",
                "message": f"Plan model not available: {e}",
                "recommendation": "Fix Plan model import before testing structured output"
            }
        
        if not plan_available:
            raise HTTPException(status_code=500, detail="Plan model not available for testing")
        
        # Get planner LLM
        planner_llm_type = AGENT_LLM_MAP.get("planner", "basic")
        logger.info(f"🎯 Testing planner LLM type: {planner_llm_type}")
        
        base_llm = get_llm_by_type(planner_llm_type)
        
        test_results = {}
        
        # Test Method 1: Direct .with_structured_output() (current failing method)
        logger.info("🔬 Testing Method 1: Direct .with_structured_output()")
        try:
            structured_llm = base_llm.with_structured_output(Plan)
            result = structured_llm.invoke("Create a research plan to find the height of the Eiffel Tower.")
            
            if isinstance(result, Plan):
                test_results["direct_structured"] = {
                    "status": "SUCCESS",
                    "result_type": str(type(result)),
                    "title": getattr(result, 'title', 'No title')
                }
            else:
                test_results["direct_structured"] = {
                    "status": "FAILED", 
                    "result_type": str(type(result)),
                    "issue": "Returned wrong type instead of Plan object (this is your current issue)"
                }
                
        except Exception as e:
            test_results["direct_structured"] = {
                "status": "EXCEPTION",
                "error": str(e)
            }
        
        # Test Method 2: Manual bind_tools + parser (Avery's suggested fix)
        logger.info("🔬 Testing Method 2: Manual bind_tools + PydanticToolsParser")
        try:
            from langchain_core.output_parsers import PydanticToolsParser
            from langchain_core.messages import HumanMessage
            
            llm_with_tools = base_llm.bind_tools([Plan], tool_choice="Plan")
            parser = PydanticToolsParser(tools=[Plan], first_tool_only=True)
            chain = llm_with_tools | parser
            
            result = chain.invoke([HumanMessage(content="Create a research plan to find the height of the Eiffel Tower.")])
            
            if isinstance(result, Plan):
                test_results["manual_tools"] = {
                    "status": "SUCCESS",
                    "result_type": str(type(result)),
                    "title": getattr(result, 'title', 'No title')
                }
            else:
                test_results["manual_tools"] = {
                    "status": "FAILED",
                    "result_type": str(type(result)),
                    "issue": "Returned wrong type instead of Plan object"
                }
                
        except Exception as e:
            test_results["manual_tools"] = {
                "status": "EXCEPTION", 
                "error": str(e)
            }
        
        # Determine recommendation
        working_methods = [method for method, result in test_results.items() if result.get("status") == "SUCCESS"]
        
        if working_methods:
            recommendation = f"✅ SUCCESS: Use {working_methods[0]} method for planner_node"
            overall_status = "success"
            fix_instructions = f"Update your planner_node function to use the {working_methods[0]} approach"
        else:
            recommendation = "❌ CRITICAL: No working methods found - planner_node will continue to fail"
            overall_status = "critical_issue"
            fix_instructions = "Check Vertex AI configuration and Plan model availability"
        
        return {
            "status": overall_status,
            "planner_llm_type": planner_llm_type,
            "llm_class": type(base_llm).__name__,
            "test_results": test_results,
            "working_methods": working_methods,
            "recommendation": recommendation,
            "fix_instructions": fix_instructions,
            "message": "Check application logs for detailed diagnostic output"
        }
        
    except Exception as e:
        error_msg = f"Planner test failed: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

# ============================================================================
# EXISTING DEERFLOW ENDPOINTS (UNCHANGED)
# ============================================================================

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id
    if thread_id == "__default__":
        thread_id = str(uuid4())
    return StreamingResponse(
        _astream_workflow_generator(
            request.model_dump()["messages"],
            thread_id,
            request.resources,
            request.max_plan_iterations,
            request.max_step_num,
            request.max_search_results,
            request.auto_accepted_plan,
            request.interrupt_feedback,
            request.mcp_settings,
            request.enable_background_investigation,
            request.report_style,
        ),
        media_type="text/event-stream",
    )


async def _astream_workflow_generator(
    messages: List[ChatMessage],
    thread_id: str,
    resources: List[Resource],
    max_plan_iterations: int,
    max_step_num: int,
    max_search_results: int,
    auto_accepted_plan: bool,
    interrupt_feedback: str,
    mcp_settings: dict,
    enable_background_investigation: bool,
    report_style: ReportStyle,
):
    input_ = {
        "messages": messages,
        "plan_iterations": 0,
        "final_report": "",
        "current_plan": None,
        "observations": [],
        "auto_accepted_plan": auto_accepted_plan,
        "enable_background_investigation": enable_background_investigation,
    }
    if not auto_accepted_plan and interrupt_feedback:
        resume_msg = f"[{interrupt_feedback}]"
        # add the last message to the resume message
        if messages:
            resume_msg += f" {messages[-1]['content']}"
        input_ = Command(resume=resume_msg)
    async for agent, _, event_data in graph.astream(
        input_,
        config={
            "thread_id": thread_id,
            "resources": resources,
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "max_search_results": max_search_results,
            "mcp_settings": mcp_settings,
            "report_style": report_style.value,
        },
        stream_mode=["messages", "updates"],
        subgraphs=True,
    ):
        if isinstance(event_data, dict):
            if "__interrupt__" in event_data:
                yield _make_event(
                    "interrupt",
                    {
                        "thread_id": thread_id,
                        "id": event_data["__interrupt__"][0].ns[0],
                        "role": "assistant",
                        "content": event_data["__interrupt__"][0].value,
                        "finish_reason": "interrupt",
                        "options": [
                            {"text": "Edit plan", "value": "edit_plan"},
                            {"text": "Start research", "value": "accepted"},
                        ],
                    },
                )
            continue
        message_chunk, message_metadata = cast(
            tuple[BaseMessage, dict[str, any]], event_data
        )
        event_stream_message: dict[str, any] = {
            "thread_id": thread_id,
            "agent": agent[0].split(":")[0],
            "id": message_chunk.id,
            "role": "assistant",
            "content": message_chunk.content,
        }
        if message_chunk.response_metadata.get("finish_reason"):
            event_stream_message["finish_reason"] = message_chunk.response_metadata.get(
                "finish_reason"
            )
        if isinstance(message_chunk, ToolMessage):
            # Tool Message - Return the result of the tool call
            event_stream_message["tool_call_id"] = message_chunk.tool_call_id
            yield _make_event("tool_call_result", event_stream_message)
        elif isinstance(message_chunk, AIMessageChunk):
            # AI Message - Raw message tokens
            if message_chunk.tool_calls:
                # AI Message - Tool Call
                event_stream_message["tool_calls"] = message_chunk.tool_calls
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_calls", event_stream_message)
            elif message_chunk.tool_call_chunks:
                # AI Message - Tool Call Chunks
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_call_chunks", event_stream_message)
            else:
                # AI Message - Raw message tokens
                yield _make_event("message_chunk", event_stream_message)


def _make_event(event_type: str, data: dict[str, any]):
    if data.get("content") == "":
        data.pop("content")
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using volcengine TTS API."""
    try:
        app_id = os.getenv("VOLCENGINE_TTS_APPID", "")
        if not app_id:
            raise HTTPException(
                status_code=400, detail="VOLCENGINE_TTS_APPID is not set"
            )
        access_token = os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN", "")
        if not access_token:
            raise HTTPException(
                status_code=400, detail="VOLCENGINE_TTS_ACCESS_TOKEN is not set"
            )
        cluster = os.getenv("VOLCENGINE_TTS_CLUSTER", "volcano_tts")
        voice_type = os.getenv("VOLCENGINE_TTS_VOICE_TYPE", "BV700_V2_streaming")

        tts_client = VolcengineTTS(
            appid=app_id,
            access_token=access_token,
            cluster=cluster,
            voice_type=voice_type,
        )
        # Call the TTS API
        result = tts_client.text_to_speech(
            text=request.text[:1024],
            encoding=request.encoding,
            speed_ratio=request.speed_ratio,
            volume_ratio=request.volume_ratio,
            pitch_ratio=request.pitch_ratio,
            text_type=request.text_type,
            with_frontend=request.with_frontend,
            frontend_type=request.frontend_type,
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=str(result["error"]))

        # Decode the base64 audio data
        audio_data = base64.b64decode(result["audio_data"])

        # Return the audio file
        return Response(
            content=audio_data,
            media_type=f"audio/{request.encoding}",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=tts_output.{request.encoding}"
                )
            },
        )
    except Exception as e:
        logger.exception(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/podcast/generate")
async def generate_podcast(request: GeneratePodcastRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_podcast_graph()
        final_state = workflow.invoke({"input": report_content})
        audio_bytes = final_state["output"]
        return Response(content=audio_bytes, media_type="audio/mp3")
    except Exception as e:
        logger.exception(f"Error occurred during podcast generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/ppt/generate")
async def generate_ppt(request: GeneratePPTRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_ppt_graph()
        final_state = workflow.invoke({"input": report_content})
        generated_file_path = final_state["generated_file_path"]
        with open(generated_file_path, "rb") as f:
            ppt_bytes = f.read()
        return Response(
            content=ppt_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    except Exception as e:
        logger.exception(f"Error occurred during ppt generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/prose/generate")
async def generate_prose(request: GenerateProseRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Generating prose for prompt: {sanitized_prompt}")
        workflow = build_prose_graph()
        events = workflow.astream(
            {
                "content": request.prompt,
                "option": request.option,
                "command": request.command,
            },
            stream_mode="messages",
            subgraphs=True,
        )
        return StreamingResponse(
            (f"data: {event[0].content}\n\n" async for _, event in events),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(f"Error occurred during prose generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/prompt/enhance")
async def enhance_prompt(request: EnhancePromptRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Enhancing prompt: {sanitized_prompt}")

        # Convert string report_style to ReportStyle enum
        report_style = None
        if request.report_style:
            try:
                # Handle both uppercase and lowercase input
                style_mapping = {
                    "ACADEMIC": ReportStyle.ACADEMIC,
                    "POPULAR_SCIENCE": ReportStyle.POPULAR_SCIENCE,
                    "NEWS": ReportStyle.NEWS,
                    "SOCIAL_MEDIA": ReportStyle.SOCIAL_MEDIA,
                    "academic": ReportStyle.ACADEMIC,
                    "popular_science": ReportStyle.POPULAR_SCIENCE,
                    "news": ReportStyle.NEWS,
                    "social_media": ReportStyle.SOCIAL_MEDIA,
                }
                report_style = style_mapping.get(
                    request.report_style, ReportStyle.ACADEMIC
                )
            except Exception:
                # If invalid style, default to ACADEMIC
                report_style = ReportStyle.ACADEMIC
        else:
            report_style = ReportStyle.ACADEMIC

        workflow = build_prompt_enhancer_graph()
        final_state = workflow.invoke(
            {
                "prompt": request.prompt,
                "context": request.context,
                "report_style": report_style,
            }
        )
        return {"result": final_state["output"]}
    except Exception as e:
        logger.exception(f"Error occurred during prompt enhancement: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/mcp/server/metadata", response_model=MCPServerMetadataResponse)
async def mcp_server_metadata(request: MCPServerMetadataRequest):
    """Get information about an MCP server."""
    try:
        # Set default timeout with a longer value for this endpoint
        timeout = 300  # Default to 300 seconds for this endpoint

        # Use custom timeout from request if provided
        if request.timeout_seconds is not None:
            timeout = request.timeout_seconds

        # Load tools from the MCP server using the utility function
        tools = await load_mcp_tools(
            server_type=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            timeout_seconds=timeout,
        )

        # Create the response with tools
        response = MCPServerMetadataResponse(
            transport=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            tools=tools,
        )

        return response
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.exception(f"Error in MCP server metadata endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)
        raise


@app.get("/api/rag/config", response_model=RAGConfigResponse)
async def rag_config():
    """Get the config of the RAG."""
    return RAGConfigResponse(provider=SELECTED_RAG_PROVIDER)


@app.get("/api/rag/resources", response_model=RAGResourcesResponse)
async def rag_resources(request: Annotated[RAGResourceRequest, Query()]):
    """Get the resources of the RAG."""
    retriever = build_retriever()
    if retriever:
        return RAGResourcesResponse(resources=retriever.list_resources(request.query))
    return RAGResourcesResponse(resources=[])
