# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import base64
import json
import logging
import os
from datetime import datetime
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

# 🔧 DIAGNOSTIC IMPORTS - Added for system monitoring and testing
import traceback
import sys
import subprocess
from datetime import datetime
from src.llms.llm import (
    get_llm_by_type, 
    verify_environment_variables, 
    test_vertex_ai_configuration,
    get_cached_llm_info,
    get_provider_info
)
from src.prompts.planner_model import Plan
from langchain_core.output_parsers import PydanticToolsParser

logger = logging.getLogger(__name__)

INTERNAL_SERVER_ERROR_DETAIL = "Internal Server Error"

app = FastAPI(
    title="DeerFlow API",
    description="API for Deer",
    version="0.1.0",
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

# 🔧 DIAGNOSTIC ENDPOINTS - Added for comprehensive system testing

@app.get("/api/diagnostics/health")
async def diagnostic_health():
    """Basic health check with timestamp and container info"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deerflow-backend",
        "version": "0.1.0",
        "container_id": os.getenv("HOSTNAME", "unknown"),
        "working_directory": os.getcwd()
    }

@app.get("/api/diagnostics/environment")
async def diagnostic_environment():
    """Check environment configuration and deployment state"""
    try:
        logger.info("🧪 Running environment diagnostics...")
        
        # Environment variables check
        env_vars = verify_environment_variables()
        
        # Git deployment information
        git_info = {}
        try:
            git_info["commit_hash"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd="/app"
            ).decode().strip()[:8]
            git_info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd="/app"
            ).decode().strip()
            git_info["status"] = "available"
        except Exception as git_error:
            git_info = {"status": "unavailable", "error": str(git_error)}
        
        # System information
        system_info = {
            "python_version": sys.version,
            "hostname": os.getenv("HOSTNAME", "unknown"),
            "working_directory": os.getcwd(),
            "environment": os.getenv("NODE_ENV", "unknown")
        }
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "environment_variables": env_vars,
            "git_deployment": git_info,
            "system_info": system_info
        }
        
    except Exception as e:
        logger.error(f"❌ Environment diagnostic failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/api/diagnostics/llm-test")
async def diagnostic_llm_test():
    """Test all LLM configurations with Vertex AI"""
    try:
        logger.info("🧪 Starting comprehensive LLM diagnostic...")
        
        # Environment check first
        env_status = verify_environment_variables()
        
        # Test Vertex AI configuration
        vertex_results = test_vertex_ai_configuration()
        
        # Cache information
        cache_info = get_cached_llm_info()
        
        # Provider information for each agent type
        agent_providers = {}
        from src.config.agents import AGENT_LLM_MAP
        for agent_name, llm_type in AGENT_LLM_MAP.items():
            try:
                provider_info = get_provider_info(llm_type)
                agent_providers[agent_name] = provider_info
            except Exception as e:
                agent_providers[agent_name] = {"error": str(e)}
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "environment_check": env_status,
            "vertex_ai_tests": vertex_results,
            "cache_information": cache_info,
            "agent_providers": agent_providers
        }
        
    except Exception as e:
        logger.error(f"❌ LLM diagnostic test failed: {e}")
        return {
            "status": "error", 
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/api/diagnostics/planner-test")
async def diagnostic_planner_test():
    """Test planner structured output with exact nodes.py approach"""
    try:
        logger.info("🧪 Testing planner structured output...")
        
        from src.config.agents import AGENT_LLM_MAP
        
        test_messages = [
            {"role": "user", "content": "Create a simple research plan about renewable energy with 2 steps."}
        ]
        
        results = {}
        
        # Test the exact approach from nodes.py
        try:
            # Step 1: Get the base LLM for the planner
            base_llm = get_llm_by_type(AGENT_LLM_MAP["planner"])
            
            # Step 2: Bind the Plan schema as a tool (exact approach from nodes.py)
            llm_with_tools = base_llm.bind_tools([Plan], tool_choice=Plan)
            
            # Step 3: Define the parser
            parser = PydanticToolsParser(tools=[Plan], first_tool_only=True)
            
            # Step 4: Create the chain
            chain = llm_with_tools | parser
            
            # Step 5: Invoke the chain
            response_plan_object = chain.invoke(test_messages)
            
            results["current_nodes_approach"] = {
                "status": "success",
                "llm_class": base_llm.__class__.__name__,
                "response_type": str(type(response_plan_object)),
                "is_plan_object": isinstance(response_plan_object, Plan),
                "is_none": response_plan_object is None,
                "response_preview": str(response_plan_object)[:300] if response_plan_object else "None",
                "plan_title": getattr(response_plan_object, 'title', 'N/A') if isinstance(response_plan_object, Plan) else 'N/A',
                "plan_steps_count": len(getattr(response_plan_object, 'steps', [])) if isinstance(response_plan_object, Plan) else 0
            }
            
        except Exception as e:
            results["current_nodes_approach"] = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()[:500]
            }
        
        # Test alternative approaches for comparison
        try:
            base_llm = get_llm_by_type(AGENT_LLM_MAP["planner"])
            
            # Alternative 1: with_structured_output auto
            try:
                structured_llm_auto = base_llm.with_structured_output(Plan)
                response_auto = structured_llm_auto.invoke(test_messages)
                results["with_structured_output_auto"] = {
                    "status": "success",
                    "response_type": str(type(response_auto)),
                    "is_plan_object": isinstance(response_auto, Plan),
                    "is_none": response_auto is None
                }
            except Exception as e:
                results["with_structured_output_auto"] = {
                    "status": "error",
                    "error": str(e)[:200]
                }
            
            # Alternative 2: with_structured_output json_mode
            try:
                structured_llm_json = base_llm.with_structured_output(Plan, method="json_mode")
                response_json = structured_llm_json.invoke(test_messages)
                results["with_structured_output_json"] = {
                    "status": "success",
                    "response_type": str(type(response_json)),
                    "is_plan_object": isinstance(response_json, Plan),
                    "is_none": response_json is None
                }
            except Exception as e:
                results["with_structured_output_json"] = {
                    "status": "error",
                    "error": str(e)[:200]
                }
                
        except Exception as e:
            results["alternatives_error"] = str(e)
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "test_results": results,
            "summary": {
                "current_approach_works": results.get("current_nodes_approach", {}).get("is_plan_object", False),
                "current_approach_returns_none": results.get("current_nodes_approach", {}).get("is_none", True),
                "alternative_methods_tested": len([k for k in results.keys() if k.startswith("with_structured")])
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Planner diagnostic test failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/api/diagnostics/workflow-test")
async def diagnostic_workflow_test():
    """Test minimal DeerFlow workflow to identify bottlenecks"""
    try:
        logger.info("🧪 Testing minimal workflow execution...")
        
        # Import workflow components
        from src.graph.nodes import planner_node
        from src.config.configuration import Configuration
        
        # Create minimal test state (matching actual workflow state)
        test_state = {
            "messages": [{"role": "user", "content": "Research renewable energy trends in 2025"}],
            "plan_iterations": 0,
            "enable_background_investigation": False,
            "locale": "en-US"
        }
        
        # Create test configuration (matching actual workflow config)
        test_config = {
            "thread_id": "diagnostic_test_" + str(int(datetime.now().timestamp())),
            "max_plan_iterations": 1,
            "max_step_num": 2,
            "max_search_results": 3
        }
        
        # Test planner_node execution directly
        try:
            result = planner_node(test_state, test_config)
            
            workflow_result = {
                "status": "success",
                "result_type": str(type(result)),
                "has_current_plan": "current_plan" in result.update if hasattr(result, 'update') and result.update else False,
                "goto_destination": getattr(result, 'goto', 'unknown'),
                "result_preview": str(result)[:400] if result else "None"
            }
            
            # If result has update with current_plan, analyze it
            if hasattr(result, 'update') and result.update and 'current_plan' in result.update:
                plan = result.update['current_plan']
                workflow_result.update({
                    "plan_object_type": str(type(plan)),
                    "plan_is_plan_instance": isinstance(plan, Plan),
                    "plan_title": getattr(plan, 'title', 'N/A') if plan else 'N/A',
                    "plan_steps_count": len(getattr(plan, 'steps', [])) if plan else 0
                })
            
        except Exception as workflow_error:
            workflow_result = {
                "status": "error",
                "error": str(workflow_error),
                "traceback": traceback.format_exc()
            }
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "workflow_test": workflow_result,
            "test_configuration": test_config
        }
        
    except Exception as e:
        logger.error(f"❌ Workflow diagnostic test failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# 🔧 END DIAGNOSTIC ENDPOINTS

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
