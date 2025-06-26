# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import re
import dataclasses
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.prebuilt.chat_agent_executor import AgentState
from src.config.configuration import Configuration

# --- Avery's Note: Define which LLM providers require the strict JSON mode ---
JSON_MODE_PROVIDERS = ["openai", "groq", "perplexity"]

# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _extract_block(full_text: str, block_name: str) -> str:
    """
    A helper function to extract a specific instruction block from a prompt file.
    It looks for a block defined by [block_name] and [/block_name].
    """
    pattern = re.compile(
        rf"\[{block_name}\](.*?)\[/{block_name}\]", re.DOTALL
    )
    match = pattern.search(full_text)
    if not match:
        # If the specific block isn't found, return an empty string.
        return ""
    return match.group(1).strip()


def apply_prompt_template(
    prompt_name: str, state: AgentState, configurable: Configuration = None
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.
    This function now intelligently selects the correct output instructions
    based on the active LLM provider.
    """
    try:
        # --- Avery's Note: Step 1: Load the raw template content first ---
        with open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md"), "r") as f:
            prompt_content = f.read()

        instructions = ""
        # --- Avery's Note: Step 2: Determine which provider is active ---
        # We check the provider from the 'configurable' object passed into the function.
        if configurable and hasattr(configurable, 'llm') and hasattr(configurable.llm, 'provider'):
            active_provider = configurable.llm.provider
            
            # --- Avery's Note: Step 3: Select the correct instruction block ---
            if active_provider in JSON_MODE_PROVIDERS:
                # For OpenAI, Groq, etc., we grab the JSON mode instructions.
                instructions = _extract_block(prompt_content, "output_instructions_json_mode")
            else:
                # For everyone else (Gemini, Anthropic), we default to Tool Calling mode.
                instructions = _extract_block(prompt_content, "output_instructions_tool_mode")
        
        else:
            # Fallback if the provider can't be determined, default to tool mode.
            instructions = _extract_block(prompt_content, "output_instructions_tool_mode")

        # --- Avery's Note: Step 4: Inject the chosen instructions ---
        # We replace the placeholder in the prompt with the block we selected.
        final_prompt_text = prompt_content.replace("{{ OUTPUT_INSTRUCTIONS }}", instructions)

        # Convert state to dict for template rendering
        state_vars = {
            "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
            **state,
        }

        # Add configurable variables
        if configurable:
            state_vars.update(dataclasses.asdict(configurable))

        # --- Avery's Note: Step 5: Render the final prompt with Jinja2 ---
        # We now render the modified prompt text.
        template = env.from_string(final_prompt_text)
        system_prompt = template.render(**state_vars)
        
        return [{"role": "system", "content": system_prompt}] + state["messages"]

    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")
