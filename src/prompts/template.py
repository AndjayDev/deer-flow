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

# --- Avery's Note: RESTORING this function for compatibility ---
def get_prompt_template(prompt_name: str) -> str:
    """
    Load and return a prompt template using Jinja2. This is kept for
    compatibility with other parts of the application.
    """
    try:
        template = env.get_template(f"{prompt_name}.md")
        return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")


def _extract_block(full_text: str, block_name: str) -> str:
    """
    A helper function to extract a specific instruction block from a prompt file.
    """
    pattern = re.compile(
        rf"\[{block_name}\](.*?)\[/{block_name}\]", re.DOTALL
    )
    match = pattern.search(full_text)
    if not match:
        return ""
    return match.group(1).strip()


# --- Avery's Note: KEEPING our new, intelligent apply_prompt_template ---
def apply_prompt_template(
    prompt_name: str, state: AgentState, configurable: Configuration = None
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.
    This function now intelligently selects the correct output instructions
    based on the active LLM provider.
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md"), "r") as f:
            prompt_content = f.read()

        instructions = ""
        if configurable and hasattr(configurable, 'llm') and hasattr(configurable.llm, 'provider'):
            active_provider = configurable.llm.provider
            
            if active_provider in JSON_MODE_PROVIDERS:
                instructions = _extract_block(prompt_content, "output_instructions_json_mode")
            else:
                instructions = _extract_block(prompt_content, "output_instructions_tool_mode")
        else:
            instructions = _extract_block(prompt_content, "output_instructions_tool_mode")

        final_prompt_text = prompt_content.replace("{{ OUTPUT_INSTRUCTIONS }}", instructions)

        state_vars = {
            "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
            **state,
        }

        if configurable:
            state_vars.update(dataclasses.asdict(configurable))

        template = env.from_string(final_prompt_text)
        system_prompt = template.render(**state_vars)
        
        return [{"role": "system", "content": system_prompt}] + state["messages"]

    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")
