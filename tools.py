from openai import OpenAI, AsyncOpenAI
import requests
from pathlib import Path
import asyncio
import json
import toml
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
from anthropic import AsyncAnthropic
import streamlit as st

# Check if the API key exists in Streamlit secrets or environment variables
try:
    # Try to get from secrets
    anthropic_api_key = st.secrets.get("anthropic", {}).get("api_key")
    if not anthropic_api_key:
        # Fall back to environment variables
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not anthropic_api_key:
        st.error("Anthropic API key not found in Streamlit secrets or environment variables.")
        st.info("Please add your API key to .streamlit/secrets.toml under [anthropic] section or set the ANTHROPIC_API_KEY environment variable.")
        st.stop()  # Stop execution if no API key is found
except Exception as e:
    st.error(f"Error retrieving Anthropic API key: {str(e)}")
    st.stop()

# Initialize the Anthropic client
client = AsyncAnthropic(api_key=anthropic_api_key)

# Get Perplexity API key from Streamlit secrets or environment variables
try:
    # Try to get from secrets
    perplexity_api_key = st.secrets.get("perplexity", {}).get("api_key")
    if not perplexity_api_key:
        # Fall back to environment variables
        perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
    
    if not perplexity_api_key:
        # Fallback to the hardcoded key as last resort
        perplexity_api_key = "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"
        st.warning("Using default Perplexity API key. For production use, please add your own key to .streamlit/secrets.toml under [perplexity] section.")
except Exception as e:
    # Fallback to the hardcoded key if any error occurs
    perplexity_api_key = "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"
    st.warning(f"Error retrieving Perplexity API key: {str(e)}. Using default key instead.")

async def topic_research(query: str) -> dict:
    """
    Research blockchain and crypto ecosystem developments using Perplexity's API.
    
    This function gathers comprehensive information about specified topics for Token Relations
    newsletters, with emphasis on factual accuracy, relevant metrics, and proper sources from official ecosystem websites and reputable analytics platforms.
    
    Args:
        query: The blockchain/crypto topic to research
        
    Returns:
        Dictionary with research_content, sources, and structured metrics
    """
    url = "https://api.perplexity.ai/chat/completions"
    
    structured_prompt = f"""
    Analyze the following topic and provide a structured response in JSON format with these exact keys:
    - research_content: detailed analysis and insights (str)
    - sources: an array of all sources used for the research and metrics (include name, type, and URL when available)
    - metrics: object containing relevant numerical data with these subfields:
        - market_metrics: market-related numerical data
        - ecosystem_metrics: ecosystem-related numerical data
        - growth_metrics: growth-related numerical data
    
    Topic: {query}
    
    Return ONLY valid JSON without any additional text.
    """
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are a crypto and web3 expert. Provide precise metrics and research in structured JSON format from reputable sources."
            },
            {
                "role": "user",
                "content": structured_prompt
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {perplexity_api_key}"
    }
    
    try:
        # Execute API request
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, json=payload, headers=headers)
        )
        
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Parse the JSON string from the response
        try:
            structured_data = json.loads(content)
            return structured_data
        except json.JSONDecodeError:
            return {
                "research_content": content,
                "metrics": {
                    "market_metrics": {},
                    "ecosystem_metrics": {},
                    "growth_metrics": {}
                },
                "sources": []
            }
            
    except Exception as e:
        # Return structured error response
        return {
            "research_content": f"ERROR: {str(e)}. Please refine query or research manually.",
            "metrics": {
                "market_metrics": {},
                "ecosystem_metrics": {},
                "growth_metrics": {}
            },
            "sources": [],
            "error": str(e)
        }

async def edit_for_style(evaluation_output: dict) -> dict:
    """
    Edit content to align with Token Relations' distinctive voice, audience needs, and brand ethos.
    
        evaluation_output: Dictionary containing:
            {
                "category_scores": {
                    "Factual Accuracy": number (0-10),
                    "Neutrality and Balance": number (0-10),
                    "Clarity and Accessibility": number (0-10),
                    "Depth and Context": number (0-10),
                    "Structural Integrity": number (0-10)
                },
                "overall_score": number (weighted average),
                "strengths": [string],
                "issues_to_address": [string],
                "marketing_language_detected": [string],
                "recommendation": "Publish" | "Minor Revisions" | "Major Revisions"
            }
        
    Returns:
        Dictionary with style-aligned content and insights
    """
    try:
        # Initialize the Anthropic client
        client = AsyncOpenAI(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1"
        )
        
        # Extract the content from the evaluation output - handle both string and dict inputs
        content = evaluation_output if isinstance(evaluation_output, str) else evaluation_output.get("refined_content", "")
        
        response = await client.chat.completions.create(
            model="claude-3-7-sonnet-latest",
            messages=[
                {
                    "role": "system",
                    "content": """You are the voice guardian for Token Relations. You understand that our distinctive value comes from our factual neutrality, contextual insight, and ability to make complex blockchain developments accessible without oversimplification. Our readers trust us precisely because we avoid hype cycles and marketing language while still conveying genuine significance through context and data."""
                },
                {
                    "role": "user",
                    "content": f"""Refine this blockchain newsletter to perfectly match Token Relations' distinctive style.

                    ABOUT TOKEN RELATIONS:
                    - We provide ecosystem relations for blockchains and protocols
                    - Our audience is informed professionals who need clear, factual insights
                    - We bridge the gap between technical complexity and practical significance
                    
                    OUR WRITING ETHOS:
                    - Data-focused but accessible (metrics with meaning)
                    - Technical but not technical for technical's sake
                    - Neutral without being dry (informative but engaging)
                    - Forward-looking without speculation
                    - Educational without condescension
                    
                    OUR UNIQUE VOICE:
                    - We use specific metrics attributed to sources
                    - We explain "why it matters" by connecting dots between developments
                    - We define technical concepts in context
                    - We avoid ALL marketing language ("revolutionary," "game-changing," etc.)
                    - We never claim something is "first" or "best"
                    - We let facts speak for themselves
                    
                    NEWSLETTER STRUCTURE:
                    1. "What happened:" - Concise, specific event with key metrics
                    2. "Why it matters:" - Connecting development to practical significance
                    3. "The big picture:" - Broader context and measured implications
                    
                    DRAFT CONTENT:
                    {content}
                    
                    Return JSON with:
                    1. "refined_content": The edited newsletter that captures our voice
                    2. "style_insights": Brief notes on key style adjustments made"""
                }
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        return {
            "refined_content": content,
            "style_insights": [f"Style refinement failed: {str(e)}"],
            "error": str(e)
        }
    

async def evaluate_newsletter(content: str, client_name: str = "") -> dict:
    """
    Evaluate newsletter content against Token Relations' quality standards.
    
    Args:
        content (str): The newsletter content to evaluate
        client_name (str, optional): Name of the client the newsletter is for
        
    Returns:
        dict: A structured evaluation with these fields:
            {
                "category_scores": {
                    "Factual Accuracy": number (0-10),
                    "Neutrality and Balance": number (0-10),
                    "Clarity and Accessibility": number (0-10),
                    "Depth and Context": number (0-10),
                    "Structural Integrity": number (0-10)
                },
                "overall_score": number (weighted average),
                "strengths": [string],
                "issues_to_address": [string],
                "marketing_language_detected": [string],
                "recommendation": "Publish" | "Minor Revisions" | "Major Revisions"
            }
    """
    try:
        # Initialize the Anthropic client
        client = AsyncOpenAI(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1"
        )
        
        # Prepare the evaluation prompt with client context if provided
        client_context = f"Client: {client_name}\n" if client_name else ""
        
        response = await client.chat.completions.create(
            model="claude-3-7-sonnet-latest",
            messages=[
                {
                    "role": "system",
                    "content": """You are a newsletter quality evaluator for Token Relations. 
                    Evaluate the content against our quality standards and return a JSON response.
                    
                    EVALUATION CRITERIA:
                    1. Factual Accuracy (Critical - 30%)
                    - Are all stated facts verifiably accurate?
                    - Are metrics current and properly contextualized?
                    - Are attributions correct and specific?
                    - Are technical explanations precise and error-free?
                    - Score 0-10
                    
                    2. Neutrality and Balance (Critical - 25%)
                    - Is the language free from marketing terminology and hype?
                    - Are multiple positive perspectives fairly represented?
                    - Is the tone consistently neutral while remaining informative?
                    - Are implications presented without bias or overstatement?
                    - Score 0-10
                    
                    3. Clarity and Accessibility (High Priority - 20%)
                    - Would someone with basic blockchain knowledge understand the content?
                    - Are technical concepts explained in accessible terms?
                    - Is the narrative flow logical and easy to follow?
                    - Are analogies appropriate and illuminating?
                    - Score 0-10
                    
                    4. Depth and Context (High Priority - 15%)
                    - Is sufficient background provided for context?
                    - Are connections made to broader industry trends?
                    - Is the significance adequately explained?
                    - Are insights provided beyond surface-level reporting?
                    - Score 0-10
                    
                    5. Structural Integrity (Medium Priority - 10%)
                    - Does the content follow the three-section format?
                    - Does each section fulfill its specific purpose?
                    - Does the flow move logically from what → why → broader implications?
                    - Is the length appropriate for the significance of the development?
                    - Score 0-10
                    
                    IMPORTANT: Return ONLY valid JSON without any additional text or formatting.
                    The response must be a single JSON object with these exact fields:
                    {
                        "category_scores": {
                            "Factual Accuracy": number (0-10),
                            "Neutrality and Balance": number (0-10),
                            "Clarity and Accessibility": number (0-10),
                            "Depth and Context": number (0-10),
                            "Structural Integrity": number (0-10)
                        },
                        "overall_score": number (weighted average),
                        "strengths": [string],
                        "issues_to_address": [string],
                        "marketing_language_detected": [string],
                        "recommendation": "Publish" | "Minor Revisions" | "Major Revisions"
                    }"""
                },
                {
                    "role": "user",
                    "content": f"{client_context}Evaluate this newsletter content:\n\n{content}"
                }
            ],
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response with error handling
        try:
            evaluation = json.loads(response.choices[0].message.content)
            # Validate required fields
            required_fields = ["category_scores", "overall_score", "strengths", "issues_to_address", "marketing_language_detected", "recommendation"]
            if not all(field in evaluation for field in required_fields):
                raise ValueError("Missing required fields in evaluation response")
            return evaluation
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")
        
    except Exception as e:
        # Return a clear error response with the same structure
        return {
            "category_scores": {
                "Factual Accuracy": 0,
                "Neutrality and Balance": 0,
                "Clarity and Accessibility": 0,
                "Depth and Context": 0,
                "Structural Integrity": 0
            },
            "overall_score": 0,
            "strengths": [],
            "issues_to_address": [f"Evaluation failed: {str(e)}"],
            "marketing_language_detected": [],
            "recommendation": "Evaluation Error"
        }


tools = [evaluate_newsletter, edit_for_style, topic_research]