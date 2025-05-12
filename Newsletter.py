import toml
import os
import os
import sys
import streamlit as st
from typing import Optional
from datetime import datetime
from supabase import create_client, Client  # Import Supabase client
import json
from langchain_anthropic import ChatAnthropic  # Changed from langchain.llms
from langchain_openai import ChatOpenAI
from PIL import Image
import io
import base64
import requests
from langchain.chains import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
import imageio
import numpy as np
from scipy.interpolate import interp1d
import math
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool
import asyncio
from anthropic import AsyncAnthropic
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic

from tools import topic_research, edit_for_style, evaluate_newsletter
# First, let's modify how we wrap the async functions in Tool objects
from functools import partial
import asyncio

# Create a wrapper function to handle async functions
def async_wrapper(func):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

# Wrap the async functions in Tool objects with the async wrapper
tools = [
    Tool(
        name="topic_research",
        description="Research blockchain and crypto ecosystem developments using Perplexity's API",
        func=async_wrapper(topic_research)
    ),
    Tool(
        name="edit_for_style",
        description="Edit content to align with Token Relations' distinctive voice and style",
        func=async_wrapper(edit_for_style)
    ),
    Tool(
        name="evaluate_newsletter",
        description="Evaluate newsletter content against Token Relations' quality standards",
        func=async_wrapper(evaluate_newsletter)
    )
]

# Create the ReAct prompt template with required variables
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the newsletter editing agent for Token Relations, a firm providing ecosystem relations for blockchains and protocols. Your primary responsibility is to evaluate and refine newsletter content to ensure it meets our quality standards and style guidelines.

Core Mission
You evaluate newsletter content and make necessary edits to improve clarity, neutrality, and effectiveness while maintaining our distinctive voice.

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

TASK HANDLING:
1. For simple tasks (like research):
   - Execute the task ONCE
   - When you receive the research results, you MUST:
     a. Acknowledge the results
     b. End with "TASK COMPLETE"
     c. DO NOT make another tool call
   - Example format:
     Thought: I have received the research results. The task is complete.
     Final Answer: TASK COMPLETE - [Research results summary]

2. For complex tasks (like newsletter editing):
   - Follow the full evaluation and editing process
   - Continue until quality standards are met
   - Show all changes and improvements
   - After final edit, ALWAYS end with "TASK COMPLETE"

Available tools:
{tools}

Tool names:
{tool_names}

{agent_scratchpad}

IMPORTANT: You must follow this EXACT format for each step:
Thought: (your reasoning about what to do)
Action: (the tool to use)
Action Input: (the input for the tool)
Observation: (the result of the tool)

For simple tasks, after receiving results:
Thought: I have received the research results. The task is complete.
Final Answer: TASK COMPLETE - [Brief summary of results]

For complex tasks, end with:
Thought: I have completed the evaluation and editing process
Final Answer: TASK COMPLETE - (show the user):
1. Original content
2. Evaluation results with scores and recommendations
3. Edited content with style improvements
4. List of specific changes made

STOPPING CONDITIONS:
1. After executing a simple task ONCE
2. After completing a complex task's evaluation and editing
3. When you see "TASK COMPLETE" in your own output
4. After three editing attempts
5. When the overall score reaches 8 or higher

DO NOT:
- Execute the same tool multiple times for simple tasks
- Continue after seeing "TASK COMPLETE"
- Skip showing any changes to the user
- Leave any edits unexplained
- Use marketing language or hype
- Make claims without specific metrics or sources
- Oversimplify technical concepts
- Use absolute terms without evidence
- Include speculative statements without context"""),
    ("human", "{input}")
])

# Create the agent with the proper prompt template and error handling
agent = create_react_agent(
    llm=ChatAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model="claude-3-7-sonnet-latest",
        temperature=0,
        timeout=None,
        max_retries=2,
        streaming=True
    ),
    tools=tools,
    prompt=prompt
)

# Create and store the agent executor with corrected configuration
st.session_state.agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,  # Increased from 2 to 10
    return_intermediate_steps=True,
    timeout=300  # 5 minute timeout
)

# Try to import Google Gemini, but silently handle if it's not available
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# ----------------------------------------------------------------------
# Install Dependencies (via terminal/pip):
#   pip install langchain anthropic streamlit supabase
#   or adapt to your environment as needed.
# --------------------------
# --- Helper function to fetch recent tweets ---
def fetch_recent_tweets(supabase: Optional[Client], clients: list, limit_per_fetch: int = 100):
    """Fetches recent tweets for specified clients from Supabase and returns a single sorted list."""
    all_tweets_list = []
    if not supabase or not clients:
        st.warning("Supabase client not available or no clients selected.")
        return all_tweets_list

    try:
        clients_lower = [c.lower() for c in clients]
        st.write(f"Fetching up to {limit_per_fetch} most recent tweets across clients: {', '.join(clients_lower)}") # Debugging output

        response = (
            supabase.table('tweets')
            .select('tweet_id, client, text, url, created_at, likes, retweets, replies, is_quote')
            .in_('client', clients_lower)
            .order('created_at', desc=True)
            .limit(limit_per_fetch) # Fetch a single batch of the most recent tweets across selected clients
            .execute()
        )

        if response.data:
            st.write(f"Fetched {len(response.data)} total tweets from Supabase.") # Debugging
            for tweet in response.data:
                # Parse datetime
                if tweet.get('created_at'):
                    try:
                        tweet['created_at'] = datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        tweet['created_at'] = datetime.now() # Fallback
                else:
                    tweet['created_at'] = datetime.now() # Fallback

                # Ensure client name is capitalized for display consistency
                tweet['client'] = tweet.get('client', 'unknown').capitalize()

                all_tweets_list.append(tweet)

            # The list is already sorted by date due to the Supabase query order
            st.write(f"Processed {len(all_tweets_list)} tweets.") # Debugging

        else:
             st.warning(f"No tweets found for the selected clients in the database.")
             if response.error:
                 st.error(f"Supabase error: {response.error.message}")

    except Exception as e:
        st.error(f"Error fetching tweets from Supabase: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

    return all_tweets_list

# --- Helper function to fetch links for a client ---
def fetch_client_links(supabase: Optional[Client], client: str, limit: int = 20):
    """Fetches recent links for a specified client from the 'links' table."""
    links_list = []
    if not supabase or not client:
        st.warning("Supabase client not available or client not selected.")
        return links_list

    client_lower = client.lower()
    st.write(f"Fetching links for: {client_lower}") # Debug

    try:
        response = (
            supabase.table('links')
            # Select relevant columns - 'url' and 'title' seem most useful
            .select('url, title, first_seen, last_checked')
            .eq('client', client_lower)
            .order('last_checked', desc=True) # Order by most recently checked/seen
            .limit(limit)
            .execute()
        )

        if response.data:
            st.write(f"Fetched {len(response.data)} links for {client} from DB.") # Debug
            # Format for display
            for link in response.data:
                 link_title = link.get('title') or link.get('url') # Use URL as fallback title
                 link_url = link.get('url')
                 last_checked_raw = link.get('last_checked')
                 try:
                     last_checked_dt = datetime.fromisoformat(last_checked_raw.replace('Z', '+00:00')) if last_checked_raw else None
                     last_checked_str = last_checked_dt.strftime('%Y-%m-%d') if last_checked_dt else 'N/A'
                 except:
                     last_checked_str = 'Invalid Date'

                 links_list.append({
                     "title": link_title,
                     "url": link_url,
                     "last_checked": last_checked_str
                 })

        else:
            st.info(f"No links found for {client} in the database.")
            # Check for explicit errors if needed
            # if response.error: st.error(...)

    except Exception as e:
        st.error(f"Error fetching links for {client}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

    return links_list

# --- Helper function to format fetched data for context ---
def format_fetched_data_for_context(tweets: list, links: list) -> str:
    """Formats fetched tweets and links into a string suitable for context."""
    context_str = ""

    if tweets:
        context_str += "=== RECENT TWEETS ===\n"
        for tweet in tweets:
             tweet_date = tweet.get('created_at')
             display_date = tweet_date.strftime('%Y-%m-%d %H:%M') if tweet_date else 'N/A'
             tweet_text = tweet.get('text', 'N/A')
             tweet_url = tweet.get('url', '#')
             context_str += f"- ({display_date}): {tweet_text} [Link]({tweet_url})\n"
        context_str += "\n" # Add spacing

    if links:
        context_str += "=== RECENT/RELEVANT LINKS ===\n"
        for link in links:
             link_title = link.get('title', 'N/A')
             link_url = link.get('url', '#')
             last_checked = link.get('last_checked', 'N/A')
             context_str += f"- Title: {link_title}\n  URL: {link_url}\n  (Last Checked: {last_checked})\n"
        context_str += "\n" # Add spacing

    return context_str

########################
# 1. CLIENT FILES SETUP #
########################
CLIENT_FILES = {
    "Aptos": "summaries/aptos_summary.txt",
    "Avalanche": "summaries/avalanche_summary.txt",
    "BitcoinOS": "summaries/bitcoinos_summary.txt",
    "Core DAO": "summaries/core_dao_summary.txt",
    "DeCharge": "summaries/decharge_summary.txt",
    "Flow": "summaries/flow_summary.txt",
    "Injective": "summaries/injective_summary.txt",
    "Matchain": "summaries/matchain_summary.txt",
    "Optimism": "summaries/optimism_summary.txt",
    "Polygon": "summaries/polygon_summary.txt",
    "Ripple": "summaries/ripple_summary.txt",
    "The Root Network": "summaries/the_root_network_summary.txt",
    "Sei": "summaries/sei_summary.txt",
}

# Client-specific style references - each client has their own folder but with consistent file names
CLIENT_STYLE_FOLDERS = {
    "Aptos": "style_references/aptos",
    "Avalanche": "style_references/avalanche",
    "BitcoinOS": "style_references/bitcoinos",
    "Core DAO": "style_references/core_dao",
    "DeCharge": "style_references/decharge",
    "Flow": "style_references/flow",
    "Injective": "style_references/injective",
    "Matchain": "style_references/matchain",
    "Optimism": "style_references/optimism",
    "Polygon": "style_references/polygon",
    "Ripple": "style_references/ripple",
    "The Root Network": "style_references/the_root_network",
    "Sei": "style_references/sei"
}

# Style file names are consistent across all clients
STYLE_FILE_NAMES = {
    "Technology Focused": "technology_style.txt",
    "Metrics Driven": "metrics_style.txt",
    "Partnership/New Development": "partnership_style.txt",
}

# Global bullet point example files - same for all clients
BULLET_POINT_EXAMPLES = {
    "Ecosystem": "bullet_point_examples/ecosystem_examples.txt",
    "Community": "bullet_point_examples/community_examples.txt",
}

# Tweet example files
TWEET_EXAMPLES = {
    "Short": "tweet_examples/short_tweets.txt",
    "Medium": "tweet_examples/medium_tweets.txt",
    "Long": "tweet_examples/long_tweets.txt",
}

# Title examples file
TITLE_EXAMPLES_FILE = "title_examples/titles.txt"

# Default folder for style references
DEFAULT_STYLE_FOLDER = "style_references/default"

###################
# 2. ANTHROPIC SETUP
###################

# Function to validate API key
def validate_anthropic_api_key(api_key):
    """Test if the provided Anthropic API key is valid"""
    import requests
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Simple validation request
    try:
        # Just make a minimal request to check if the API key is valid
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": "claude-3-7-sonnet-latest",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}]
            },
            timeout=5
        )
        
        if response.status_code == 200:
            return True, "API key is valid"
        elif response.status_code == 401:
            return False, "Invalid API key. Please check and try again."
        else:
            return False, f"API error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Error validating API key: {str(e)}"

# Check for API key in secrets first, then environment or session state
try:
    # Try to get from secrets, but don't fail if secrets file is missing
    api_key = ""
    try:
        api_key = st.secrets["anthropic"]["api_key"]
        # If we get here, the key was found in secrets
        os.environ["ANTHROPIC_API_KEY"] = api_key  # Set it in environment for libraries that look there
    except (KeyError, TypeError, FileNotFoundError):
        # Secrets file not found or key not in secrets - this is expected in some environments
        pass
        
    # If not in secrets, try environment or session state
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", st.session_state.get("ANTHROPIC_API_KEY", ""))
    
    # Only show the input form if no key is found anywhere
    if not api_key:
        st.warning("Your Anthropic API key is not set. Please enter it below:")
        api_key_input = st.text_input("Enter your Anthropic API key:", type="password")
        
        if api_key_input and st.button("Validate and Save API Key"):
            is_valid, message = validate_anthropic_api_key(api_key_input)
            if is_valid:
                st.session_state.ANTHROPIC_API_KEY = api_key_input
                os.environ["ANTHROPIC_API_KEY"] = api_key_input
                st.success("API key validated and saved for this session!")
                st.experimental_rerun()
            else:
                st.error(message)
        
        # Show instructions for getting an API key
        with st.expander("How to get an Anthropic API key"):
            st.markdown("""
            1. Go to [Anthropic Console](https://console.anthropic.com/)
            2. Sign up or log in to your account
            3. Navigate to the API Keys section
            4. Create a new API key
            5. Copy the key and paste it above
            """)
        
        # Stop execution if no valid API key
        st.stop()
except Exception as e:
    # Catch any other unexpected errors
    st.error(f"Error checking for API key: {str(e)}")
    st.warning("Please enter your Anthropic API key below:")
    
    api_key_input = st.text_input("Enter your Anthropic API key:", type="password")
    
    if api_key_input and st.button("Validate and Save API Key"):
        is_valid, message = validate_anthropic_api_key(api_key_input)
        if is_valid:
            st.session_state.ANTHROPIC_API_KEY = api_key_input
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
            st.success("API key validated and saved for this session!")
            st.experimental_rerun()
        else:
            st.error(message)
    
    # Stop execution if no valid API key
    st.stop()

# Initialize the Anthropic client with the API key
try:
    anthropic_llm = ChatAnthropic(
        api_key=api_key,
        model="claude-3-7-sonnet-latest",
        temperature=0,
        timeout=None,
        max_retries=2,
    )
    # Test the client with a simple query to verify it works
    test_response = anthropic_llm.invoke("Test connection. Reply with 'Connected'.")
    if "Connected" not in test_response.content:
        st.sidebar.warning("API connection test failed. The API key might be invalid.")
except Exception as e:
    st.error(f"Error initializing Anthropic client: {str(e)}")
    st.warning("Please check your API key and try again.")
    st.stop()

###################
# 3. MULTI-MODEL SETUP
###################

# Define available models
AVAILABLE_MODELS = {
    "Claude 3.7 Sonnet": {
        "provider": "anthropic",
        "model_name": "claude-3-7-sonnet-latest",
        "description": "Anthropic's Claude 3.7 Sonnet"
    },
    "GPT-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "description": "OpenAI's GPT-4o"
    },
    "OpenAI o1-mini": {
        "provider": "openai",
        "model_name": "o1-mini",
        "description": "OpenAI's o1-mini"
    },
    "Google Gemini 2.0": {
        "provider": "google",
        "model_name": "gemini-1.5-pro",
        "description": "Google's Gemini 2.0 Pro"
    }
}

# Function to initialize the selected model
def initialize_llm(model_key):
    model_info = AVAILABLE_MODELS.get(model_key)
    if not model_info:
        st.error(f"Unknown model: {model_key}")
        return None
    
    provider = model_info["provider"]
    model_name = model_info["model_name"]
    
    try:
        if provider == "anthropic":
            # Check if we have an Anthropic API key
            if not api_key:
                st.error("Anthropic API key not set. Cannot use Claude models.")
                return None
            
            return ChatAnthropic(
                api_key=api_key,
                model=model_name,
                temperature=0,
                timeout=None,
                max_retries=2,
            )
        
        elif provider == "openai":
            # Check for OpenAI API key
            openai_api_key = os.environ.get("OPENAI_API_KEY", st.session_state.get("OPENAI_API_KEY", ""))
            
            if not openai_api_key:
                st.error("OpenAI API key not set. Cannot use GPT models.")
                return None
            
            return ChatOpenAI(
                api_key=openai_api_key,
                model=model_name,
                temperature=0,
            )
        
        elif provider == "google":
            if not GOOGLE_AVAILABLE:
                st.error("Google Gemini integration is not available. Install langchain-google-genai package.")
                return None
                
            # Check for Google API key
            google_api_key = st.secrets.get("google", {}).get("api_key", os.environ.get("GOOGLE_API_KEY", ""))
            
            if not google_api_key:
                with st.expander("Set Google API Key"):
                    google_api_key = st.text_input("Google API Key", type="password", key="google_api_key_input")
                    if st.button("Save Google Key"):
                        if google_api_key:
                            st.session_state.google_api_key = google_api_key
                            st.success("Google API key saved for this session!")
                            st.experimental_rerun()
                
                if not google_api_key and not st.session_state.get("google_api_key"):
                    st.error("Google API key not set. Cannot use Gemini models.")
                    return None
                
                # Use the key from session state if available
                if not google_api_key and st.session_state.get("google_api_key"):
                    google_api_key = st.session_state.google_api_key
            
            return ChatGoogleGenerativeAI(
                api_key=google_api_key,
                model=model_name,
                temperature=0,
            )
        
        else:
            st.error(f"Unsupported provider: {provider}")
            return None
            
    except Exception as e:
        st.error(f"Error initializing {model_key}: {str(e)}")
        return None

# Add model selection to the sidebar
with st.sidebar:
    st.title("Model Settings")
    
    selected_model = st.selectbox(
        "Select LLM Provider",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,  # Default to Claude 3.7 Sonnet
        format_func=lambda x: f"{x} ({AVAILABLE_MODELS[x]['provider']})"
    )
    
    # Show model description
    st.info(AVAILABLE_MODELS[selected_model]["description"])
    
    # Initialize the selected model
    if "current_model" not in st.session_state or st.session_state.current_model != selected_model:
        with st.spinner(f"Initializing {selected_model}..."):
            llm = initialize_llm(selected_model)
            if llm:
                st.session_state.llm = llm
                st.session_state.current_model = selected_model
                st.success(f"{selected_model} initialized successfully!")
            else:
                st.error(f"Failed to initialize {selected_model}")
                # Update the fallback model in the sidebar section
                if api_key and "llm" not in st.session_state:
                    st.warning("Falling back to Claude 3.7 Sonnet...")
                    st.session_state.llm = ChatAnthropic(
                        api_key=api_key,
                        model="claude-3-7-sonnet-latest",
                        temperature=0
                    )
                    st.session_state.current_model = "Claude 3.7 Sonnet"

##########################
# 4. DEFINING PROMPTS
##########################
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# (A) EXTRACT POINTS PROMPT
extract_points_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Analyst: Extract key points from text for web3 newsletter. Identify: 1. What Happened (1-2 sentences), 2. Why Important (6-7 paragraphs), 3. Bigger Picture (3 paragraphs). Return structured bullet points/headings. Concise, unbiased, informative style."
    ),
    (
        "human",
        "Context:\n{context_text}\n\n"
        "List all relevant key points for 'What Happened', 'Why Is It Important', and 'The Bigger Picture'."
    )
])

# (B) DRAFT 'WHAT HAPPENED' PROMPT
draft_what_happened_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Newsletter Writer: Draft 'What Happened' (1-2 sentences) for web3 newsletter. Match example style. Concise, neutral, clear."
    ),
    (
        "human",
        "Key Points:\n{key_points}\n\n"
        "Newsletter Topic: {topic}\n\n"
        "Draft a 'What happened' section (1–2 sentences) in a style similar to this example:\n{newsletter_example}\n\n"
        "{additional_instructions}"  # This will be an empty string if no additional instructions
    )
])

# (C) DRAFT 'WHY IT MATTERS' PROMPT
draft_why_matters_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Newsletter Writer: Draft 'Why It Matters' (3-4 paragraphs) for web3 newsletter. Match example style. Explain significance, impact, implications. Concise, neutral."
    ),
    (
        "human",
        "Key Points:\n{key_points}\n\n"
        "Newsletter Topic: {topic}\n\n"
        "Draft the 'Why does it matter' section (6–7 small paragraphs) in a style similar to this example:\n{newsletter_example}"
        "{additional_instructions}"
    )
])

# COMBINED 'THE BIG PICTURE' PROMPT (Draft and Enhance)
combined_big_picture_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Newsletter Editor & Writer:  Add to 'The Big Picture' (total 2-3 paragraphs) for web3 newsletter. Style, keeping it concise and neutral. Enhance it to add deeper insight, long-term vision, and ecosystem impact, informed by the client vision and based on documentation. Return the final enhanced 'The Big Picture' section."
    ),
    (
        "human",
        "Key Points (for initial draft):\n{key_points}\n\nNewsletter Topic:\n{topic}\n\nNewsletter Example (for style reference):\n{newsletter_example}\n\nClient vision + docs (for enhancement):\n{long_term_doc}\n\n"
        "{additional_instructions}"
    )
])


# (F) STYLE CHECK PROMPT
style_check_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Style Expert: Compare generated newsletter style to example. Check tone, paragraph/sentence structure, clarity, language, section lengths. Give actionable feedback."
        "Add simple explanations on complex topics that needs developer or financial market expertise, use analogies sparingly and when it makes sense"
        "Use words like can when highlighting the larger impact rather than absolutes like will and significant and other strong language" 
    ),
    (
        "human",
        "Example Newsletter:\n{newsletter_example}\n\n"
        "Final Enhanced Newsletter:\n{enhanced_newsletter}\n\n"
        "Compare them in detail and suggest improvements to match the Style/Vibe: simple, concise, informative, positive neutral, non bias."
    )
])

# (G) STYLE EDIT PROMPT
style_edit_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Editor: Revise newsletter based on style feedback to match example. Focus on tone, structure, conciseness, clarity. Output final newsletter."
    ),
    (
        "human",
        "Style Feedback:\n{style_comparison}\n\n"
        "Draft Newsletter:\n{newsletter_enhanced}\n\n" # Note: using newsletter_enhanced as input now.
        "Example Newsletter:\n{newsletter_example}\n\n"
        "Revise the Draft Newsletter based on the feedback to match the example's style."
    )
])

# (H) IMPROVED ECOSYSTEM BULLET POINT PROMPT
ecosystem_bullet_point_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Generate concise ecosystem bullet points for web3 newsletters. Each bullet should:\n"
        "- Start with an action verb (Check out, Learn about, Tune into, etc.)\n"
        "- Be brief (15-25 words maximum per bullet)\n"
        "- Focus on specific ecosystem events, partnerships, integrations, or content\n"
        "- Use present tense\n"
        "- Include specific names, platforms, and clear value propositions\n"
        "- Avoid marketing language or excessive technical details\n"
        "- End with minimal or no punctuation"
    ),
    (
        "human",
        "Context:\n{context_text}\n\n"
        "Example Bullet Points (carefully emulate this style and length):\n{example_bullet_points}\n\n"
        "Please generate ecosystem bullet points following this format:\n"
        "- [Action verb] [subject] [specific detail about partnership/integration/event]\n"
        "- [Action verb] [another announcement or update] [specific platform or benefit]\n\n"
        "Focus on the most important developments, keeping each bullet short and direct."
    )
])

# (I) IMPROVED COMMUNITY BULLET POINT PROMPT
community_bullet_point_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Generate concise community bullet points for web3 newsletters. Each bullet should:\n"
        "- Start with an action verb or question phrase (Dive into, Join, Going to, Interested in, Tune in, Learn how)\n"
        "- Be brief (10-20 words maximum per bullet)\n"
        "- Focus on community engagement, events, programs, or educational opportunities\n"
        "- Use a friendly, inviting tone with direct calls to action\n"
        "- Include specific community programs, events, or platforms\n"
        "- Emphasize participation and involvement\n"
        "- End with minimal or no punctuation"
    ),
    (
        "human",
        "Context:\n{context_text}\n\n"
        "Example Bullet Points (carefully emulate this style and length):\n{example_bullet_points}\n\n"
        "Please generate community bullet points following this format:\n"
        "- [Action verb/Question] [community opportunity] [specific platform or event]\n"
        "- [Action verb/Question] [another community activity] [how to get involved]\n\n"
        "Focus on encouraging community participation and highlighting upcoming events or programs."
    )
])

# (J) NEW TWEET GENERATION PROMPT
tweet_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Social Media Writer: Create web3 tweets that are concise, informative, and match example style. Highlight key developments, use appropriate hashtags and mentions, and maintain neutral yet engaging tone. Focus on facts over hype."
    ),
    (
        "human",
        "Context/Newsletter Content:\n{newsletter_content}\n\n"
        "Tweet Format: {tweet_format}\n\n"
        "Example Tweets (carefully emulate this style):\n{tweet_examples}\n\n"
        "Client Name: {client_name}\n\n"
        "Please generate a {tweet_format} tweet that summarizes the key information from the newsletter content in a style similar to the examples. Include appropriate @ mentions of the client and relevant hashtags."
    )
])

# (K) NEW TITLE GENERATION PROMPT
title_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Newsletter Title Writer: Create concise, engaging titles for web3 newsletters that match example style. Highlight key developments while maintaining neutral yet engaging tone. Focus on clarity and accuracy."
    ),
    (
        "human",
        "Newsletter Content:\n{newsletter_content}\n\n"
        "Example Titles (carefully emulate this style):\n{title_examples}\n\n"
        "Client Name: {client_name}\n\n"
        "Please generate a title for this newsletter that captures the key information and matches the style of the examples."
    )
])

# Update the image generation prompt for professional, impactful imagery
image_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Create professional image prompts (15-25 words) for financial/tech publications like Bloomberg or The Times. Focus on 3-5 key elements maximum. Use clean, impactful imagery with strong visual hierarchy. Prefer abstract representations, data visualizations, or symbolic elements. Avoid text, faces, and cluttered scenes."
    ),
    (
        "human",
        "Newsletter topic: {newsletter_title}\nClient: {client_name}\n\nCreate a Bloomberg/Times-style image prompt (15-25 words) with only 3-5 key visual elements for a professional newsletter cover."
    )
])

################################
# 5. CREATE LLM CHAINS
################################
# Update the initialize_chains function to include the bullet point chains
def initialize_chains():
    # Use the selected model from session state, or fall back to anthropic_llm
    llm = st.session_state.get("llm", anthropic_llm)
    
    # Initialize all chains with the selected model
    chain_extract = LLMChain(llm=llm, prompt=extract_points_prompt)
    chain_what_happened = LLMChain(llm=llm, prompt=draft_what_happened_prompt)
    chain_why_matters = LLMChain(llm=llm, prompt=draft_why_matters_prompt)
    chain_combined_big_picture = LLMChain(llm=llm, prompt=combined_big_picture_prompt)
    chain_style = LLMChain(llm=llm, prompt=style_check_prompt)
    chain_edit = LLMChain(llm=llm, prompt=style_edit_prompt)
    chain_tweet_generation = LLMChain(llm=llm, prompt=tweet_generation_prompt)
    chain_title_generation = LLMChain(llm=llm, prompt=title_generation_prompt)
    chain_ecosystem_bullet_point_prompt = LLMChain(llm=llm, prompt=ecosystem_bullet_point_prompt)
    chain_community_bullet_point_prompt = LLMChain(llm=llm, prompt=community_bullet_point_prompt)
    
    return {
        "extract": chain_extract,
        "what_happened": chain_what_happened,
        "why_matters": chain_why_matters,
        "big_picture": chain_combined_big_picture,
        "style": chain_style,
        "edit": chain_edit,
        "tweet": chain_tweet_generation,
        "title": chain_title_generation,
        "ecosystem_bullet": chain_ecosystem_bullet_point_prompt,
        "community_bullet": chain_community_bullet_point_prompt
    }

# Initialize chains
chains = initialize_chains()
chain_extract = chains["extract"]
chain_what_happened = chains["what_happened"]
chain_why_matters = chains["why_matters"]
chain_combined_big_picture = chains["big_picture"]
chain_style = chains["style"]
chain_edit = chains["edit"]
chain_tweet_generation = chains["tweet"]
chain_title_generation = chains["title"]
chain_ecosystem_bullet_point_prompt = chains["ecosystem_bullet"]
chain_community_bullet_point_prompt = chains["community_bullet"]

# Function to load bullet point examples
def load_bullet_point_examples(bullet_point_type):
    # Load from global example files
    file_path = BULLET_POINT_EXAMPLES[bullet_point_type]
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

# Function to load tweet examples
def load_tweet_examples(tweet_format):
    file_path = TWEET_EXAMPLES.get(tweet_format)
    if not file_path:
        return ""
    
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

# Function to load title examples
def load_title_examples():
    try:
        with open(TITLE_EXAMPLES_FILE, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

################################
# 7. DATABASE SETUP FOR FINE-TUNING WITH SUPABASE
################################

# Update the Supabase configuration section to include the table name from environment variables

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://aatuvtaqtvfqeaaopfvq.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFhdHV2dGFxdHZmcWVhYW9wZnZxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzcyMjYzOTAsImV4cCI6MjA1MjgwMjM5MH0.-uEtosE4q40N6Vuf-EQsO-JAIeQ0k0q3tLAvHrHGF34")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "NL-Fine-Tuning")

# Function to initialize Supabase client
def initialize_supabase():
    # Check if Supabase credentials are set
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.sidebar.warning("Supabase credentials not set. Data will not be saved.")
        
        # Allow setting credentials in the UI if not in environment
        with st.sidebar.expander("Set Supabase Credentials"):
            supabase_url = st.text_input("Supabase URL", type="password")
            supabase_key = st.text_input("Supabase API Key", type="password")
            
            if st.button("Save Credentials"):
                if supabase_url and supabase_key:
                    # Store in session state (not persistent across app restarts)
                    st.session_state.supabase_url = supabase_url
                    st.session_state.supabase_key = supabase_key
                    # Initialize the client with the new credentials
                    try:
                        st.session_state.supabase_client = create_client(supabase_url, supabase_key)
                        st.success("Credentials saved and connected for this session!")
                    except Exception as e:
                        st.error(f"Failed to connect to Supabase: {str(e)}")
                else:
                    st.error("Please provide both URL and API key")
    else:
        # Store environment variables in session state and initialize client
        st.session_state.supabase_url = SUPABASE_URL
        st.session_state.supabase_key = SUPABASE_KEY
        try:
            st.session_state.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            st.sidebar.error(f"Failed to connect to Supabase: {str(e)}")

# Function to save newsletter data to Supabase
def save_to_supabase(client, topic, context, generated, final, style, quality_rating, feedback_notes):
    # Get Supabase client from session state
    supabase_client = st.session_state.get("supabase_client")
    
    if not supabase_client:
        st.error("Supabase client not initialized. Cannot save data.")
        return False
    
    # Prepare data for Supabase
    data = {
        "client": client,
        "topic": topic,
        "context_input": context,
        "generated_newsletter": generated,
        "final_newsletter": final,
        "style_reference": style,
        "quality_rating": quality_rating,
        "feedback_notes": feedback_notes,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        # Insert data using the Supabase client
        response = supabase_client.table(SUPABASE_TABLE).insert(data).execute()
        
        # Check if the insert was successful
        if response.data:
            return True
        else:
            st.error(f"Error saving to Supabase: {response.error}")
            return False
    except Exception as e:
        st.error(f"Exception when saving to Supabase: {str(e)}")
        return False

# Function to load saved newsletters from Supabase
def load_saved_newsletters():
    # Get Supabase client from session state
    supabase_client = st.session_state.get("supabase_client")
    
    if not supabase_client:
        st.error("Supabase client not initialized. Cannot load data.")
        return []
    
    try:
        # Query data using the Supabase client
        response = (
            supabase_client
            .table(SUPABASE_TABLE)
            .select("id,client,topic,created_at")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        
        # Return the data if successful
        if response.data:
            return response.data
        else:
            st.error(f"Error loading from Supabase: {response.error}")
            return []
    except Exception as e:
        st.error(f"Exception when loading from Supabase: {str(e)}")
        return []

# Add a function to get a specific newsletter by ID
def get_newsletter_by_id(newsletter_id):
    supabase_client = st.session_state.get("supabase_client")
    
    if not supabase_client:
        st.error("Supabase client not initialized. Cannot load data.")
        return None
    
    try:
        response = (
            supabase_client
            .table(SUPABASE_TABLE)
            .select("*")
            .eq("id", newsletter_id)
            .execute()
        )
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            st.error(f"Newsletter with ID {newsletter_id} not found.")
            return None
    except Exception as e:
        st.error(f"Exception when loading newsletter: {str(e)}")
        return None

# Add a function to export newsletters for fine-tuning
def export_newsletters_for_finetuning():
    supabase_client = st.session_state.get("supabase_client")
    
    if not supabase_client:
        st.error("Supabase client not initialized. Cannot export data.")
        return None
    
    try:
        response = (
            supabase_client
            .table(SUPABASE_TABLE)
            .select("context_input,final_newsletter,quality_rating")
            .gte("quality_rating", 7)  # Only export high-quality newsletters
            .execute()
        )
        
        if response.data:
            # Format data for fine-tuning
            finetuning_data = []
            for item in response.data:
                finetuning_data.append({
                    "input": item["context_input"],
                    "output": item["final_newsletter"]
                })
            return finetuning_data
        else:
            st.warning("No high-quality newsletters found for export.")
            return None
    except Exception as e:
        st.error(f"Exception when exporting newsletters: {str(e)}")
        return None

# Initialize Supabase on app startup
initialize_supabase()

################################
# 6. STREAMLIT UI (UPDATED)
################################
# Hide the "Created by" footer
hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

st.title("Token Relations 📊 Newsletter")

# Create tabs for Newsletter Generator, Bullet Point Generator, Tweet Generator, Title Generator, Cover Image, Graph Plotter, and Edit & Save
newsletter_tab, react_agent_tab, bullet_points_tab, tweet_tab, title_tab, image_tab, graph_tab, edit_save_tab = st.tabs([
    "Newsletter Generator", 
    "React Agent Chat",
    "Bullet Point Generator", 
    "Tweet Generator", 
    "Title Generator", 
    "Cover Image", 
    "Graph Plotter",
    "Edit & Save"
])

# Check for OpenAI API key in secrets first, then environment or session state
try:
    # Try to get from secrets, but don't fail if secrets file is missing
    openai_api_key = ""
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        # If we get here, the key was found in secrets
        os.environ["OPENAI_API_KEY"] = openai_api_key  # Set it in environment for libraries that look there
    except (KeyError, TypeError, FileNotFoundError):
        # Secrets file not found or key not in secrets - this is expected in some environments
        pass
        
    # If not in secrets, try environment or session state
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY", st.session_state.get("OPENAI_API_KEY", ""))
    
    # Only show the input form if no key is found anywhere
    if not openai_api_key:
        st.warning("Your OpenAI API key is not set. Please enter it below:")
        openai_api_key_input = st.text_input("Enter your OpenAI API key:", type="password", key="openai_api_key_input")
        
        if openai_api_key_input and st.button("Save OpenAI API Key"):
            st.session_state.OPENAI_API_KEY = openai_api_key_input
            os.environ["OPENAI_API_KEY"] = openai_api_key_input
            st.success("OpenAI API key saved for this session!")
            st.experimental_rerun()
        
        # Show instructions for getting an API key
        with st.expander("How to get an OpenAI API key"):
            st.markdown("""
            1. Go to [OpenAI Platform](https://platform.openai.com/)
            2. Sign up or log in to your account
            3. Navigate to the API Keys section
            4. Create a new API key
            5. Copy the key and paste it above
            """)
except Exception as e:
    # Catch any other unexpected errors
    st.error(f"Error checking for OpenAI API key: {str(e)}")

# Add this near the top of your app, after the imports
# Debug section to check API keys
with st.expander("Debug API Keys"):
    st.write("Checking API keys...")
    
    # Check OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        openai_key = st.session_state.get("OPENAI_API_KEY", "")
    
    if openai_key:
        st.success("OpenAI API key found!")
        # Show first few and last few characters
        masked_key = openai_key[:4] + "..." + openai_key[-4:]
        st.write(f"OpenAI API key: {masked_key}")
    else:
        st.error("OpenAI API key not found!")
    
    # Check if secrets are accessible
    try:
        has_secrets = hasattr(st, "secrets")
        st.write(f"Has secrets attribute: {has_secrets}")
        
        if has_secrets:
            has_openai_section = "openai" in st.secrets
            st.write(f"Has OpenAI section in secrets: {has_openai_section}")
            
            if has_openai_section:
                has_api_key = "api_key" in st.secrets.openai
                st.write(f"Has API key in OpenAI section: {has_api_key}")
    except Exception as e:
        st.error(f"Error checking secrets: {str(e)}")

# Add this near the beginning of your app, after imports
# Clear any old API keys from session state
if "openai_api_key" in st.session_state:
    # This is the old key name format, remove it
    del st.session_state.openai_api_key

with newsletter_tab:
    st.markdown(
        """
        **Newsletter Generator Steps:** \n
        **Step 1:** Extract key points from the context and client documentation + create detailed structure for each section of the newsletter \n
        **Step 2:** Human review and edit extracted key points.\n
        **Step 3:** Draft 'What happened' section.\n
        **Step 4:** Draft 'Why it matters' section.\n
        **Step 5:** Draft & Enhance 'The big picture' section. \n
        **Step 6:** Compare the generated newsletter's style against the example.\n
        **Step 7:** Apply style edits to the enhanced newsletter based on feedback.\n
        """
    )

    # --- Client selection & loading client documentation ---
    selected_client = st.selectbox("Select a Client", list(CLIENT_FILES.keys()), key="newsletter_client")
    long_term_doc = ""
    if selected_client:
        try:
            with open(CLIENT_FILES[selected_client], "r") as f:
                long_term_doc = f.read()
        except FileNotFoundError:
            st.error(f"File not found for {selected_client} at {CLIENT_FILES[selected_client]}.")

    # --- Style Reference Selection based on selected client ---
    # Determine which folder to use for style files
    if selected_client and selected_client in CLIENT_STYLE_FOLDERS:
        style_folder = CLIENT_STYLE_FOLDERS[selected_client]
        # Check if the folder exists
        if not os.path.exists(style_folder):
            st.warning(f"Client-specific style folder not found: {style_folder}. Using default styles.")
            style_folder = DEFAULT_STYLE_FOLDER
    else:
        # Fallback to default styles if client doesn't have a specific folder
        style_folder = DEFAULT_STYLE_FOLDER
        st.info(f"Using default style references for {selected_client}.")

    selected_style = st.selectbox("Select Newsletter Style Reference", list(STYLE_FILE_NAMES.keys()))
    newsletter_example = ""
    if selected_style:
        # Combine the folder path with the selected style filename
        style_file_path = os.path.join(style_folder, STYLE_FILE_NAMES[selected_style])
        try:
            with open(style_file_path, "r") as f:
                newsletter_example = f.read()
        except FileNotFoundError:
            st.error(f"Style file not found at {style_file_path}. Please ensure the file exists.")
            # Provide information about the expected directory structure
            st.info(f"Make sure to create the style file: {STYLE_FILE_NAMES[selected_style]} in the folder: {style_folder}")

    # Initialize session state for context_text if it doesn't exist
    if 'context_text' not in st.session_state:
        st.session_state.context_text = ""

    # Add tweet and link selection section
    st.markdown("### Recent Tweets and Links")

    # Create two columns for tweets and links
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Recent Tweets")
        # Fetch recent tweets for the selected client
        recent_tweets = fetch_recent_tweets(st.session_state.get("supabase_client"), [selected_client], limit_per_fetch=10)
        
        if recent_tweets:
            # Create a list of tweet options for the dropdown
            tweet_options = [f"{tweet['created_at'].strftime('%Y-%m-%d %H:%M')}: {tweet['text'][:50]}..." for tweet in recent_tweets]
            selected_tweets = st.multiselect(
                "Select tweets to include",
                options=tweet_options,
                key="tweet_selector"
            )
            
            # Show selected tweets in an expander
            if selected_tweets:
                with st.expander("Selected Tweets"):
                    for tweet_option in selected_tweets:
                        # Find the corresponding tweet in recent_tweets
                        tweet = next((t for t in recent_tweets if tweet_option.startswith(t['created_at'].strftime('%Y-%m-%d %H:%M'))), None)
                        if tweet:
                            st.markdown(f"- **{tweet['created_at'].strftime('%Y-%m-%d %H:%M')}**: {tweet['text']}")
                            st.markdown(f"  [Link]({tweet['url']})")
        else:
            st.info("No recent tweets found for this client.")

    with col2:
        st.markdown("#### Recent Links")
        # Fetch recent links for the selected client
        recent_links = fetch_client_links(st.session_state.get("supabase_client"), selected_client, limit=10)
        
        if recent_links:
            # Create a list of link options for the dropdown
            link_options = [f"{link['title']} (Last checked: {link['last_checked']})" for link in recent_links]
            selected_links = st.multiselect(
                "Select links to include",
                options=link_options,
                key="link_selector"
            )
            
            # Show selected links in an expander
            if selected_links:
                with st.expander("Selected Links"):
                    for link_option in selected_links:
                        # Find the corresponding link in recent_links
                        link = next((l for l in recent_links if link_option.startswith(l['title'])), None)
                        if link:
                            st.markdown(f"- **{link['title']}**")
                            st.markdown(f"  URL: {link['url']}")
                            st.markdown(f"  Last checked: {link['last_checked']}")
        else:
            st.info("No recent links found for this client.")

    # Add a button to add selected tweets and links to context
    if st.button("Add Selected Items to Context"):
        # Format selected tweets and links
        selected_tweet_data = [t for t in recent_tweets if f"{t['created_at'].strftime('%Y-%m-%d %H:%M')}: {t['text'][:50]}..." in selected_tweets]
        selected_link_data = [l for l in recent_links if f"{l['title']} (Last checked: {l['last_checked']})" in selected_links]
        
        # Format the data
        formatted_data = format_fetched_data_for_context(selected_tweet_data, selected_link_data)
        
        # Add to existing context or create new context
        if st.session_state.context_text:
            st.session_state.context_text = f"{st.session_state.context_text}\n\n{formatted_data}"
        else:
            st.session_state.context_text = formatted_data
        
        st.success("Selected items added to context!")

    # Use session state for the context text area
    context_text = st.text_area("Context Information (Newsletter)", height=150, value=st.session_state.context_text)
    topic = st.text_area("Newsletter Topic", height=70, key="newsletter_topic_input")
    
    # Add additional tailoring instructions right after context input
    st.markdown("### Additional Tailoring Instructions")
    st.markdown("Provide specific guidance on style, focus areas, or other aspects you want to emphasize in this newsletter.")
    
    additional_instructions = st.text_area(
        "Custom Instructions",
        placeholder="Examples:\n- Focus more on technical aspects\n- Emphasize community growth metrics\n- Use more concrete examples\n- Keep a neutral but optimistic tone\n- Highlight potential impact on developers",
        height=100,
        key="additional_instructions_input"
    )
    
    # More detailed options in an expander
    with st.expander("Advanced Tailoring Options"):
        # Specific focus areas with sliders
        st.markdown("### Focus Area Weights")
        st.markdown("Adjust these sliders to emphasize different aspects in the newsletter:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            technical_focus = st.slider("Technical Details", 1, 10, 5, 
                                       help="Higher values emphasize technical aspects and implementation details")
            business_focus = st.slider("Business Impact", 1, 10, 5,
                                      help="Higher values emphasize business implications and market effects")
            
        with col2:
            community_focus = st.slider("Community Aspects", 1, 10, 5,
                                       help="Higher values emphasize community engagement and adoption")
            future_focus = st.slider("Future Implications", 1, 10, 5,
                                    help="Higher values emphasize long-term vision and potential outcomes")
        
        # Combine the focus preferences into a structured format
        focus_preferences = f"""
        Focus Preferences:
        - Technical Details: {technical_focus}/10
        - Business Impact: {business_focus}/10
        - Community Aspects: {community_focus}/10
        - Future Implications: {future_focus}/10
        """
        
        # Option to include these preferences in the prompts
        include_preferences = st.checkbox("Include focus preferences in prompts", value=True)
    
    # Combine all additional instructions
    combined_instructions = ""
    
    if additional_instructions:
        combined_instructions += f"Additional Instructions:\n{additional_instructions}\n\n"
    
    if 'include_preferences' in locals() and include_preferences and any(x != 5 for x in [technical_focus, business_focus, community_focus, future_focus]):
        combined_instructions += focus_preferences
    
    if combined_instructions:
        st.success("Your tailoring instructions will be included in the generation process.")

    # Initialize session state variables
    if 'key_points_output' not in st.session_state:
        st.session_state.key_points_output = ""
    if 'edited_key_points' not in st.session_state:
        st.session_state.edited_key_points = ""
    if 'step1_completed' not in st.session_state:
        st.session_state.step1_completed = False
    if 'step2_started' not in st.session_state:
        st.session_state.step2_started = False
    if 'final_newsletter' not in st.session_state:
        st.session_state.final_newsletter = ""
    if 'newsletter_title' not in st.session_state:
        st.session_state.newsletter_title = ""
        
    # Button: Extract Key Points (Step 1)
    if st.button("Step 1: Extract Key Points + Structure for Newsletter", key="extract_key_points") and not st.session_state.step1_completed:
        if context_text:
            with st.spinner("Extracting key points..."):
                # Include additional instructions if provided
                extraction_context = context_text
                if 'combined_instructions' in locals() and combined_instructions:
                    extraction_context = f"{context_text}\n\n===ADDITIONAL INSTRUCTIONS===\n{combined_instructions}"
                
                st.session_state.key_points_output = chain_extract.run(context_text=extraction_context, long_term_doc=long_term_doc)
                st.session_state.edited_key_points = st.session_state.key_points_output
                st.session_state.step1_completed = True
                
                # Store the additional instructions in session state for later steps
                if 'combined_instructions' in locals():
                    st.session_state.additional_instructions = combined_instructions
        else:
            st.error("Please enter Context Information.")

    # Display extracted key points and allow for editing
    if st.session_state.step1_completed and not st.session_state.step2_started:
        st.markdown("### Extracted Key Points")
        st.write("Please review and edit the key points + structure below if needed:")
        st.session_state.edited_key_points = st.text_area("Edit Key Points", 
                                                        value=st.session_state.key_points_output, 
                                                        height=1000)
        
        # Button to confirm key points and continue
        if st.button("Step 2: Generate Newsletter Draft", key="generate_draft") and st.session_state.step1_completed and not st.session_state.step2_started:
            # Check if style reference is selected
            if not newsletter_example:
                st.error("Please select a Style Reference. This is required for proper formatting.")
            else:
                # Make topic optional with a default value
                if not topic:
                    st.warning("No Topic entered. Using a generic topic.")
                    topic = "Recent developments"
                
            st.session_state.step2_started = True
            
            with st.spinner("Generating newsletter draft..."):
                # Step 2a
                with st.spinner("Drafting 'What happened'..."):
                    # Get additional instructions from session state if available
                    additional_instr = st.session_state.get("additional_instructions", "")
                    
                    # Always include additional_instructions in run_params, even if empty
                    run_params = {
                        "newsletter_example": newsletter_example,
                        "key_points": st.session_state.edited_key_points,
                        "topic": topic,
                        "additional_instructions": additional_instr  # Always include this parameter
                    }
                    
                    what_happened_draft = chain_what_happened.run(**run_params)
            st.markdown("### Draft - What Happened")
            st.write(what_happened_draft)

            # Step 2b
            with st.spinner("Drafting 'Why it matters'..."):
                        # Always include additional_instructions in run_params, even if empty
                        run_params = {
                            "newsletter_example": newsletter_example,
                            "key_points": st.session_state.edited_key_points,
                            "topic": topic,
                            "additional_instructions": additional_instr  # Always include this parameter
                        }
                        
                        why_matters_draft = chain_why_matters.run(**run_params)
            st.markdown("### Draft - Why It Matters")
            st.write(why_matters_draft)

            # Step 2c (Combined Draft & Enhance 'Big Picture')
            with st.spinner("Drafting & Enhancing 'The big picture'..."):
                        # Always include additional_instructions in run_params, even if empty
                        run_params = {
                            "newsletter_example": newsletter_example,
                            "key_points": st.session_state.edited_key_points,
                            "topic": topic,
                            "long_term_doc": long_term_doc,
                            "additional_instructions": additional_instr  # Always include this parameter
                        }
                        
                        big_picture_enhanced = chain_combined_big_picture.run(**run_params)
            st.markdown("### Draft + Enhanced - The Big Picture")
            st.write(big_picture_enhanced)

            newsletter_draft = (
                f"**What happened:**\n{what_happened_draft}\n\n"
                f"**Why does it matter:**\n{why_matters_draft}\n\n"
                f"**The big picture:**\n{big_picture_enhanced}" # Use enhanced big picture directly
            )
            st.markdown("### Combining Draft Newsletter")
            

            # Step 4
            with st.spinner("Step 4: Comparing writing style..."):
                style_comparison = chain_style.run(
                    newsletter_example=newsletter_example,
                    enhanced_newsletter=newsletter_draft # Compare against the combined draft newsletter now
                )
            st.markdown("### Style Comparison Feedback")
            st.write(style_comparison)

            # Step 5
            with st.spinner("Step 5: Applying style edits..."):
                        newsletter_style_edited = chain_edit.run(
                    style_comparison=style_comparison,
                    newsletter_enhanced=newsletter_draft, # Pass the combined draft for style editing
                    newsletter_example=newsletter_example
                )
            st.markdown("### Style Edited Newsletter")
            st.write(newsletter_style_edited)
            
            # Store the final newsletter in session state for use in Tweet generator
            st.session_state.final_newsletter = newsletter_style_edited
            
            # Generate title based on the newsletter content
            with st.spinner("Generating newsletter title..."):
                # Load title examples
                title_examples = load_title_examples()
                
                # Generate title
                generated_title = chain_title_generation.run(
                    newsletter_content=newsletter_style_edited,
                    title_examples=title_examples,
                    client_name=selected_client
                )
                
                # Store and display the title
                st.session_state.newsletter_title = generated_title
                st.markdown("### Generated Newsletter Title")
                st.write(generated_title)

        else:
            st.error("Please select a Style Reference and enter a Topic.")

        # Button to restart the process
        if st.button("Start Over", key="start_over_newsletter"):
            st.session_state.key_points_output = ""
            st.session_state.edited_key_points = ""
            st.session_state.step1_completed = False
            st.session_state.step2_started = False
            st.session_state.final_newsletter = ""
            st.session_state.newsletter_title = ""
            st.experimental_rerun()

with react_agent_tab:
    st.title("React Agent Chat")
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(
            chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
        )
        st.session_state.msgs = msgs
        st.session_state.memory = memory
        st.session_state.steps = {}
        
        # Add initial message
        msgs.add_ai_message("I'm your Token Relations newsletter assistant. How can I help you today?")
    else:
        msgs = st.session_state.msgs
        memory = st.session_state.memory
    
    # Display chat messages
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    # Handle different tools differently
                    if step[0].tool == "topic_research" and isinstance(step[1], (dict, str)):
                        try:
                            # Convert to dict if it's a string
                            if isinstance(step[1], str):
                                import json
                                result = json.loads(step[1])
                            else:
                                result = step[1]
                            
                            # Research content at the top
                            st.markdown("### Research Content")
                            st.write(result.get('research_content', 'No content available'))
                            
                            # Sources as a list
                            if result.get('sources'):
                                st.markdown("### Sources")
                                for source in result.get('sources', []):
                                    st.markdown(f"- **{source.get('name', 'Unknown')}** ({source.get('type', 'Unknown')}): [{source.get('url', '#')}]({source.get('url', '#')})")
                            
                            # Metrics in sections
                            if result.get('metrics'):
                                st.markdown("### Metrics")
                                metrics = result.get('metrics', {})
                                
                                # Market metrics
                                if metrics.get('market_metrics'):
                                    st.markdown("#### Market Metrics")
                                    for k, v in metrics.get('market_metrics', {}).items():
                                        st.markdown(f"- **{k}**: {v}")
                                
                                # Ecosystem metrics
                                if metrics.get('ecosystem_metrics'):
                                    st.markdown("#### Ecosystem Metrics")
                                    for k, v in metrics.get('ecosystem_metrics', {}).items():
                                        st.markdown(f"- **{k}**: {v}")
                                
                                # Growth metrics
                                if metrics.get('growth_metrics'):
                                    st.markdown("#### Growth Metrics")
                                    for k, v in metrics.get('growth_metrics', {}).items():
                                        st.markdown(f"- **{k}**: {v}")
                        except:
                            # Fallback to standard display
                            st.write(step[1])
                    
                    # Special handling for edit_for_style tool
                    elif step[0].tool == "edit_for_style":
                        try:
                            # Try to parse as JSON if it's a string that looks like JSON
                            if isinstance(step[1], str) and step[1].strip().startswith('{'):
                                import json
                                data = json.loads(step[1])
                                
                                # If we have 'refined_content', show that directly
                                if 'refined_content' in data:
                                    st.markdown(data['refined_content'])
                                else:
                                    # Otherwise just show the raw response
                                    st.write(step[1])
                            # If it's already a dict
                            elif isinstance(step[1], dict) and 'refined_content' in step[1]:
                                st.markdown(step[1]['refined_content'])
                            else:
                                # Just display as-is if not JSON format
                                st.write(step[1])
                        except:
                            # If parsing fails, display as-is
                            st.write(step[1])
                    
                    # Standard display for other tools
                    else:
                        st.write(step[1])
            
            # Display the message content
            st.write(msg.content)
    
    # Chat input
    if prompt := st.chat_input(placeholder="Ask about blockchain developments or request newsletter help"):
        st.chat_message("user").write(prompt)
        
        # Use existing tools with the ConversationalChatAgent
        chat_agent = ConversationalChatAgent.from_llm_and_tools(
            llm=ChatAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                model="claude-3-7-sonnet-latest",
                temperature=0,
                streaming=True
            ),
            tools=tools,
            # System message with specific tool handling instructions
            system_message="""You are the newsletter editing agent for Token Relations, a firm providing ecosystem relations for blockchains and protocols.

When handling research tasks (topic_research):
1. Run the topic_research tool with the user's query
2. After receiving the results, just say "Research complete."
3. Don't analyze or summarize the research results

When handling style editing tasks (edit_for_style):
1. Run the edit_for_style tool with the user's content
2. After receiving results, just say "Editing complete."
3. Don't provide additional commentary or analysis

When handling newsletter evaluation (evaluate_newsletter):
1. Run the evaluate_newsletter tool with the user's content
2. After receiving results, just say "Evaluation complete."
3. The results will be displayed directly to the user

For other tasks:
- Provide helpful, concise responses
- Maintain our distinctive factual, clear, and substantive voice

Remember, keep your final answers brief - the tool outputs will be displayed automatically."""
        )
        
        # Create executor with the Conversational agent
        executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            memory=memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=5  # Keep it low to prevent looping
        )
        
        # Execute and display results
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            cfg = RunnableConfig()
            cfg["callbacks"] = [st_cb]
            
            response = executor.invoke(prompt, cfg)
            
            # Handle different tool outputs differently
            has_tool_output = False
            for step in response["intermediate_steps"]:
                tool_name = step[0].tool if hasattr(step[0], 'tool') else ""
                if tool_name in ["topic_research", "edit_for_style", "evaluate_newsletter"]:
                    has_tool_output = True
                    break
            
            if has_tool_output:
                # For tool-based tasks, use a simple completion message
                if "topic_research" in [s[0].tool for s in response["intermediate_steps"]]:
                    st.write("Research complete.")
                elif "edit_for_style" in [s[0].tool for s in response["intermediate_steps"]]:
                    st.write("Editing complete.")
                elif "evaluate_newsletter" in [s[0].tool for s in response["intermediate_steps"]]:
                    st.write("Evaluation complete.")
            else:
                # For other types of requests, show the normal response
                st.write(response["output"])
            
            # Store steps for future rendering
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
    
    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.msgs.clear()
        st.session_state.msgs.add_ai_message("I'm your Token Relations newsletter assistant. How can I help you today?")
        st.session_state.steps = {}
        st.experimental_rerun()
    
    # Add example prompts
    with st.expander("Example Prompts"):
        st.markdown("""
        Try asking:
        - "Research the latest developments in Layer 2 scaling solutions"
        - "Help me draft a newsletter about the recent DeFi protocol upgrade"
        - "Edit this newsletter section to match our style: [paste content]"
        - "Evaluate this newsletter draft for quality: [paste content]"
        """)

with bullet_points_tab:
    st.title("Bullet Point Generator")
    
    # Create tabs for ecosystem and community bullet points
    ecosystem_tab, community_tab = st.tabs(["Ecosystem Bullet Points", "Community Bullet Points"])
    
    # Initialize session states for bullet point contexts if they don't exist
    if 'ecosystem_context' not in st.session_state:
        st.session_state.ecosystem_context = ""
    if 'community_context' not in st.session_state:
        st.session_state.community_context = ""
    
    with ecosystem_tab:
        st.markdown("### Ecosystem Bullet Points")
        st.markdown("Generate bullet points related to technical developments, partnerships, protocol upgrades, and ecosystem growth.")
        
        # Add tweet and link selection section
        st.markdown("### Recent Tweets and Links")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Recent Tweets")
            # Fetch recent tweets for the selected client
            recent_tweets_eco = fetch_recent_tweets(st.session_state.get("supabase_client"), [selected_client], limit_per_fetch=10)
            
            if recent_tweets_eco:
                tweet_options = [f"{tweet['created_at'].strftime('%Y-%m-%d %H:%M')}: {tweet['text'][:50]}..." for tweet in recent_tweets_eco]
                selected_tweets_eco = st.multiselect(
                    "Select tweets to include",
                    options=tweet_options,
                    key="ecosystem_tweet_selector"
                )
                
                if selected_tweets_eco:
                    with st.expander("Selected Tweets"):
                        for tweet_option in selected_tweets_eco:
                            tweet = next((t for t in recent_tweets_eco if tweet_option.startswith(t['created_at'].strftime('%Y-%m-%d %H:%M'))), None)
                            if tweet:
                                st.markdown(f"- **{tweet['created_at'].strftime('%Y-%m-%d %H:%M')}**: {tweet['text']}")
                                st.markdown(f"  [Link]({tweet['url']})")
            else:
                st.info("No recent tweets found for this client.")
        
        with col2:
            st.markdown("#### Recent Links")
            recent_links_eco = fetch_client_links(st.session_state.get("supabase_client"), selected_client, limit=10)
            
            if recent_links_eco:
                link_options = [f"{link['title']} (Last checked: {link['last_checked']})" for link in recent_links_eco]
                selected_links_eco = st.multiselect(
                    "Select links to include",
                    options=link_options,
                    key="ecosystem_link_selector"
                )
                
                if selected_links_eco:
                    with st.expander("Selected Links"):
                        for link_option in selected_links_eco:
                            link = next((l for l in recent_links_eco if link_option.startswith(l['title'])), None)
                            if link:
                                st.markdown(f"- **{link['title']}**")
                                st.markdown(f"  URL: {link['url']}")
                                st.markdown(f"  Last checked: {link['last_checked']}")
            else:
                st.info("No recent links found for this client.")
        
        # Add button to add selected items to ecosystem context
        if st.button("Add Selected Items to Ecosystem Context"):
            selected_tweet_data = [t for t in recent_tweets_eco if f"{t['created_at'].strftime('%Y-%m-%d %H:%M')}: {t['text'][:50]}..." in selected_tweets_eco]
            selected_link_data = [l for l in recent_links_eco if f"{l['title']} (Last checked: {l['last_checked']})" in selected_links_eco]
            
            formatted_data = format_fetched_data_for_context(selected_tweet_data, selected_link_data)
            
            if st.session_state.ecosystem_context:
                st.session_state.ecosystem_context = f"{st.session_state.ecosystem_context}\n\n{formatted_data}"
            else:
                st.session_state.ecosystem_context = formatted_data
            
            st.success("Selected items added to ecosystem context!")
        
        # Load ecosystem examples from global file but don't display them to the user
        default_ecosystem_examples = load_bullet_point_examples("Ecosystem")
        
        ecosystem_context = st.text_area(
            "Ecosystem Context Information", 
            height=150,
            value=st.session_state.ecosystem_context
        )
        
        if st.button("Generate Ecosystem Bullet Points", key="gen_ecosystem"):
            if ecosystem_context:
                with st.spinner("Generating ecosystem bullet points..."):
                    ecosystem_bullets = chain_ecosystem_bullet_point_prompt.run(
                        context_text=ecosystem_context,
                        example_bullet_points=default_ecosystem_examples
                    )
                st.markdown("#### Generated Ecosystem Bullet Points")
                st.write(ecosystem_bullets)
                
                if st.button("Copy to Clipboard", key="copy_ecosystem"):
                    st.success("Ecosystem bullet points copied to clipboard!")
            else:
                st.error("Please fill in the Ecosystem Context.")
    
    with community_tab:
        st.markdown("### Community Bullet Points")
        st.markdown("Generate bullet points related to community engagement, events, social metrics, and user adoption.")
        
        # Add tweet and link selection section for community
        st.markdown("### Recent Tweets and Links")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Recent Tweets")
            recent_tweets_comm = fetch_recent_tweets(st.session_state.get("supabase_client"), [selected_client], limit_per_fetch=10)
            
            if recent_tweets_comm:
                tweet_options = [f"{tweet['created_at'].strftime('%Y-%m-%d %H:%M')}: {tweet['text'][:50]}..." for tweet in recent_tweets_comm]
                selected_tweets_comm = st.multiselect(
                    "Select tweets to include",
                    options=tweet_options,
                    key="community_tweet_selector"
                )
                
                if selected_tweets_comm:
                    with st.expander("Selected Tweets"):
                        for tweet_option in selected_tweets_comm:
                            tweet = next((t for t in recent_tweets_comm if tweet_option.startswith(t['created_at'].strftime('%Y-%m-%d %H:%M'))), None)
                            if tweet:
                                st.markdown(f"- **{tweet['created_at'].strftime('%Y-%m-%d %H:%M')}**: {tweet['text']}")
                                st.markdown(f"  [Link]({tweet['url']})")
            else:
                st.info("No recent tweets found for this client.")
        
        with col2:
            st.markdown("#### Recent Links")
            recent_links_comm = fetch_client_links(st.session_state.get("supabase_client"), selected_client, limit=10)
            
            if recent_links_comm:
                link_options = [f"{link['title']} (Last checked: {link['last_checked']})" for link in recent_links_comm]
                selected_links_comm = st.multiselect(
                    "Select links to include",
                    options=link_options,
                    key="community_link_selector"
                )
                
                if selected_links_comm:
                    with st.expander("Selected Links"):
                        for link_option in selected_links_comm:
                            link = next((l for l in recent_links_comm if link_option.startswith(l['title'])), None)
                            if link:
                                st.markdown(f"- **{link['title']}**")
                                st.markdown(f"  URL: {link['url']}")
                                st.markdown(f"  Last checked: {link['last_checked']}")
            else:
                st.info("No recent links found for this client.")
        
        # Add button to add selected items to community context
        if st.button("Add Selected Items to Community Context"):
            selected_tweet_data = [t for t in recent_tweets_comm if f"{t['created_at'].strftime('%Y-%m-%d %H:%M')}: {t['text'][:50]}..." in selected_tweets_comm]
            selected_link_data = [l for l in recent_links_comm if f"{l['title']} (Last checked: {l['last_checked']})" in selected_links_comm]
            
            formatted_data = format_fetched_data_for_context(selected_tweet_data, selected_link_data)
            
            if st.session_state.community_context:
                st.session_state.community_context = f"{st.session_state.community_context}\n\n{formatted_data}"
            else:
                st.session_state.community_context = formatted_data
            
            st.success("Selected items added to community context!")
        
        # Load community examples from global file but don't display them to the user
        default_community_examples = load_bullet_point_examples("Community")
        
        community_context = st.text_area(
            "Community Context Information", 
            height=150,
            value=st.session_state.community_context
        )
        
        if st.button("Generate Community Bullet Points", key="gen_community"):
            if community_context:
                with st.spinner("Generating community bullet points..."):
                    community_bullets = chain_community_bullet_point_prompt.run(
                        context_text=community_context,
                        example_bullet_points=default_community_examples
                    )
                st.markdown("#### Generated Community Bullet Points")
                st.write(community_bullets)
                
                if st.button("Copy to Clipboard", key="copy_community"):
                    st.success("Community bullet points copied to clipboard!")
            else:
                st.error("Please fill in the Community Context.")

# New tab for Tweet Generator
with tweet_tab:
    st.title("Tweet Generator")
    st.markdown("Generate tweets based on newsletter content or custom context information.")
    
    # Client selection for tweet
    selected_tweet_client = st.selectbox("Select a Client (Tweet)", list(CLIENT_FILES.keys()), key="tweet_client")
    
    # Tweet format selection
    tweet_format = st.selectbox("Select Tweet Format", ["Short", "Medium", "Long"], key="tweet_format")
    
    # Load tweet examples (backend only, not shown to user)
    tweet_examples = load_tweet_examples(tweet_format)
    
    # Option to use generated newsletter or custom input
    use_newsletter = st.checkbox("Use Generated Newsletter Content", value=True)
    
    newsletter_content = ""
    if use_newsletter:
        if st.session_state.final_newsletter:
            newsletter_content = st.session_state.final_newsletter
            st.success("Using the generated newsletter content")
        else:
            st.warning("No newsletter has been generated yet. Please generate a newsletter first or uncheck to enter custom content.")
    
    # Custom content input if not using newsletter
    if not use_newsletter or not newsletter_content:
        newsletter_content = st.text_area("Enter Custom Content for Tweet", height=200)
    
    # Generate tweet button
    if st.button("Generate Tweet", key="gen_tweet"):
        if newsletter_content and selected_tweet_client:
            with st.spinner(f"Generating {tweet_format} tweet..."):
                generated_tweet = chain_tweet_generation.run(
                    newsletter_content=newsletter_content,
                    tweet_format=tweet_format,
                    tweet_examples=tweet_examples,
                    client_name=selected_tweet_client
                )
            st.markdown("### Generated Tweet")
            st.write(generated_tweet)
            
            # Show character count
            char_count = len(generated_tweet)
            st.info(f"Character count: {char_count}/280")
            
            # Copy button
            if st.button("Copy Tweet to Clipboard", key="copy_tweet"):
                st.success("Tweet copied to clipboard!")
        else:
            st.error("Please provide content and select a client for the tweet.")

# New tab for Title Generator
with title_tab:
    st.title("Title Generator")
    st.markdown("Generate a title for your newsletter based on its content.")
    
    # Client selection for title
    selected_title_client = st.selectbox("Select a Client (Title)", list(CLIENT_FILES.keys()), key="title_client")
    
    # Option to use generated newsletter or custom input
    use_newsletter_for_title = st.checkbox("Use Generated Newsletter Content", value=True, key="use_newsletter_title")
    
    title_content = ""
    if use_newsletter_for_title:
        if st.session_state.final_newsletter:
            title_content = st.session_state.final_newsletter
            st.success("Using the generated newsletter content")
        else:
            st.warning("No newsletter has been generated yet. Please generate a newsletter first or uncheck to enter custom content.")
    
    # Custom content input if not using newsletter
    if not use_newsletter_for_title or not title_content:
        title_content = st.text_area("Enter Newsletter Content for Title Generation", height=300)
    
    # Load title examples
    title_examples = load_title_examples()
    
    # Option to view and edit title examples
    with st.expander("View/Edit Title Examples"):
        edited_title_examples = st.text_area(
            "Title Examples (for reference)",
            value=title_examples,
            height=200
        )
        if edited_title_examples != title_examples and st.button("Update Title Examples"):
            title_examples = edited_title_examples
    
    # Generate title button
    if st.button("Generate Title", key="gen_title"):
        if title_content and selected_title_client:
            with st.spinner("Generating title..."):
                generated_title = chain_title_generation.run(
                    newsletter_content=title_content,
                    title_examples=title_examples,
                    client_name=selected_title_client
                )
            st.markdown("### Generated Title")
            st.write(generated_title)
            
            # Show character count
            char_count = len(generated_title)
            st.info(f"Character count: {char_count}")
            
            # Copy button
            if st.button("Copy Title to Clipboard", key="copy_title"):
                st.success("Title copied to clipboard!")
                
            # Option to save this title to the session state
            if st.button("Use This Title for Newsletter", key="use_title"):
                st.session_state.newsletter_title = generated_title
                st.success("Title saved and will be used for the newsletter!")
        else:
            st.error("Please provide newsletter content and select a client for the title.")
    
    # Generate multiple titles option
    st.markdown("---")
    st.markdown("### Generate Multiple Title Options")
    num_titles = st.slider("Number of title options to generate", min_value=2, max_value=5, value=3)
    
    if st.button("Generate Multiple Titles", key="gen_multiple_titles"):
        if title_content and selected_title_client:
            with st.spinner(f"Generating {num_titles} title options..."):
                # Create a custom prompt for multiple titles
                multiple_titles_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "Newsletter Title Writer: Create multiple concise, engaging titles for web3 newsletters that match example style. Provide distinct options with different angles or emphases. Focus on clarity and accuracy."
                    ),
                    (
                        "human",
                        f"Newsletter Content:\n{title_content}\n\n"
                        f"Example Titles (carefully emulate this style):\n{title_examples}\n\n"
                        f"Client Name: {selected_title_client}\n\n"
                        f"Please generate {num_titles} different title options for this newsletter that capture the key information and match the style of the examples. Number each option."
                    )
                ])
                
                # Create a temporary chain for multiple titles
                chain_multiple_titles = LLMChain(llm=anthropic_llm, prompt=multiple_titles_prompt)
                
                # Generate multiple titles
                multiple_titles = chain_multiple_titles.run(
                    newsletter_content=title_content,
                    title_examples=title_examples,
                    client_name=selected_title_client,
                    num_titles=num_titles
                )
            
            st.markdown("### Generated Title Options")
            st.write(multiple_titles)
            
            # Option to select one of the titles
            st.markdown("If you like one of these titles, copy it and use the 'Enter Custom Title' option below.")
            
            # Custom title input
            custom_title = st.text_input("Enter Custom Title", key="custom_title_input")
            if custom_title and st.button("Use Custom Title", key="use_custom_title"):
                st.session_state.newsletter_title = custom_title
                st.success("Custom title saved and will be used for the newsletter!")
        else:
            st.error("Please provide newsletter content and select a client for the titles.")

# New tab for image generation
with image_tab:
    st.title("Newsletter Cover Image Generator")
    st.markdown("Generate a cover image for your newsletter based on its content.")
    
    # Client selection for image
    selected_image_client = st.selectbox("Select a Client", list(CLIENT_FILES.keys()), key="image_client")
    
    # Option to use generated newsletter or custom input
    use_newsletter_for_image = st.checkbox("Use Generated Newsletter Content", value=True, key="use_newsletter_image")
    
    # Initialize session state variables if they don't exist
    if "image_content" not in st.session_state:
        st.session_state.image_content = ""
    if "image_title" not in st.session_state:
        st.session_state.image_title = ""
    if "generated_image_prompt" not in st.session_state:
        st.session_state.generated_image_prompt = ""
    if "generated_image_url" not in st.session_state:
        st.session_state.generated_image_url = ""
    
    # Get content based on checkbox selection
    if use_newsletter_for_image:
        if st.session_state.final_newsletter:
            st.session_state.image_content = st.session_state.final_newsletter
            st.session_state.image_title = st.session_state.newsletter_title if st.session_state.newsletter_title else "Newsletter"
            st.success("Using the generated newsletter content and title")
        else:
            st.warning("No newsletter has been generated yet. Please generate a newsletter first or uncheck to enter custom content.")
    
    # Custom content input if not using newsletter or if no newsletter is available
    if not use_newsletter_for_image or not st.session_state.image_content:
        content_input = st.text_area(
            "Enter Newsletter Content for Image Generation", 
            value=st.session_state.image_content,
            height=200
        )
        title_input = st.text_input(
            "Enter Newsletter Title", 
            value=st.session_state.image_title if st.session_state.image_title else "Newsletter"
        )
        
        # Update session state with user input
        st.session_state.image_content = content_input
        st.session_state.image_title = title_input
    
    # Image style options
    st.markdown("### Image Style Options")
    image_size = st.selectbox(
        "Image Size", 
        ["1024x1024", "1024x1792", "1792x1024", "256x256", "512x512"],
        index=0,
        key="image_size_select"
    )
    
    image_quality = st.selectbox(
        "Image Quality", 
        ["standard", "hd"],
        index=0,
        key="image_quality_select"
    )
    
    # Add the generate_image_dalle function
    def generate_image_dalle(prompt, api_key, size="1024x1024", quality="standard"):
        try:
            from openai import OpenAI
            
            # Initialize the client with the API key
            client = OpenAI(api_key=api_key)
            
            # Generate the image
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=size,
                quality=quality
            )
            
            # Extract the image URL from the response
            image_url = response.data[0].url
            return image_url, None
        except ImportError:
            return None, "OpenAI Python package not installed. Please install it with 'pip install openai'."
        except Exception as e:
            return None, f"Error generating image: {str(e)}"
    
    # In the image_tab section, update the image generation part:

    # Add an option for custom prompt
    use_custom_prompt = st.checkbox("Use custom image prompt", value=False, key="use_custom_prompt")

    # Initialize content variables
    has_content = False
    newsletter_content = ""
    newsletter_title = ""

    if use_custom_prompt:
        # Let the user write their own prompt
        custom_prompt = st.text_area(
            "Custom image prompt", 
            placeholder="A minimalist abstract representation of blockchain technology with blue and purple gradients...",
            height=100,
            key="custom_image_prompt"
        )
        # Store in session state
        if custom_prompt:
            st.session_state.generated_image_prompt = custom_prompt
    else:
        # Better newsletter content extraction
        # Check if we have content and title
        if use_newsletter_for_image and st.session_state.final_newsletter:
            newsletter_content = st.session_state.final_newsletter
            newsletter_title = st.session_state.newsletter_title if st.session_state.newsletter_title else "Newsletter"
            has_content = True
        elif 'image_content' in st.session_state and st.session_state.image_content:
            newsletter_content = st.session_state.image_content
            newsletter_title = st.session_state.image_title if 'image_title' in st.session_state and st.session_state.image_title else "Newsletter"
            has_content = True
        else:
            content_input = st.text_area(
                "Enter Newsletter Content for Image Generation", 
                height=200,
                key="new_image_content_input"
            )
            title_input = st.text_input(
                "Enter Newsletter Title", 
                value="Newsletter",
                key="new_image_title_input"
            )
            
            if content_input:
                newsletter_content = content_input
                newsletter_title = title_input
                has_content = True
                # Store in session state
                st.session_state.image_content = content_input
                st.session_state.image_title = title_input

    # One-shot generate button
    if st.button("Generate Cover Image", key="gen_cover_image"):
        if (has_content or use_custom_prompt) and selected_image_client:
            # Step 1: Get or generate the image prompt
            with st.spinner("Preparing image prompt..."):
                if use_custom_prompt and 'generated_image_prompt' in st.session_state and st.session_state.generated_image_prompt:
                    # Just use the custom prompt directly
                    generated_prompt = st.session_state.generated_image_prompt
                else:
                    # Generate prompt based on content
                    # Use the selected model from session state, or fall back to anthropic_llm
                    llm = st.session_state.get("llm", anthropic_llm)
                    
                    # Create a more focused prompt template for image generation
                    image_generation_prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            "Create professional image prompts for financial/tech publications. Analyze the newsletter content and title to extract the main themes and concepts. Focus on creating a visually striking, abstract representation with 3-5 key elements that capture the essence of the content. Your prompt should be 15-25 words only, avoiding any text elements in the image itself. Prefer abstract representations, data visualizations, or symbolic elements with clean professional aesthetics."
                        ),
                        (
                            "human",
                            """Newsletter Title: {newsletter_title}
                            
                            Newsletter Content: 
                            {newsletter_content}
                            
                            Client: {client_name}
                            
                            Create a focused, professional image prompt (15-25 words) capturing the key themes from this content. The image should be suitable for a financial/blockchain newsletter cover."""
                        )
                    ])
                    
                    chain_image_generation = LLMChain(llm=llm, prompt=image_generation_prompt)
                    
                    # Extract key sections if possible to avoid token limits
                    if len(newsletter_content) > 2000:  # If content is too long
                        # Try to extract just the "What happened" and "Big picture" sections
                        import re
                        what_happened = re.search(r'What happened:.*?(?=Why does it matter:|$)', newsletter_content, re.DOTALL)
                        big_picture = re.search(r'The big picture:.*?(?=$)', newsletter_content, re.DOTALL)
                        
                        extracted_content = ""
                        if what_happened:
                            extracted_content += what_happened.group(0) + "\n\n"
                        if big_picture:
                            extracted_content += big_picture.group(0)
                        
                        if extracted_content:
                            newsletter_content = extracted_content
                        else:
                            # If extraction failed, just take the first 1000 chars
                            newsletter_content = newsletter_content[:1000] + "..."
                    
                        # Generate the prompt
                        generated_prompt = chain_image_generation.run(
                            newsletter_content=newsletter_content,
                            newsletter_title=newsletter_title,
                            client_name=selected_image_client
                        )
                        
                        # Clean up the generated prompt
                        # Remove any formatting or quotation marks that might be included
                        generated_prompt = generated_prompt.strip().strip('"\'')
                        
                        # Store the generated prompt in session state
                        st.session_state.generated_image_prompt = generated_prompt
            
            # Display the generated prompt
            st.markdown("### Image Prompt")
            st.write(st.session_state.generated_image_prompt)
            
            # Step 2: Generate the image using OpenAI API
            with st.spinner("Generating image with GPT-Image-1..."):
                try:
                    # Get OpenAI API key from environment or session state
                    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
                    
                    # If not in environment, try to get from session state
                    if not openai_api_key:
                        openai_api_key = st.session_state.get("OPENAI_API_KEY", "")
                    
                    # Debug: Show a masked version of the key to verify it's being found
                    if openai_api_key:
                        masked_key = openai_api_key[:8] + "..." + openai_api_key[-4:] if len(openai_api_key) > 12 else "***"
                        st.write(f"Using API key: {masked_key}")
                    else:
                        st.error("OpenAI API key not found in environment or session state.")
                        st.stop()
                    
                    # Generate the image
                    image_data, error = generate_image_dalle(
                        prompt=st.session_state.generated_image_prompt,
                        api_key=openai_api_key,
                        size=image_size,
                        quality=image_quality
                    )
                    
                    if image_data:
                        # Store the image data in session state
                        st.session_state.generated_image_data = image_data
                        
                        # Display the image directly
                        st.markdown("### Generated Image")
                        st.image(image_data, caption="Generated cover image")
                        
                        # Add download button
                        btn = st.download_button(
                            label="Download Image",
                            data=image_data,
                            file_name=f"newsletter_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                        
                        # Image analysis option
                        with st.expander("Image Analysis"):
                            st.markdown("### Image Analysis")
                            if st.button("Analyze Image", key="analyze_image_button"):
                                with st.spinner("Analyzing image with Claude Vision..."):
                                    # For Claude Vision we need a URL, so this would need modification
                                    # to either upload the image or convert it to data URL
                                    st.info("Image analysis with local images requires uploading first. Feature coming soon.")
                    else:
                        st.error(f"Failed to generate image: {error}")
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")
                    st.info("Make sure your OpenAI API key is set correctly and has access to GPT-Image-1.")
        else:
            st.error("Please provide newsletter content or a custom prompt, and select a client.")
    
    # Display previously generated image if available
    elif "generated_image_url" in st.session_state and st.session_state.generated_image_url:
        st.markdown("### Previously Generated Image")
        st.image(st.session_state.generated_image_url, caption="Generated cover image")
        
        if "generated_image_prompt" in st.session_state and st.session_state.generated_image_prompt:
            with st.expander("View Image Prompt"):
                st.write(st.session_state.generated_image_prompt)

# New tab for graph plotting
with graph_tab:
    st.title("Animated and Static Graph Plotter")
    st.markdown("Generate custom graphs from CSV data with animation support.")
    
    # Add custom CSS for the graph tab
    st.markdown("""
    <style>
        .main-title {
            font-size: 42px !important;
            font-weight: bold;
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }
        .section-header {
            font-size: 24px !important;
            font-weight: bold;
            margin-top: 10px !important;
            margin-bottom: 5px !important;
            padding-top: 10px !important;
            padding-bottom: 0px !important;
        }
        .subsection-header {
            font-size: 20px !important;
            font-weight: bold;
            margin-top: 5px !important;
            margin-bottom: 0px !important;
            padding-top: 5px !important;
            padding-bottom: 0px !important;
        }
        /* Reduce spacing between elements */
        .stSlider, .stCheckbox, .stRadio, .stSelectbox {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add the format_y_tick function
    def format_y_tick(value, pos, use_dollar=False):
        """Format y-axis ticks with K, M, B suffixes and optional dollar sign."""
        if value == 0:
            return '$0' if use_dollar else '0'
        magnitude = 0
        while abs(value) >= 1000:
            magnitude += 1
            value /= 1000.0
        prefix = '$' if use_dollar else ''
        return f'{prefix}{value:.1f}{["", "K", "M", "B", "T"][magnitude]}'
    
    # Add the save_frames_as_gif function
    def save_frames_as_gif(fig, frames, speed, width=1200, height=800):
        images = []
        try:
            for frame in frames:
                fig.update(data=frame.data)
                img_bytes = fig.to_image(format="png", width=width, height=height, scale=1.0)
                img_array = imageio.imread(io.BytesIO(img_bytes))
                images.append(img_array)
            
            gif_buffer = io.BytesIO()
            imageio.mimsave(gif_buffer, images, format='gif', duration=speed, loop=0)
            gif_buffer.seek(0)
            return gif_buffer
        except Exception as e:
            st.error(f"Error creating GIF: {str(e)}")
            return None
    
    # Add the normalize_data function
    def normalize_data(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    
    # Modified file upload section to handle multiple CSVs
    num_files = st.number_input("Number of CSV files to upload", min_value=1, max_value=10, value=1)
    uploaded_files = []
    dataframes = []
    file_names = []
    
    # First, collect all files
    for i in range(num_files):
        file = st.file_uploader(f"Upload CSV #{i+1}", type=["csv"], key=f"graph_file_{i}")
        if file:
            uploaded_files.append(file)
            try:
                # Read with pandas explicitly specifying encoding
                df = pd.read_csv(file, encoding='utf-8-sig')
                
                # Debug information
                st.write(f"File {i+1} columns: {df.columns.tolist()}")
                st.write(f"First few rows:")
                st.write(df.head())
                
                # Store in dataframes list
                dataframes.append(df)
                file_names.append(file.name)
            except Exception as e:
                st.error(f"Error reading file #{i+1}: {str(e)}")
    
    # Add logo and theme options
    logo_file = st.file_uploader("Upload a Logo (Optional, PNG/JPG)", type=["png", "jpg", "jpeg"], key="graph_logo")
    theme = st.radio("Select Theme", ["dark", "light"], index=0, key="graph_theme")
    
    # Add text formatting options
    st.markdown('<p class="section-header">Text Formatting</p>', unsafe_allow_html=True)
    title_size = st.slider("Title Font Size", 10, 36, 18, 1, key="graph_title_size")
    axis_label_size = st.slider("Axis Label Font Size", 8, 24, 14, 1, key="graph_axis_label_size")
    tick_label_size = st.slider("Tick Label Font Size", 8, 20, 12, 1, key="graph_tick_label_size")
    legend_font_size = st.slider("Legend Font Size", 8, 20, 10, 1, key="graph_legend_font_size")
    
    # Add animation and logo options
    animation_speed = st.slider("Select Animation Speed (seconds per frame)", 0.05, 1.0, 0.1, 0.05, key="graph_animation_speed")
    logo_x = st.slider("Logo X Position", 0.0, 1.0, 0.95, 0.01, key="graph_logo_x")
    logo_y = st.slider("Logo Y Position", 0.0, 1.0, 0.05, 0.01, key="graph_logo_y")
    
    # Add legend options
    show_legend = st.checkbox("Show Legend", value=True, key="graph_show_legend")
    legend_x = st.slider("Legend X Position", 0.0, 1.0, 0.9, 0.01, key="graph_legend_x")
    legend_y = st.slider("Legend Y Position", 0.0, 1.0, 0.9, 0.01, key="graph_legend_y")
    
    # Add title and subtitle options
    st.markdown('<p class="section-header">Title and Subtitle</p>', unsafe_allow_html=True)
    custom_title = st.text_input("Graph Title", "Blockchain Comparison", key="graph_custom_title")
    custom_subtitle = st.text_input("Graph Subtitle (Optional)", "", key="graph_custom_subtitle")
    subtitle_size = st.slider("Subtitle Font Size", 8, 24, 14, 1, key="graph_subtitle_size")
    
    # Add graph type selection
    graph_type = st.radio("Select Graph Type", ["Line", "Bar"], index=0, key="graph_type")
    
    # Now process the dataframes if we have any
    if len(dataframes) > 0:
        try:
            # Create a container for date column selection
            date_cols = {}
            st.markdown('<p class="subsection-header">Date Column Selection</p>', unsafe_allow_html=True)
            
            # Process each dataframe
            for i, df in enumerate(dataframes):
                date_cols[i] = st.selectbox(f"Select date column for {file_names[i]}", 
                                          df.columns, key=f"graph_date_col_{i}")
                
                # Convert date column to datetime
                dataframes[i][date_cols[i]] = pd.to_datetime(dataframes[i][date_cols[i]], errors="coerce")
                
                # Convert numeric columns after date column is selected
                for col in df.columns:
                    # Skip date column
                    if col != date_cols[i]:
                        # Try to convert to numeric, replacing commas
                        try:
                            # First remove commas and convert to numeric
                            dataframes[i][col] = dataframes[i][col].astype(str).str.replace(',', '').astype(float)
                            st.success(f"Converted column {col} to numeric")
                        except Exception as e:
                            # If conversion fails, it might not be a numeric column
                            pass
                
                # Sort the dataframe by the selected date column
                dataframes[i] = dataframes[i].sort_values(by=date_cols[i])
                st.success(f"Data in {file_names[i]} sorted by {date_cols[i]}")
                
                # Show the date range
                min_date = dataframes[i][date_cols[i]].min()
                max_date = dataframes[i][date_cols[i]].max()
                st.info(f"Date range: {min_date} to {max_date}")
                
                # After showing the date range, add date range filter
                if st.checkbox(f"Filter date range for {file_names[i]}", key=f"graph_date_filter_{i}"):
                    # Create date range selector
                    date_range = st.date_input(
                        f"Select date range for {file_names[i]}",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        key=f"graph_date_range_{i}"
                    )
                    
                    # Apply date filter if a range is selected
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        dataframes[i] = dataframes[i][
                            (dataframes[i][date_cols[i]].dt.date >= start_date) & 
                            (dataframes[i][date_cols[i]].dt.date <= end_date)
                        ]
                        st.success(f"Applied date filter: {start_date} to {end_date}")
            
            # Add segmentation options
            segment_data = {}
            segment_columns = {}
            segment_values = {}
            
            st.markdown('<p class="subsection-header">Data Segmentation</p>', unsafe_allow_html=True)
            for i, df in enumerate(dataframes):
                if st.checkbox(f"Segment data in {file_names[i]}", key=f"graph_segment_{i}"):
                    # Get string columns (object type)
                    string_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if string_cols:
                        # Let user select which column to use for segmentation
                        segment_columns[i] = st.selectbox(
                            f"Select column to segment by in {file_names[i]}", 
                            string_cols,
                            key=f"graph_segment_col_{i}"
                        )
                        
                        # Get unique values in the selected column
                        unique_values = df[segment_columns[i]].unique().tolist()
                        
                        # Let user select which values to include
                        segment_values[i] = st.multiselect(
                            f"Select values to include from {segment_columns[i]} in {file_names[i]}", 
                            unique_values,
                            default=unique_values[:min(3, len(unique_values))],  # Default to first 3 values
                            key=f"graph_segment_values_{i}"
                        )
                        
                        # Create segmented dataframes
                        segment_data[i] = {}
                        for value in segment_values[i]:
                            segment_data[i][value] = df[df[segment_columns[i]] == value].copy()
                            
                        # Display segmented data in tabs
                        segment_tabs = st.tabs([f"{value}" for value in segment_values[i]])
                        for j, (value, tab) in enumerate(zip(segment_values[i], segment_tabs)):
                            with tab:
                                st.write(f"Data for {segment_columns[i]} = {value}")
                                st.dataframe(segment_data[i][value])
                    else:
                        st.warning(f"No string columns found in {file_names[i]} for segmentation")
            
            # Create sliders for row ranges for each dataframe
            row_ranges = {}
            st.markdown('<p class="subsection-header">Row Range Selection</p>', unsafe_allow_html=True)
            for i, df in enumerate(dataframes):
                row_ranges[i] = st.slider(f"Row range for {file_names[i]}", 
                                         0, len(df) - 1, 
                                         (0, min(50, len(df) - 1)), 
                                         key=f"graph_row_range_{i}")
            
            # Create sub-dataframes based on selected row ranges
            sub_dfs = {}
            for i, df in enumerate(dataframes):
                rs, re = row_ranges[i]
                # If segmentation is active for this dataframe, use the segmented data
                if i in segment_data:
                    # Create a combined dataframe with all selected segments
                    combined_segments = pd.DataFrame()
                    for value in segment_values[i]:
                        segment_df = segment_data[i][value]
                        # Apply row range to segmented data
                        segment_df = segment_df.iloc[min(rs, len(segment_df)-1):min(re+1, len(segment_df))]
                        # Add a new column to identify the segment
                        segment_df = segment_df.copy()
                        segment_df[f'_segment_{segment_columns[i]}'] = value
                        combined_segments = pd.concat([combined_segments, segment_df])
                    
                    sub_dfs[i] = combined_segments
                else:
                    # Use the original dataframe with row range
                    sub_dfs[i] = df.iloc[rs:re + 1].copy()
            
            # Display the sub-dataframes in tabs
            tabs = st.tabs([f"Data {i+1}: {name}" for i, name in enumerate(file_names)])
            for i, tab in enumerate(tabs):
                with tab:
                    st.dataframe(sub_dfs[i])
            
            # Column selection for each dataframe
            st.markdown('<p class="subsection-header">Column Selection</p>', unsafe_allow_html=True)
            y_cols_by_file = {}
            for i, df in enumerate(sub_dfs.values()):
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                y_cols_by_file[i] = st.multiselect(f"Select numeric columns to plot from {file_names[i]}", 
                                                  numeric_cols, 
                                                  key=f"graph_y_cols_{i}")
            
            # Check if any columns are selected
            any_cols_selected = any(len(cols) > 0 for cols in y_cols_by_file.values())
            
            if any_cols_selected:
                normalize = st.checkbox("Normalize Values", value=False, key="graph_normalize")
                format_numbers = st.checkbox("Format Y-axis (1K, 1M, 1B)", value=False, key="graph_format_numbers")
                use_dollar = st.checkbox("Add Dollar Sign ($) to Y-axis", value=False, key="graph_use_dollar")
                
                # Color picker for each selected column in each file
                colors = {}
                st.markdown('<p class="section-header">Color Selection</p>', unsafe_allow_html=True)
                for file_idx, cols in y_cols_by_file.items():
                    for col in cols:
                        # If segmentation is active, create color pickers for each segment
                        if file_idx in segment_data:
                            for value in segment_values[file_idx]:
                                color_key = f"{file_idx}_{col}_{value}"
                                colors[color_key] = st.color_picker(
                                    f"Pick color for {file_names[file_idx]} - {col} - {value}", 
                                    "#" + ''.join([hex(hash(color_key + str(i)) % 256)[2:].zfill(2) for i in range(3)]),
                                    key=f"graph_color_{color_key}"
                                )
                        else:
                            color_key = f"{file_idx}_{col}"
                            colors[color_key] = st.color_picker(
                                f"Pick color for {file_names[file_idx]} - {col}", 
                                "#" + ''.join([hex(hash(color_key + str(i)) % 256)[2:].zfill(2) for i in range(3)]),
                                key=f"graph_color_{color_key}"
                            )
                
                # Custom legend labels section
                use_custom_labels = st.checkbox("Use Custom Legend Labels", value=False, key="graph_use_custom_labels")
                legend_labels = {}
                if use_custom_labels:
                    st.write("Enter custom legend labels:")
                    for file_idx, cols in y_cols_by_file.items():
                        for col in cols:
                            # If segmentation is active, create label inputs for each segment
                            if file_idx in segment_data:
                                for value in segment_values[file_idx]:
                                    label_key = f"{file_idx}_{col}_{value}"
                                    legend_labels[label_key] = st.text_input(
                                        f"Custom label for {file_names[file_idx]} - {col} - {value}", 
                                        value=f"{file_names[file_idx]}: {col} ({value})",
                                        key=f"graph_label_{label_key}"
                                    )
                            else:
                                label_key = f"{file_idx}_{col}"
                                legend_labels[label_key] = st.text_input(
                                    f"Custom label for {file_names[file_idx]} - {col}", 
                                    value=f"{file_names[file_idx]}: {col}",
                                    key=f"graph_label_{label_key}"
                                )
                
                custom_xaxis_title = st.text_input("Enter the X-axis title", "Date", key="graph_xaxis_title")
                
                y_axis_title_default = "Normalized Values" if normalize else "Values"
                if use_dollar and not normalize:
                    y_axis_title_default = "Dollar Values"
                custom_yaxis_title = st.text_input("Enter the Y-axis title", y_axis_title_default, key="graph_yaxis_title")
                
                # Normalize data if requested
                if normalize:
                    for file_idx, df in sub_dfs.items():
                        if y_cols_by_file[file_idx]:
                            sub_dfs[file_idx][y_cols_by_file[file_idx]] = normalize_data(df[y_cols_by_file[file_idx]])
                
                # Buttons for generating graphs
                col1, col2 = st.columns(2)
                with col1:
                    generate_static = st.button("Generate Static Graph", key="graph_generate_static")
                with col2:
                    generate_animated = st.button("Generate Animated Graph", key="graph_generate_animated")
                
                # Generate graphs based on button clicks
                if generate_static or generate_animated:
                    st.write("Generating graph... Please wait.")
                    # Add your graph generation code here
                    # This would be a simplified version of the graph generation code from stre.py
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.write("Please check your data and try again.")

# New tab for editing and saving the newsletter
with edit_save_tab:
    st.title("Edit & Save Newsletter")
    st.markdown("Edit the generated newsletter and save the final version for fine-tuning.")
    
    # Check if we have a generated newsletter to edit
    if st.session_state.final_newsletter:
        # Display the title
        if st.session_state.newsletter_title:
            st.markdown(f"### Current Title: {st.session_state.newsletter_title}")
            edited_title = st.text_input("Edit Title", value=st.session_state.newsletter_title)
        else:
            edited_title = st.text_input("Add Title")
        
        # Allow editing of the newsletter content
        edited_newsletter = st.text_area(
            "Edit Newsletter Content", 
            value=st.session_state.final_newsletter,
            height=500
        )
        
        # Feedback on the generated newsletter
        quality_rating = st.number_input(
            "Rate the quality of the generated newsletter (1-10)",
            min_value=1,
            max_value=10,
            value=7,
            step=1
        )
        
        feedback_notes = st.text_area(
            "Additional feedback or notes on what was changed",
            height=150
        )
        
        # Save button
        if st.button("Save Newsletter for Fine-tuning"):
            if edited_newsletter:
                # Get the current client, topic, and context
                current_client = st.session_state.get("newsletter_client", "")
                current_topic = topic if 'topic' in locals() else ""
                current_context = context_text if 'context_text' in locals() else ""
                current_style = selected_style if 'selected_style' in locals() else ""
                
                # Save to Supabase
                save_success = save_to_supabase(
                    client=current_client,
                    topic=current_topic,
                    context=current_context,
                    generated=st.session_state.final_newsletter,
                    final=edited_newsletter,
                    style=current_style,
                    quality_rating=quality_rating,
                    feedback_notes=feedback_notes
                )
                
                if save_success:
                    st.success("Newsletter saved successfully for fine-tuning!")
                    
                    # Display summary of saved data
                    st.markdown("### Saved Newsletter Data")
                    st.markdown(f"**Client:** {current_client}")
                    st.markdown(f"**Topic:** {current_topic}")
                    st.markdown(f"**Style Reference:** {current_style}")
                    st.markdown(f"**Quality Rating:** {quality_rating}/10")
                    if feedback_notes:
                        st.markdown(f"**Feedback Notes:** {feedback_notes}")
                else:
                    st.error("Failed to save newsletter data. Please try again.")
            else:
                st.error("Please provide newsletter content to save.")
    else:
        st.warning("No newsletter has been generated yet. Please generate a newsletter in the Newsletter Generator tab first.")
        
        # Option to view previously saved newsletters
        st.markdown("### View Previously Saved Newsletters")
        if st.button("Load Saved Newsletters"):
            saved_newsletters = load_saved_newsletters()
            
            if saved_newsletters:
                st.markdown("#### Recent Saved Newsletters")
                for newsletter in saved_newsletters:
                    st.markdown(f"**ID:** {newsletter['id']} | **Client:** {newsletter['client']} | **Topic:** {newsletter['topic']} | **Date:** {newsletter['created_at']}")
            else:
                st.info("No saved newsletters found in the database.")

    # Add a section for viewing and exporting newsletters
    st.markdown("---")
    st.markdown("### Newsletter Management")

    # Create two columns for different actions
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### View Specific Newsletter")
        newsletter_id = st.number_input("Enter Newsletter ID", min_value=1, step=1)
        
        if st.button("View Newsletter"):
            if newsletter_id:
                newsletter_data = get_newsletter_by_id(newsletter_id)
                
                if newsletter_data:
                    st.markdown(f"**Client:** {newsletter_data.get('client', 'N/A')}")
                    st.markdown(f"**Topic:** {newsletter_data.get('topic', 'N/A')}")
                    st.markdown(f"**Quality Rating:** {newsletter_data.get('quality_rating', 'N/A')}/10")
                    
                    # Show the original context
                    with st.expander("Original Context"):
                        st.text_area("Context", value=newsletter_data.get('context_input', ''), height=200, disabled=True)
                    
                    # Show the generated newsletter
                    with st.expander("Generated Newsletter"):
                        st.text_area("Generated", value=newsletter_data.get('generated_newsletter', ''), height=300, disabled=True)
                    
                    # Show the final newsletter
                    with st.expander("Final Newsletter"):
                        st.text_area("Final", value=newsletter_data.get('final_newsletter', ''), height=300, disabled=True)
                    
                    # Show feedback notes if any
                    if newsletter_data.get('feedback_notes'):
                        with st.expander("Feedback Notes"):
                            st.write(newsletter_data.get('feedback_notes', ''))

    with col2:
        st.markdown("#### Export for Fine-tuning")
        st.write("Export high-quality newsletters (rating ≥ 7) for model fine-tuning.")
        
        if st.button("Export Fine-tuning Data"):
            finetuning_data = export_newsletters_for_finetuning()
            
            if finetuning_data:
                # Convert to JSON
                json_data = json.dumps(finetuning_data, indent=2)
                
                # Create a download button
                st.download_button(
                    label="Download JSON for Fine-tuning",
                    data=json_data,
                    file_name="newsletter_finetuning_data.json",
                    mime="application/json"
                )
                
                # Show a preview
                st.markdown(f"**{len(finetuning_data)} newsletters** ready for fine-tuning")
                with st.expander("Preview Fine-tuning Data"):
                    st.json(finetuning_data[:2] if len(finetuning_data) > 2 else finetuning_data)

# Function to generate image description using Anthropic Claude 3 Sonnet Vision
def generate_image_description_claude(image_url, api_key):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": "claude-3-7-sonnet-latest",
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in detail, focusing on its composition, colors, style, and how well it would work as a newsletter cover image. What emotions or themes does it convey?"
                            }
                        ]
                    }
                ]
            }
        )
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception: {str(e)}"

try:
    
    pass  # Replace with actual code
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check the logs for more details.")
    import traceback
    st.code(traceback.format_exc())


