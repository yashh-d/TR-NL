import os
import streamlit as st
from typing import Optional
from datetime import datetime
from supabase import create_client, Client  # Import Supabase client
import json

# ----------------------------------------------------------------------
# Install Dependencies (via terminal/pip):
#   pip install langchain anthropic streamlit supabase
#   or adapt to your environment as needed.
# ----------------------------------------------------------------------

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
from langchain_anthropic import ChatAnthropic

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
                "model": "claude-3-sonnet-20240229",
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
    api_key = st.secrets["anthropic"]["api_key"]
    # If we get here, the key was found in secrets
    os.environ["ANTHROPIC_API_KEY"] = api_key  # Set it in environment for libraries that look there
except (KeyError, TypeError):
    # If not in secrets, try environment or session state
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
        "Draft a 'What happened' section (1–2 sentences) in a style similar to this example:\n{newsletter_example}"
        "{additional_instructions}"
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

################################
# 5. CREATE LLM CHAINS
################################
chain_extract = LLMChain(llm=anthropic_llm, prompt=extract_points_prompt)
chain_what_happened = LLMChain(llm=anthropic_llm, prompt=draft_what_happened_prompt)
chain_why_matters = LLMChain(llm=anthropic_llm, prompt=draft_why_matters_prompt)
chain_combined_big_picture = LLMChain(llm=anthropic_llm, prompt=combined_big_picture_prompt) # Combined Chain
chain_style = LLMChain(llm=anthropic_llm, prompt=style_check_prompt)
chain_style_edit = LLMChain(llm=anthropic_llm, prompt=style_edit_prompt)

# Separate chains for ecosystem and community bullet points
chain_ecosystem_bullet_points = LLMChain(llm=anthropic_llm, prompt=ecosystem_bullet_point_prompt)
chain_community_bullet_points = LLMChain(llm=anthropic_llm, prompt=community_bullet_point_prompt)

# New chain for tweet generation
chain_tweet_generation = LLMChain(llm=anthropic_llm, prompt=tweet_generation_prompt)

# New chain for title generation
chain_title_generation = LLMChain(llm=anthropic_llm, prompt=title_generation_prompt)

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

# Create tabs for Newsletter Generator, Bullet Point Generator, Tweet Generator, Title Generator, and Edit & Save
newsletter_tab, bullet_points_tab, tweet_tab, title_tab, edit_save_tab = st.tabs(["Newsletter Generator", "Bullet Point Generator", "Tweet Generator", "Title Generator", "Edit & Save"])

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

    # --- Newsletter user inputs ---
    context_text = st.text_area("Context Information (Newsletter)", height=150)
    topic = st.text_area("Newsletter Topic", height=50)
    
    # Add additional tailoring instructions right after context input
    st.markdown("### Additional Tailoring Instructions")
    st.markdown("Provide specific guidance on style, focus areas, or other aspects you want to emphasize in this newsletter.")
    
    additional_instructions = st.text_area(
        "Custom Instructions",
        placeholder="Examples:\n- Focus more on technical aspects\n- Emphasize community growth metrics\n- Use more concrete examples\n- Keep a neutral but optimistic tone\n- Highlight potential impact on developers",
        height=100
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
        if st.button("Confirm Key Points and Generate Newsletter", key="confirm_key_points"):
            st.session_state.step2_started = True
            
    # Generate the rest of the newsletter (Steps 2-5)
    if st.session_state.step2_started:
        if newsletter_example and topic:
            # Step 2a
            with st.spinner("Drafting 'What happened'..."):
                # Get additional instructions from session state if available
                additional_instr = st.session_state.get("additional_instructions", "")
                
                # Add to the run parameters if instructions exist
                run_params = {
                    "newsletter_example": newsletter_example,
                    "key_points": st.session_state.edited_key_points,
                    "topic": topic
                }
                
                if additional_instr:
                    run_params["additional_instructions"] = additional_instr
                
                what_happened_draft = chain_what_happened.run(**run_params)
            st.markdown("### Draft - What Happened")
            st.write(what_happened_draft)

            # Step 2b
            with st.spinner("Drafting 'Why it matters'..."):
                why_matters_draft = chain_why_matters.run(
                    newsletter_example=newsletter_example,
                    key_points=st.session_state.edited_key_points,
                    topic=topic
                )
            st.markdown("### Draft - Why It Matters")
            st.write(why_matters_draft)

            # Step 2c (Combined Draft & Enhance 'Big Picture')
            with st.spinner("Drafting & Enhancing 'The big picture'..."):
                big_picture_enhanced = chain_combined_big_picture.run(
                    newsletter_example=newsletter_example,
                    key_points=st.session_state.edited_key_points,
                    topic=topic,
                    long_term_doc=long_term_doc,  # Pass long_term_doc for combined chain
                    additional_instructions=combined_instructions
                )
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
                newsletter_style_edited = chain_style_edit.run(
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

with bullet_points_tab:
    st.title("Bullet Point Generator")
    
    # Create tabs for ecosystem and community bullet points
    ecosystem_tab, community_tab = st.tabs(["Ecosystem Bullet Points", "Community Bullet Points"])
    
    with ecosystem_tab:
        st.markdown("### Ecosystem Bullet Points")
        st.markdown("Generate bullet points related to technical developments, partnerships, protocol upgrades, and ecosystem growth.")
        
        # Load ecosystem examples from global file
        default_ecosystem_examples = load_bullet_point_examples("Ecosystem")
        
        ecosystem_example = st.text_area(
            "Ecosystem Bullet Point Examples",
            value=default_ecosystem_examples,
            height=200
        )
        
        ecosystem_context = st.text_area(
            "Ecosystem Context Information", 
            height=150
        )
        
        if st.button("Generate Ecosystem Bullet Points", key="gen_ecosystem"):
            if ecosystem_context:
                with st.spinner("Generating ecosystem bullet points..."):
                    ecosystem_bullets = chain_ecosystem_bullet_points.run(
                        context_text=ecosystem_context,
                        example_bullet_points=ecosystem_example
                    )
                st.markdown("#### Generated Ecosystem Bullet Points")
                st.write(ecosystem_bullets)
                
                # Add copy button
                if st.button("Copy to Clipboard", key="copy_ecosystem"):
                    st.success("Ecosystem bullet points copied to clipboard!")
            else:
                st.error("Please fill in the Ecosystem Context.")
    
    with community_tab:
        st.markdown("### Community Bullet Points")
        st.markdown("Generate bullet points related to community engagement, events, social metrics, and user adoption.")
        
        # Load community examples from global file
        default_community_examples = load_bullet_point_examples("Community")
        
        community_example = st.text_area(
            "Community Bullet Point Examples",
            value=default_community_examples,
            height=200
        )
        
        community_context = st.text_area(
            "Community Context Information", 
            height=150
        )
        
        if st.button("Generate Community Bullet Points", key="gen_community"):
            if community_context:
                with st.spinner("Generating community bullet points..."):
                    community_bullets = chain_community_bullet_points.run(
                        context_text=community_context,
                        example_bullet_points=community_example
                    )
                st.markdown("#### Generated Community Bullet Points")
                st.write(community_bullets)
                
                # Add copy button
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
