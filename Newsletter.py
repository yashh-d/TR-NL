import os
import streamlit as st
from typing import Optional

# ----------------------------------------------------------------------
# Install Dependencies (via terminal/pip):
#   pip install langchain anthropic streamlit
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
if "ANTHROPIC_API_KEY" not in os.environ:
    st.warning("Your Anthropic API key is not set in the environment. Please enter it below:")
    api_key = st.text_input("Enter your Anthropic API key:", type="password")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

anthropic_llm = ChatAnthropic(
    model="claude-3-7-sonnet-latest",  # Adjust model if needed
    temperature=0,
    timeout=None,
    max_retries=2,
)

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
    )
])

# (C) DRAFT 'WHY IT MATTERS' PROMPT
draft_why_matters_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Newsletter Writer: Draft 'Why It Matters' (6-7 paragraphs) for web3 newsletter. Match example style. Explain significance, impact, implications. Concise, neutral."
    ),
    (
        "human",
        "Key Points:\n{key_points}\n\n"
        "Newsletter Topic: {topic}\n\n"
        "Draft the 'Why does it matter' section (6–7 small paragraphs) in a style similar to this example:\n{newsletter_example}"
    )
])

# COMBINED 'THE BIG PICTURE' PROMPT (Draft and Enhance)
combined_big_picture_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Newsletter Editor & Writer:  Add to 'The Big Picture' (total 3 paragraphs) for web3 newsletter. Style, keeping it concise and neutral. Enhance it to add deeper insight, long-term vision, and ecosystem impact, informed by the client vision and based on documentation. Return the final enhanced 'The Big Picture' section."
    ),
    (
        "human",
        "Key Points (for initial draft):\n{key_points}\n\nNewsletter Topic:\n{topic}\n\nNewsletter Example (for style reference):\n{newsletter_example}\n\nClient vision + docs (for enhancement):\n{long_term_doc}\n\n"
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
        "Editor: Revise newsletter based on style feedback to match example. Focus on tone, structure, clarity. Output final newsletter."
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
# 6. STREAMLIT UI
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

# Create tabs for Newsletter Generator, Bullet Point Generator, and Tweet Generator
newsletter_tab, bullet_points_tab, tweet_tab = st.tabs(["Newsletter Generator", "Bullet Point Generator", "Tweet Generator"])

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
    selected_client = st.selectbox("Select a Client (Newsletter)", list(CLIENT_FILES.keys()), key="newsletter_client")
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
    topic = st.text_input("Newsletter Topic")

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
                st.session_state.key_points_output = chain_extract.run(context_text=context_text, long_term_doc=long_term_doc)
                st.session_state.edited_key_points = st.session_state.key_points_output
                st.session_state.step1_completed = True
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
                what_happened_draft = chain_what_happened.run(
                    newsletter_example=newsletter_example,
                    key_points=st.session_state.edited_key_points,
                    topic=topic
                )
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
                    long_term_doc=long_term_doc  # Pass long_term_doc for combined chain
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

# No instructions at the bottom