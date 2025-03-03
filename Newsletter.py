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
        # Explain how this can relate to the broader context of what is happening in the broader ecosystem, impacting users developers and more 
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

# (H) ECOSYSTEM BULLET POINT PROMPT
ecosystem_bullet_point_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Analyst: Generate concise ecosystem bullet points for web3 newsletter. Focus on technical developments, partnerships, protocol upgrades, integrations, and ecosystem growth. Simple, neutral, informative style. Match example format if provided."
    ),
    (
        "human",
        "Context:\n{context_text}\n\n"
        "Example Bullet Points:\n{example_bullet_points}\n\n"
        "Generate ecosystem-focused bullet points with the most important information."
    )
])

# (I) COMMUNITY BULLET POINT PROMPT
community_bullet_point_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Analyst: Generate concise community bullet points for web3 newsletter. Focus on community events, user adoption, social metrics, governance participation, and community engagement. Simple, neutral, informative style. Match example format if provided."
    ),
    (
        "human",
        "Context:\n{context_text}\n\n"
        "Example Bullet Points:\n{example_bullet_points}\n\n"
        "Generate community-focused bullet points with the most important information."
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

st.markdown(
    """
    **Newsletter Generator Steps:**
    **Step 1:** Extract key points from the context and client documentation + create detailed structure for each section of the newsletter \n
    **Step 2:** Human review and edit of extracted key points.\n
    **Step 3:** Draft 'What happened' section.\n
    **Step 4:** Draft 'Why it matters' section.\n
    **Step 5:** Draft & Enhance 'The big picture' section. \n
    **Step 6:** Compare the generated newsletter's style against the example.\n
    **Step 7:** Apply style edits to the enhanced newsletter based on feedback.\n
    """
)

# --- Client selection & loading client documentation ---
selected_client = st.selectbox("Select a Client (Newsletter)", list(CLIENT_FILES.keys()))
long_term_doc = ""
if selected_client:
    try:
        with open(CLIENT_FILES[selected_client], "r") as f:
            long_term_doc = f.read()
    except FileNotFoundError:
        st.error(f"File not found for {selected_client} at {CLIENT_FILES[selected_client]}.")

# --- Style Reference Selection ---
STYLE_FILES = {
    "Technology Focused": "style_references/technology_style.txt",
    "Metrics Driven": "style_references/metrics_style.txt",
    "New Dev/Partnership": "style_references/partnership_style.txt",
}
selected_style = st.selectbox("Select Newsletter Style Reference", list(STYLE_FILES.keys()))
newsletter_example = ""
if selected_style:
    try:
        with open(STYLE_FILES[selected_style], "r") as f:
            newsletter_example = f.read()
    except FileNotFoundError:
        st.error(f"Style file not found at {STYLE_FILES[selected_style]}. Please ensure the file exists.")

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
    
# Button: Extract Key Points (Step 1)
if st.button("Step 1: Extract Key Points + Structure for Newsletter") and not st.session_state.step1_completed:
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
    if st.button("Confirm Key Points and Generate Newsletter"):
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

    else:
        st.error("Please select a Style Reference and enter a Topic.")

    # Button to restart the process
    if st.button("Start Over"):
        st.session_state.key_points_output = ""
        st.session_state.edited_key_points = ""
        st.session_state.step1_completed = False
        st.session_state.step2_started = False
        st.experimental_rerun()

# Display the bullet point generator section only if not in the middle of newsletter generation
if not (st.session_state.step1_completed and not st.session_state.step2_started):
    # Separator
    st.markdown("---")
    st.title("Bullet Point Generator")
    
    # Create tabs for ecosystem and community bullet points
    ecosystem_tab, community_tab = st.tabs(["Ecosystem Bullet Points", "Community Bullet Points"])
    
    with ecosystem_tab:
        st.markdown("### Ecosystem Bullet Points")
        st.markdown("Generate bullet points related to technical developments, partnerships, protocol upgrades, and ecosystem growth.")
        
        ecosystem_example = st.text_area("Example Ecosystem Bullet Points (Optional)",                            
                                         height=100)
        ecosystem_context = st.text_area("Ecosystem Context", 
                                         height=150)
        
        if st.button("Generate Ecosystem Bullet Points"):
            if ecosystem_context:
                with st.spinner("Generating ecosystem bullet points..."):
                    ecosystem_bullets = chain_ecosystem_bullet_points.run(
                        context_text=ecosystem_context,
                        example_bullet_points=ecosystem_example
                    )
                st.markdown("#### Generated Ecosystem Bullet Points")
                st.write(ecosystem_bullets)
            else:
                st.error("Please fill in the Ecosystem Context.")
    
    with community_tab:
        st.markdown("### Community Bullet Points")
        st.markdown("Generate bullet points related to community engagement, events, social metrics, and user adoption.")
        
        community_example = st.text_area("Example Community Bullet Points (Optional)",
                                         height=100)
        community_context = st.text_area("Community Context", 
                                         height=150)
        
        if st.button("Generate Community Bullet Points"):
            if community_context:
                with st.spinner("Generating community bullet points..."):
                    community_bullets = chain_community_bullet_points.run(
                        context_text=community_context,
                        example_bullet_points=community_example
                    )
                st.markdown("#### Generated Community Bullet Points")
                st.write(community_bullets)
            else:
                st.error("Please fill in the Community Context.")