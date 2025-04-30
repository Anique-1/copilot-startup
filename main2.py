import os
import json
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import serpapi
from dotenv import load_dotenv
import asyncio
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client for AI/ML API
aimlapi_key = "c469a9eb66fa4a26a45b851d112a5807"
if not aimlapi_key:
    raise ValueError("AIMLAPI_KEY environment variable not set. Please check your .env file.")
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=aimlapi_key
)

# SerpAPI key for competitor analysis
SERPAPI_KEY = "d7ba1fde70335b113f681f1a5d0359c6057a70e5a85a4fabca8e632d635472a2"
if not SERPAPI_KEY:
    raise ValueError("SERPAPI_KEY environment variable not set. Please check your .env file.")

# Define state for LangGraph
class StartupState(TypedDict):
    idea: str
    industry: str
    refined_idea: str
    market_gap: str
    business_plan: Dict
    pitch_deck: List[Dict]
    competitors: List[Dict]

# 1. Startup Idea Assistant
idea_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a startup expert. Refine the user's startup idea and identify market gaps in the following format:\nRefined Idea: [description]\nMarket Gap: [description]"),
    HumanMessage(content="Industry: {industry}\nIdea: {idea}\nRefine the idea and suggest market gaps.")
])

async def idea_assistant(state: StartupState) -> StartupState:
    prompt = idea_prompt.format_messages(industry=state["industry"], idea=state["idea"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt[0].content},
            {"role": "user", "content": prompt[1].content}
        ]
    )
    content = response.choices[0].message.content
    print("Raw Idea Assistant Response:", content)

    try:
        output = json.loads(content)
    except json.JSONDecodeError:
        output = {
            "refined_idea": content.split("Refined Idea:")[1].split("Market Gap:")[0].strip() if "Refined Idea:" in content and "Market Gap:" in content else content,
            "market_gap": content.split("Market Gap:")[1].strip() if "Market Gap:" in content else ""
        }
    state["refined_idea"] = output.get("refined_idea", "No refined idea generated.")
    state["market_gap"] = output.get("market_gap", "No market gap identified.")
    print("Parsed Idea Assistant Output:", output)
    return state

# 2. Business Plan Generator
business_plan_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Generate a lean business canvas for a startup in the following format:\nProblem: [description]\nSolution: [description]\nValue Proposition: [description]\nCustomer Segments: [description]\nChannels: [description]\nRevenue Streams: [description]\nCost Structure: [description]\nKey Metrics: [description]\nUnfair Advantage: [description]"),
    HumanMessage(content="Idea: {refined_idea}\nIndustry: {industry}\nMarket Gap: {market_gap}\nCreate a lean business canvas with: Problem, Solution, Value Proposition, Customer Segments, Channels, Revenue Streams, Cost Structure, Key Metrics, Unfair Advantage.")
])

async def business_plan_generator(state: StartupState) -> StartupState:
    prompt = business_plan_prompt.format_messages(
        refined_idea=state["refined_idea"],
        industry=state["industry"],
        market_gap=state["market_gap"]
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt[0].content},
            {"role": "user", "content": prompt[1].content}
        ]
    )
    content = response.choices[0].message.content
    print("Raw Business Plan Response:", content)

    try:
        state["business_plan"] = json.loads(content)
    except json.JSONDecodeError:
        sections = [
            "Problem", "Solution", "Value Proposition", "Customer Segments",
            "Channels", "Revenue Streams", "Cost Structure", "Key Metrics", "Unfair Advantage"
        ]
        business_plan = {}
        current_section = None
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for section in sections:
                if line.startswith(f"{section}:"):
                    current_section = section
                    business_plan[section] = line[len(section) + 1:].strip()
                    break
            else:
                if current_section and line:
                    business_plan[current_section] += f" {line}"
        # Ensure all sections are present
        for section in sections:
            if section not in business_plan:
                business_plan[section] = "Not specified."
        state["business_plan"] = business_plan
    print("Parsed Business Plan:", state["business_plan"])
    return state

# 3. Pitch Deck Slide Writer
pitch_deck_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Generate text content for a startup pitch deck in the following format:\nProblem: [description]\nSolution: [description]\nMarket: [description]\nTeam: [description]\nAsk: [description]"),
    HumanMessage(content="Idea: {refined_idea}\nBusiness Plan: {business_plan}\nGenerate text for 5 slides: Problem, Solution, Market, Team, Ask.")
])

async def pitch_deck_writer(state: StartupState) -> StartupState:
    prompt = pitch_deck_prompt.format_messages(
        refined_idea=state["refined_idea"],
        business_plan=json.dumps(state["business_plan"])
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt[0].content},
            {"role": "user", "content": prompt[1].content}
        ]
    )
    content = response.choices[0].message.content
    print("Raw Pitch Deck Response:", content)

    # Initialize pitch deck with default empty values
    slides = [
        {"slide": "Problem", "content": ""},
        {"slide": "Solution", "content": ""},
        {"slide": "Market", "content": ""},
        {"slide": "Team", "content": ""},
        {"slide": "Ask", "content": ""}
    ]

    # Try parsing as JSON first
    try:
        parsed = json.loads(content)
        for slide in slides:
            slide["content"] = parsed.get(slide["slide"], "Not specified.")
    except json.JSONDecodeError:
        # Fallback to string splitting with robust parsing
        lines = content.split("\n")
        current_slide = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if the line starts with a slide name (e.g., "Problem:")
            for slide in slides:
                if line.startswith(f"{slide['slide']}:"):
                    current_slide = slide["slide"]
                    slide["content"] = line[len(slide["slide"]) + 1:].strip()
                    break
            # If we're in a slide section, append to the current slide's content
                elif current_slide:
                    for slide in slides:
                        if slide["slide"] == current_slide:
                            slide["content"] += f" {line}"
                            break

    state["pitch_deck"] = slides
    print("Parsed Pitch Deck:", state["pitch_deck"])
    return state

# 4. Competitor Analyzer
async def competitor_analyzer(state: StartupState) -> StartupState:
    params = {
        "engine": "google",
        "q": f"{state['industry']} startup competitors",
        "api_key": SERPAPI_KEY
    }
    try:
        results = serpapi.search(params)
        competitors = []
        for result in results.get("organic_results", [])[:3]:
            competitors.append({
                "title": result.get("title", "Unknown"),
                "link": result.get("link", "#"),
                "snippet": result.get("snippet", "No description available.")
            })
        state["competitors"] = competitors
    except Exception as e:
        print(f"Competitor Analyzer Error: {e}")
        state["competitors"] = []
    print("Competitor Analysis Results:", state["competitors"])
    return state

# Define LangGraph workflow
workflow = StateGraph(StartupState)

# Add nodes
workflow.add_node("idea_assistant", idea_assistant)
workflow.add_node("business_plan_generator", business_plan_generator)
workflow.add_node("pitch_deck_writer", pitch_deck_writer)
workflow.add_node("competitor_analyzer", competitor_analyzer)

# Define edges
workflow.add_edge("idea_assistant", "business_plan_generator")
workflow.add_edge("business_plan_generator", "pitch_deck_writer")
workflow.add_edge("pitch_deck_writer", "competitor_analyzer")
workflow.add_edge("competitor_analyzer", END)

# Set entry point
workflow.set_entry_point("idea_assistant")

# Compile graph
app = workflow.compile()

# Streamlit Interface
st.title("Startup Copilot")
st.subheader("AI-Powered Assistant for Founders")

# Input form
with st.form("startup_form"):
    industry = st.text_input("Industry", placeholder="e.g., FinTech, HealthTech")
    idea = st.text_area("Startup Idea", placeholder="e.g., A mobile app for micro-investing in ETFs")
    submitted = st.form_submit_button("Generate Startup Plan")

# Process inputs and display results
if submitted:
    if not industry or not idea:
        st.error("Please provide both an industry and a startup idea.")
    else:
        async def run_copilot(industry: str, idea: str) -> StartupState:
            initial_state = StartupState(
                idea=idea,
                industry=industry,
                refined_idea="",
                market_gap="",
                business_plan={},
                pitch_deck=[],
                competitors=[]
            )
            return await app.ainvoke(initial_state)

        # Run the copilot with a loading spinner
        with st.spinner("Generating startup plan..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_copilot(industry, idea))
            except Exception as e:
                st.error(f"Error generating startup plan: {e}")
                result = StartupState(
                    idea=idea,
                    industry=industry,
                    refined_idea="",
                    market_gap="",
                    business_plan={},
                    pitch_deck=[],
                    competitors=[]
                )

        # Display results
        st.subheader("Refined Idea")
        st.markdown(result["refined_idea"] if result["refined_idea"] else "No refined idea generated.")

        st.subheader("Market Gap Analysis")
        st.markdown(result["market_gap"] if result["market_gap"] else "No market gap analysis generated.")

        st.subheader("Lean Business Canvas")
        with st.expander("View Business Canvas Details"):
            if result["business_plan"]:
                for key, value in result["business_plan"].items():
                    st.markdown(f"**{key}**: {value}")
            else:
                st.warning("No business plan generated.")

        st.subheader("Pitch Deck Slides")
        with st.expander("View Pitch Deck Slides"):
            if result["pitch_deck"]:
                for slide in result["pitch_deck"]:
                    content = slide["content"] if slide["content"] else "No content generated for this slide."
                    st.markdown(f"**{slide['slide']}**: {content}")
            else:
                st.warning("No pitch deck slides generated.")

        st.subheader("Competitor Analysis")
        with st.expander("View Competitor Analysis"):
            if result["competitors"]:
                for competitor in result["competitors"]:
                    st.markdown(f"**{competitor['title']}**: {competitor['snippet']} [\[Link\]]({competitor['link']})")
            else:
                st.warning("No competitor analysis generated.")