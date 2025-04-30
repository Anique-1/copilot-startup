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
    SystemMessage(content="You are a startup expert. Refine the user's startup idea and identify market gaps."),
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
    try:
        output = json.loads(content)
    except json.JSONDecodeError:
        output = {
            "refined_idea": content.split("Refined Idea:")[1].split("Market Gap:")[0].strip() if "Refined Idea:" in content else content,
            "market_gap": content.split("Market Gap:")[1].strip() if "Market Gap:" in content else ""
        }
    state["refined_idea"] = output["refined_idea"]
    state["market_gap"] = output["market_gap"]
    return state

# 2. Business Plan Generator
business_plan_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Generate a lean business canvas for a startup."),
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
    try:
        state["business_plan"] = json.loads(content)
    except json.JSONDecodeError:
        state["business_plan"] = {
            "Problem": content.split("Problem:")[1].split("Solution:")[0].strip() if "Problem:" in content else "",
            "Solution": content.split("Solution:")[1].split("Value Proposition:")[0].strip() if "Solution:" in content else "",
            "Value Proposition": content.split("Value Proposition:")[1].split("Customer Segments:")[0].strip() if "Value Proposition:" in content else "",
            "Customer Segments": content.split("Customer Segments:")[1].split("Channels:")[0].strip() if "Customer Segments:" in content else "",
            "Channels": content.split("Channels:")[1].split("Revenue Streams:")[0].strip() if "Channels:" in content else "",
            "Revenue Streams": content.split("Revenue Streams:")[1].split("Cost Structure:")[0].strip() if "Revenue Streams:" in content else "",
            "Cost Structure": content.split("Cost Structure:")[1].split("Key Metrics:")[0].strip() if "Cost Structure:" in content else "",
            "Key Metrics": content.split("Key Metrics:")[1].split("Unfair Advantage:")[0].strip() if "Key Metrics:" in content else "",
            "Unfair Advantage": content.split("Unfair Advantage:")[1].strip() if "Unfair Advantage:" in content else ""
        }
    return state

# 3. Pitch Deck Slide Writer
pitch_deck_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Generate text content for a startup pitch deck."),
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
    try:
        state["pitch_deck"] = json.loads(content)
    except json.JSONDecodeError:
        state["pitch_deck"] = [
            {"slide": "Problem", "content": content.split("Problem:")[1].split("Solution:")[0].strip() if "Problem:" in content else ""},
            {"slide": "Solution", "content": content.split("Solution:")[1].split("Market:")[0].strip() if "Solution:" in content else ""},
            {"slide": "Market", "content": content.split("Market:")[1].split("Team:")[0].strip() if "Market:" in content else ""},
            {"slide": "Team", "content": content.split("Team:")[1].split("Ask:")[0].strip() if "Team:" in content else ""},
            {"slide": "Ask", "content": content.split("Ask:")[1].strip() if "Ask:" in content else ""}
        ]
    return state

# 4. Competitor Analyzer
async def competitor_analyzer(state: StartupState) -> StartupState:
    params = {
        "engine": "google",
        "q": f"{state['industry']} startup competitors",
        "api_key": SERPAPI_KEY
    }
    results = serpapi.search(params)
    competitors = []
    for result in results.get("organic_results", [])[:3]:
        competitors.append({
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("snippet")
        })
    state["competitors"] = competitors
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
if submitted and industry and idea:
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

    # Run the copilot
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(run_copilot(industry, idea))

    # Display results
    st.subheader("Refined Idea")
    st.write(result["refined_idea"])

    st.subheader("Market Gap Analysis")
    st.write(result["market_gap"])

    st.subheader("Lean Business Canvas")
    for key, value in result["business_plan"].items():
        st.write(f"**{key}**: {value}")

    st.subheader("Pitch Deck Slides")
    for slide in result["pitch_deck"]:
        st.write(f"**{slide['slide']}**: {slide['content']}")

    st.subheader("Competitor Analysis")
    for competitor in result["competitors"]:
        st.write(f"**{competitor['title']}**: {competitor['snippet']} ([Link]({competitor['link']}))")