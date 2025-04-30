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

# SerpAPI key for competitor analysis and startup ideas
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

    slides = [
        {"slide": "Problem", "content": ""},
        {"slide": "Solution", "content": ""},
        {"slide": "Market", "content": ""},
        {"slide": "Team", "content": ""},
        {"slide": "Ask", "content": ""}
    ]

    try:
        parsed = json.loads(content)
        for slide in slides:
            slide["content"] = parsed.get(slide["slide"], "Not specified.")
    except json.JSONDecodeError:
        lines = content.split("\n")
        current_slide = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for slide in slides:
                if line.startswith(f"{slide['slide']}:"):
                    current_slide = slide["slide"]
                    slide["content"] = line[len(slide["slide"]) + 1:].strip()
                    break
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
        for result in results.get("organic_results", [])[:5]:  # Limit to top 5 results
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

# 5. City-Based Startup Ideas Fetcher
async def fetch_city_startups(city: str) -> List[Dict]:
    params = {
        "engine": "google",
        "q": f"recent startups in {city} 2023..2025",
        "api_key": SERPAPI_KEY
    }
    try:
        results = serpapi.search(params)
        startups = []
        for result in results.get("organic_results", [])[:5]:  # Limit to top 5 results
            startups.append({
                "name": result.get("title", "Unknown Startup"),
                "description": result.get("snippet", "No description available."),
                "website": result.get("link", "#")
            })
        return startups
    except Exception as e:
        print(f"City Startup Fetcher Error: {e}")
        return []

# Define LangGraph workflow
workflow = StateGraph(StartupState)
workflow.add_node("idea_assistant", idea_assistant)
workflow.add_node("business_plan_generator", business_plan_generator)
workflow.add_node("pitch_deck_writer", pitch_deck_writer)
workflow.add_node("competitor_analyzer", competitor_analyzer)
workflow.add_edge("idea_assistant", "business_plan_generator")
workflow.add_edge("business_plan_generator", "pitch_deck_writer")
workflow.add_edge("pitch_deck_writer", "competitor_analyzer")
workflow.add_edge("competitor_analyzer", END)
workflow.set_entry_point("idea_assistant")
app = workflow.compile()

# Streamlit Interface with Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Startup Copilot", "City Startup Ideas"])

if page == "Startup Copilot":
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

elif page == "City Startup Ideas":
    st.title("Discover Recent Startups by City")
    st.subheader("Find innovative startups in your area")

    # Input form for city
    with st.form("city_form"):
        city = st.text_input("City", placeholder="e.g., San Francisco, Bangalore")
        submitted = st.form_submit_button("Search Startups")

    # Process city input and display startup ideas
    if submitted:
        if not city:
            st.error("Please provide a city name.")
        else:
            with st.spinner(f"Fetching recent startups in {city}..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    startups = loop.run_until_complete(fetch_city_startups(city))
                except Exception as e:
                    st.error(f"Error fetching startups: {e}")
                    startups = []

            st.subheader(f"Recent Startups in {city}")
            with st.expander("View Startup Details"):
                if startups:
                    for startup in startups:
                        st.markdown(
                            f"**{startup['name']}**: {startup['description']} "
                            f"[\[Website\]]({startup['website']})"
                        )
                else:
                    st.warning(f"No recent startups found in {city}.")