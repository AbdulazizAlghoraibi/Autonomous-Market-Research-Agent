import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from tavily import TavilyClient

load_dotenv()

st.set_page_config(page_title="Autonomous Market Research Agent", page_icon="📈", layout="wide")
st.title("📈 Autonomous Market Research Agent")
st.markdown("Powered by **CrewAI**, **Gemini 2.5 Flash**, and **Tavily Search API**.")

@tool("Web Search")
def web_search_tool(query: str) -> str:
    """Search the web for information using Tavily."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query, search_depth="basic", max_results=5)
    return str(results.get('results', []))

topic_input = st.text_input("Enter the Market Research Topic:")
run_button = st.button("Generate Strategic Report")

if run_button and topic_input:
    with st.spinner("Agents are working..."):
        
        gemini_llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        researcher = Agent(
            role='Senior Market Researcher',
            goal=f'Conduct deep, highly accurate web research on: {topic_input}. Extract the top 3 most critical market data points and recent trends.',
            backstory='An elite market researcher working for a Fortune 500 company. Known for filtering noise and finding high-value, verified data using advanced search tools.',
            verbose=True,
            allow_delegation=False,
            tools=[web_search_tool],
            llm=gemini_llm
        )

        analyst = Agent(
            role='Principal Strategic Analyst',
            goal='Synthesize raw research data into a high-level, actionable Markdown report with strengths, challenges, and future implications.',
            backstory='A top-tier business strategist who transforms raw data into executive-ready insights. Specializes in clear, impactful Markdown formatting.',
            verbose=True,
            allow_delegation=False,
            llm=gemini_llm
        )

        research_task = Task(
            description=f'Use the search tool to find the latest, most impactful information about: {topic_input}. Summarize the top 3 key findings and strictly include the source URLs.',
            expected_output='A structured text summary of 3 key data points with their respective source URLs.',
            agent=researcher
        )

        analysis_task = Task(
            description='Analyze the findings from the Researcher. Write a final strategic report in Markdown. MUST include these exact headers: ## Executive Summary, ## Key Findings, ## Market Challenges, ## Future Outlook, and ## Sources.',
            expected_output='A professional, ready-to-publish Markdown report.',
            agent=analyst
        )

        market_research_crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            process=Process.sequential
        )

        result = market_research_crew.kickoff()
        
        st.success("Research Completed!")
        st.markdown("---")
        st.markdown(str(result))