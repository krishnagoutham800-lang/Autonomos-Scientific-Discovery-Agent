from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from agents.research_analyzer import (
    retrieve_research,
    gemini_client
)


# Shared state passed between agents
class ResearchState(TypedDict, total=False):
    question: str
    evidence: list
    analysis: str
    discovery: str
    report: str


# AGENT 1: Research Retrieval Agent
def retrieval_agent(state: ResearchState):

    print("\n[Agent 1] Retrieving scientific evidence...")

    papers = retrieve_research(
        state["question"],
        top_k=8
    )

    return {"evidence": papers}


# Prepare evidence for Gemini
def format_evidence(papers):

    context = ""

    for i, paper in enumerate(papers):

        context += f"""
        SOURCE [{i + 1}]
        Title: {paper['title']}
        Year: {paper['year']}
        Page: {paper['page']}

        Content:
        {paper['content']}

        -------------------------
        """

    return context


# AGENT 2: Research Analysis Agent
def analysis_agent(state: ResearchState):

    print("\n[Agent 2] Analyzing scientific findings...")

    context = format_evidence(state["evidence"])

    prompt = f"""
    You are a scientific research analysis agent.

    Research Question:
    {state['question']}

    Scientific Evidence:
    {context}

    Analyze the evidence and provide:

    1. Research overview
    2. Main findings
    3. Comparison of findings
    4. Limitations

    Cite evidence using SOURCE numbers.

    Do not invent scientific facts or references.
    Mention when evidence is insufficient.
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"analysis": response.text or ""}


# AGENT 3: Research Discovery Agent
def discovery_agent(state: ResearchState):

    print("\n[Agent 3] Identifying potential research gaps...")

    prompt = f"""
    You are a scientific research discovery agent.

    Research Question:
    {state['question']}

    Analysis of Retrieved Evidence:
    {state['analysis']}

    Identify:

    1. Potential research gaps
    2. Unanswered research questions
    3. Possible research hypotheses
    4. Suggested future research directions

    Clearly distinguish evidence-based observations
    from your own proposed hypotheses.

    Do not claim that a gap exists across the entire
    scientific literature based only on these sources.
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"discovery": response.text or ""}


# AGENT 4: Report Generator
def report_agent(state: ResearchState):

    print("\n[Agent 4] Generating final research report...")

    report = f"""
    AUTONOMOUS SCIENTIFIC DISCOVERY REPORT

    RESEARCH QUESTION
    {state['question']}


    RESEARCH ANALYSIS
    {state['analysis']}


    RESEARCH DISCOVERY
    {state['discovery']}


    SOURCES USED
    """

    for i, paper in enumerate(state["evidence"]):
        report += f"""
        SOURCE [{i + 1}]
        Title: {paper['title']}
        Year: {paper['year']}
        Page: {paper['page']}
        """

    return {"report": report}


# BUILD LANGGRAPH WORKFLOW
workflow = StateGraph(ResearchState)

workflow.add_node("retrieval", retrieval_agent)
workflow.add_node("analysis", analysis_agent)
workflow.add_node("discovery", discovery_agent)
workflow.add_node("report", report_agent)

workflow.add_edge(START, "retrieval")
workflow.add_edge("retrieval", "analysis")
workflow.add_edge("analysis", "discovery")
workflow.add_edge("discovery", "report")
workflow.add_edge("report", END)

research_graph = workflow.compile()


# RUN THE COMPLETE WORKFLOW
def run_research_workflow(question):

    result = research_graph.invoke({
        "question": question
    })

    return result["report"]