import json
from pathlib import Path

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

from agents.research_workflow import run_research_workflow
from agents.evidence_validator import validate_research


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = BASE_DIR / "storage" / "chroma_db"
MULTIMODAL_PATH = BASE_DIR / "storage" / "multimodal_analysis"
VALIDATION_PATH = BASE_DIR / "storage" / "validation_reports"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Scientific Discovery Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Autonomous Multimodal Scientific Discovery Agent")

st.write(
    "An AI-powered research assistant for scientific paper search, "
    "research analysis, and evidence validation."
)


# --------------------------------------------------
# LOAD CHROMADB AND EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_search_components():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name="scientific_papers"
    )

    return model, collection


# --------------------------------------------------
# SEMANTIC PAPER SEARCH
# --------------------------------------------------

def search_papers(question, top_k=5):

    model, collection = load_search_components()

    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    papers = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):

        metadata = metadata or {}

        papers.append({
            "content": document,
            "title": metadata.get("title", "Unknown title"),
            "year": metadata.get("year", "Unknown"),
            "page": metadata.get(
                "page_number",
                metadata.get("page", "Unknown")
            ),
            "distance": distance
        })

    return papers


# --------------------------------------------------
# MULTIMODAL RESULTS
# --------------------------------------------------

def load_multimodal_results():

    results = []

    if not MULTIMODAL_PATH.exists():
        return results

    for file_path in MULTIMODAL_PATH.glob("*.json"):

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            results.append({
                "filename": file_path.name,
                "data": data
            })

        except Exception as error:
            results.append({
                "filename": file_path.name,
                "data": {"error": str(error)}
            })

    return results


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Research Controls")

top_k = st.sidebar.slider(
    "Number of papers to retrieve",
    min_value=1,
    max_value=10,
    value=5
)

st.sidebar.info(
    "The dashboard uses your existing ChromaDB research collection."
)


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Paper Search",
    "🧠 AI Research Analysis",
    "✅ Evidence Validation",
    "🖼️ Multimodal Results"
])


# --------------------------------------------------
# TAB 1: PAPER SEARCH
# --------------------------------------------------

with tab1:

    st.header("Scientific Paper Search")

    question = st.text_input(
        "Enter your research question",
        placeholder="Example: How can AI improve early disease detection?"
    )

    if st.button("Search Research Papers"):

        if not question.strip():

            st.warning("Please enter a research question.")

        else:

            try:

                with st.spinner("Searching your research database..."):

                    papers = search_papers(question, top_k)

                if not papers:

                    st.warning(
                        "No papers found. Check whether your ChromaDB "
                        "collection contains embedded research papers."
                    )

                else:

                    st.success(f"Found {len(papers)} relevant passages.")

                    for index, paper in enumerate(papers, start=1):

                        with st.expander(
                            f"{index}. {paper['title']}"
                        ):

                            st.write("**Year:**", paper["year"])
                            st.write("**Page:**", paper["page"])
                            st.write("**Distance:**", paper["distance"])
                            st.write("**Extracted content:**")
                            st.write(paper["content"])

            except Exception as error:

                st.error(f"Paper search failed: {error}")


# --------------------------------------------------
# TAB 2: AI RESEARCH ANALYSIS
# --------------------------------------------------

with tab2:

    st.header("AI-Powered Research Analysis")

    research_question = st.text_area(
        "Enter a research question for the multi-agent workflow",
        placeholder="Example: What are the current challenges in AI-based medical diagnosis?"
    )

    if st.button("Run Research Workflow"):

        if not research_question.strip():

            st.warning("Please enter a research question.")

        else:

            try:

                with st.spinner(
                    "Research agents are retrieving and analyzing evidence..."
                ):

                    result = run_research_workflow(
                        research_question
                    )

                st.success("Research workflow completed!")

                st.session_state["research_result"] = result

            except Exception as error:

                st.error(f"Research workflow failed: {error}")

    if "research_result" in st.session_state:

        result = st.session_state["research_result"]

        st.subheader("Research Report")

        if isinstance(result, dict):

            for section in [
                "analysis",
                "discovery",
                "report"
            ]:

                if result.get(section):

                    st.markdown(f"### {section.title()}")
                    st.write(result[section])

            with st.expander("View complete workflow output"):

                st.json(result)

        else:

            st.write(result)


# --------------------------------------------------
# TAB 3: EVIDENCE VALIDATION
# --------------------------------------------------

with tab3:

    st.header("Research Evidence Validation")

    st.write(
        "Enter scientific claims to check them against the research "
        "passages stored in your database."
    )

    claims_text = st.text_area(
        "Enter one claim per line",
        placeholder=(
            "Deep learning can help identify patterns in medical images.\n"
            "Machine learning can support early disease prediction."
        ),
        height=150
    )

    if st.button("Validate Claims"):

        claims = [
            claim.strip()
            for claim in claims_text.split("\n")
            if claim.strip()
        ]

        if not claims:

            st.warning("Please enter at least one claim.")

        else:

            try:

                with st.spinner("Validating claims against research evidence..."):

                    validation_result = validate_research(claims)

                st.success("Evidence validation completed!")

                st.session_state["validation_result"] = validation_result

            except Exception as error:

                st.error(f"Evidence validation failed: {error}")

    if "validation_result" in st.session_state:

        st.subheader("Validation Report")

        st.json(st.session_state["validation_result"])

        st.caption(
            "Review the supporting passages and exact-quote checks. "
            "Automated validation is not a substitute for expert review."
        )


# --------------------------------------------------
# TAB 4: MULTIMODAL RESULTS
# --------------------------------------------------

with tab4:

    st.header("Multimodal Paper Analysis")

    st.write(
        "View the image and page-analysis results generated by Phase 8."
    )

    if st.button("Load Multimodal Results"):

        multimodal_results = load_multimodal_results()

        st.session_state["multimodal_results"] = multimodal_results

    if "multimodal_results" in st.session_state:

        multimodal_results = st.session_state["multimodal_results"]

        if not multimodal_results:

            st.info(
                "No multimodal analysis JSON files found. "
                "Run Phase 8 first to generate results."
            )

        else:

            for item in multimodal_results:

                with st.expander(item["filename"]):

                    st.json(item["data"])


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Autonomous Multimodal Scientific Discovery Agent | "
    "Built with Python, Streamlit, ChromaDB, LangGraph and Gemini"
)