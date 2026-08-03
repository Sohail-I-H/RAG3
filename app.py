import os
import tempfile

import streamlit as st

from parser import extract_structured_sections
from rag_engine import PaperAnalysisRAG
from search import discover_papers, format_results


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="PragyanAI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PragyanAI")
st.caption("Academic Paper RAG & Research Gap Explorer")


# =============================================================================
# SESSION STATE
# =============================================================================

if "rag" not in st.session_state:
    st.session_state.rag = None

if "papers" not in st.session_state:
    st.session_state.papers = {}

if "processed" not in st.session_state:
    st.session_state.processed = False


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.header("Configuration")

    try:
        groq_key = st.secrets["GROQ_API_KEY"]
        st.success("Groq API Key Loaded")
    except Exception:
        groq_key = st.text_input(
            "Groq API Key",
            type="password"
        )

    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    process = st.button(
        "Process Papers",
        use_container_width=True
    )


# =============================================================================
# PROCESS PDFS
# =============================================================================

if process:

    if not groq_key:

        st.error("Please provide a Groq API Key.")

    elif not uploaded_files:

        st.error("Upload at least one PDF.")

    else:

        with st.spinner("Reading papers..."):

            parsed = {}

            progress = st.progress(0)

            for index, pdf in enumerate(uploaded_files):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(pdf.read())

                    path = tmp.name

                sections = extract_structured_sections(path)

                parsed[pdf.name] = sections

                os.remove(path)

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            rag = PaperAnalysisRAG(groq_key)

            rag.ingest_papers(parsed)

            st.session_state.rag = rag
            st.session_state.papers = parsed
            st.session_state.processed = True

        st.success("Papers processed successfully.")


# =============================================================================
# STOP
# =============================================================================

if not st.session_state.processed:

    st.info("Upload PDF papers from the sidebar.")

    st.stop()


rag = st.session_state.rag

paper_names = list(st.session_state.papers.keys())


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Extract Sections",
        "Comparison",
        "Research Gaps",
        "RAG Q&A",
        "Discover Papers"
    ]
)


# =============================================================================
# TAB 1
# =============================================================================

with tab1:

    st.subheader("Extract Sections")

    paper = st.selectbox(
        "Paper",
        paper_names
    )

    selected_sections = st.multiselect(

        "Sections",

        [
            "Abstract",
            "Introduction",
            "Related Work",
            "Methodology",
            "Results",
            "Discussion & Gaps",
            "Conclusion"
        ],

        default=["Abstract"]
    )

    if st.button("Extract Sections"):

        result = rag.extract_sections(
            paper,
            selected_sections
        )

        st.markdown(result)


# =============================================================================
# TAB 2
# =============================================================================

with tab2:

    st.subheader("Comparative Analysis")

    aspect = st.text_input(

        "Comparison Aspect",

        value="Methodology, Dataset, Results"
    )

    if st.button("Generate Comparison"):

        with st.spinner("Generating..."):

            result = rag.generate_comparison(
                aspect
            )

        st.markdown(result)


# =============================================================================
# TAB 3
# =============================================================================

with tab3:

    st.subheader("Research Gap Finder")

    if st.button("Analyze Research Gaps"):

        with st.spinner("Finding gaps..."):

            result = rag.identify_research_gaps()

        st.markdown(result)


# =============================================================================
# TAB 4
# =============================================================================

with tab4:

    st.subheader("Deep RAG Question Answering")

    query = st.text_area(
        "Ask a question"
    )

    section = st.selectbox(

        "Restrict Search",

        [
            "All",
            "Abstract",
            "Introduction",
            "Related Work",
            "Methodology",
            "Results",
            "Discussion & Gaps",
            "Conclusion"
        ]
    )

    if st.button("Ask"):

        if query.strip():

            with st.spinner("Thinking..."):

                answer = rag.answer_question(
                    query,
                    section
                )

            st.markdown(answer)

        else:

            st.warning("Enter a question.")


# =============================================================================
# TAB 5
# =============================================================================

with tab5:

    st.subheader("Discover Similar Papers")

    topic = st.text_input(
        "Research Topic"
    )

    if st.button("Search Papers"):

        if topic.strip():

            with st.spinner("Searching..."):

                results = discover_papers(
                    topic
                )

            st.markdown(
                format_results(results)
            )

        else:

            st.warning("Enter a research topic.")


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

stats = rag.get_statistics()

c1, c2, c3 = st.columns(3)

c1.metric("Papers", stats["papers"])

c2.metric("Sections", stats["sections"])

c3.metric(
    "Vector Store",
    "Ready" if stats["vector_store_ready"] else "Not Ready"
)
