"""
rag_engine.py

Core RAG Engine
----------------
• Builds FAISS vector database
• Uses HuggingFace embeddings
• Uses Groq Llama models
• Performs semantic search
• Generates:
    - Comparative analysis
    - Research gap analysis
    - Context-aware Q&A
"""

from typing import Dict, List

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PaperAnalysisRAG:

    def __init__(self, groq_api_key: str):

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_store = None

        self.parsed_papers = {}

    # ===========================================================
    # Ingest Papers
    # ===========================================================

    def ingest_papers(
        self,
        papers: Dict[str, Dict[str, str]]
    ):

        self.parsed_papers = papers

        documents = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        for paper_name, sections in papers.items():

            for section_name, content in sections.items():

                chunks = splitter.split_text(content)

                for chunk in chunks:

                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "paper": paper_name,
                                "section": section_name
                            }
                        )
                    )

        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )

    # ===========================================================
    # Extract Sections
    # ===========================================================

    def extract_sections(
        self,
        paper_name: str,
        selected_sections: List[str]
    ) -> str:

        if paper_name not in self.parsed_papers:
            return "Paper not found."

        sections = self.parsed_papers[paper_name]

        output = ""

        for section in selected_sections:

            output += f"# {section}\n\n"

            if section in sections:

                output += sections[section]

            else:

                output += "Section not found."

            output += "\n\n---\n\n"

        return output

    # ===========================================================
    # Comparative Analysis
    # ===========================================================

    def generate_comparison(
        self,
        aspect: str
    ) -> str:

        context = []

        for paper, sections in self.parsed_papers.items():

            context.append(f"\n===== {paper} =====\n")

            for section, content in sections.items():

                context.append(
                    f"\n[{section}]\n{content[:1000]}"
                )

        context = "\n".join(context)

        prompt = f"""
You are an experienced academic researcher.

Compare all uploaded research papers.

Comparison Aspect:

{aspect}

Generate a markdown table.

Columns:

| Paper | Methodology | Dataset | Results | Advantages | Limitations |

After the table provide:

1. Similarities
2. Differences
3. Overall observations

Paper Context:

{context[:14000]}
"""

        response = self.llm.invoke(prompt)

        return response.content

    # ===========================================================
    # Research Gap Finder
    # ===========================================================

    def identify_research_gaps(self):

        context = []

        for paper, sections in self.parsed_papers.items():

            combined = ""

            combined += sections.get("Discussion & Gaps", "")
            combined += "\n"
            combined += sections.get("Conclusion", "")
            combined += "\n"
            combined += sections.get("Methodology", "")

            context.append(

                f"""
Paper:

{paper}

Content:

{combined[:2500]}
"""
            )

        prompt = f"""
You are an expert reviewer.

Based ONLY on these papers identify:

1. Common limitations

2. Common assumptions

3. Research gaps

4. Future research directions

5. Three novel research ideas

Context:

{"".join(context)}
"""

        response = self.llm.invoke(prompt)

        return response.content

    # ===========================================================
    # Retrieve Documents
    # ===========================================================

    def retrieve(
        self,
        query: str,
        section: str = "All",
        k: int = 4
    ):

        if self.vector_store is None:
            return []

        if section == "All":

            docs = self.vector_store.similarity_search(
                query,
                k=k
            )

        else:

            docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter={
                    "section": section
                }
            )

        return docs

    # ===========================================================
    # Question Answering
    # ===========================================================

    def answer_question(
        self,
        query: str,
        section="All"
    ):

        docs = self.retrieve(query, section)

        if len(docs) == 0:

            return "No relevant information found."

        context = ""

        for doc in docs:

            context += f"""
Paper:

{doc.metadata["paper"]}

Section:

{doc.metadata["section"]}

Content:

{doc.page_content}

----------------------------------------
"""

        prompt = f"""
You are a research assistant.

Answer ONLY using the supplied context.

If the answer is unavailable,
say that it was not found in the uploaded papers.

Context:

{context}

Question:

{query}

Answer in markdown.
"""

        response = self.llm.invoke(prompt)

        return response.content

    # ===========================================================
    # Statistics
    # ===========================================================

    def get_statistics(self):

        stats = {
            "papers": len(self.parsed_papers),
            "sections": 0,
            "vector_store_ready": self.vector_store is not None
        }

        for _, sections in self.parsed_papers.items():

            stats["sections"] += len(sections)

        return stats

    # ===========================================================
    # Reset
    # ===========================================================

    def reset(self):

        self.vector_store = None

        self.parsed_papers = {}
