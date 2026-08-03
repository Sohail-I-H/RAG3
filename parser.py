import re
import fitz  # PyMuPDF
from typing import Dict

# -----------------------------------------------------------------------------
# Academic Section Patterns
# -----------------------------------------------------------------------------

SECTION_PATTERNS = {
    "Abstract": re.compile(r"^(abstract)$", re.IGNORECASE),

    "Introduction": re.compile(
        r"^(\d+\.?\s*introduction|introduction)$",
        re.IGNORECASE
    ),

    "Related Work": re.compile(
        r"^(\d+\.?\s*(related work|background|literature review))$",
        re.IGNORECASE
    ),

    "Methodology": re.compile(
        r"^(\d+\.?\s*(methodology|methods?|proposed method|system model|architecture))$",
        re.IGNORECASE
    ),

    "Results": re.compile(
        r"^(\d+\.?\s*(results?|experiments?|evaluation|performance evaluation))$",
        re.IGNORECASE
    ),

    "Discussion & Gaps": re.compile(
        r"^(\d+\.?\s*(discussion|limitations|threats to validity))$",
        re.IGNORECASE
    ),

    "Conclusion": re.compile(
        r"^(\d+\.?\s*(conclusion|future work|conclusions))$",
        re.IGNORECASE
    ),
}


# -----------------------------------------------------------------------------
# PDF Text Extraction
# -----------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts complete text from a PDF.

    Parameters
    ----------
    pdf_path : str
        Path to PDF

    Returns
    -------
    str
        Complete extracted text
    """

    doc = fitz.open(pdf_path)

    pages = []

    for page in doc:
        pages.append(page.get_text())

    doc.close()

    return "\n".join(pages)


# -----------------------------------------------------------------------------
# Structured Section Extraction
# -----------------------------------------------------------------------------

def extract_structured_sections(pdf_path: str) -> Dict[str, str]:
    """
    Extract academic sections from a research paper.

    Returns dictionary like:

    {
        "Abstract": "...",
        "Introduction": "...",
        ...
    }
    """

    text = extract_pdf_text(pdf_path)

    lines = text.split("\n")

    sections = {
        "Abstract": "",
        "Introduction": "",
        "Related Work": "",
        "Methodology": "",
        "Results": "",
        "Discussion & Gaps": "",
        "Conclusion": "",
        "Other": ""
    }

    current_section = "Other"

    for line in lines:

        clean = line.strip()

        if not clean:
            continue

        found = False

        for section_name, pattern in SECTION_PATTERNS.items():

            # Ignore long sentences.
            # Section headings are usually short.
            if len(clean) <= 80 and pattern.match(clean):

                current_section = section_name
                found = True
                break

        if not found:
            sections[current_section] += clean + "\n"

    # Remove empty sections

    cleaned = {}

    for name, content in sections.items():

        content = content.strip()

        if len(content) > 40:
            cleaned[name] = content

    return cleaned


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def get_available_sections(sections: Dict[str, str]):
    """
    Returns only available section names.
    """

    return list(sections.keys())
