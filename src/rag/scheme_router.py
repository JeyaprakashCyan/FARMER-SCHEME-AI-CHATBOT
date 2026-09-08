"""
Scheme Router
-------------
Detects farmer schemes from user questions and helps prioritize
scheme-specific documents during RAG retrieval.

This module is intentionally lightweight and deterministic.
It does NOT call an LLM.
"""

import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document


# ============================================================
# SCHEME REGISTRY
# ============================================================

SCHEME_REGISTRY: Dict[str, Dict] = {

    # --------------------------------------------------------
    # PM-KISAN
    # --------------------------------------------------------
    "pm-kisan": {
        "name": "Pradhan Mantri Kisan Samman Nidhi",
        "file_name": "Pradhan Mantri Kisan Samman Nidhi.pdf",
        "aliases": [
            "pm-kisan",
            "pm kisan",
            "pmkisan",
            "pm kisaan",
            "kisan samman nidhi",
            "kisan sammaan nidhi",
            "pm kisan samman nidhi",
        ],
    },

    # --------------------------------------------------------
    # PM-KISAN MAAN DHAN
    # --------------------------------------------------------
    "pm-kisan-maan-dhan": {
        "name": "PM Kisan Maan Dhan Yojana",
        "file_name": "PM Kisan Maan Dhan Yojana.pdf",
        "aliases": [
            "pm-kisan maan dhan",
            "pm kisan maan dhan",
            "pm kisan mandhan",
            "kisan maan dhan",
            "kisan mandhan",
            "pm kisan pension",
            "kisan pension",
            "maan dhan",
            "mandhan",
        ],
    },

    # --------------------------------------------------------
    # PMKSY
    # --------------------------------------------------------
    "pmksy": {
        "name": "Pradhan Mantri Krishi Sinchai Yojana",
        "file_name": "Pradhan Mantri Krishi Sinchai Yojana.pdf",
        "aliases": [
            "pmksy",
            "pm ksy",
            "pradhan mantri krishi sinchai yojana",
            "krishi sinchai yojana",
            "sinchai yojana",
            "irrigation scheme",
            "farm irrigation scheme",
        ],
    },

    # --------------------------------------------------------
    # AGRICULTURE INFRASTRUCTURE FUND
    # --------------------------------------------------------
    "aif": {
        "name": "Agriculture Infrastructure Fund",
        "file_name": "Agriculture Infrastructure Fund.pdf",
        "aliases": [
            "aif",
            "agriculture infrastructure fund",
            "agri infrastructure fund",
            "agriculture infrastructure",
        ],
    },

    # --------------------------------------------------------
    # AHIDF
    # --------------------------------------------------------
    "ahidf": {
        "name": "Animal Husbandry Infrastructure Development Fund",
        "file_name": "Animal Husbandry Infrastructure Development Fund.pdf",
        "aliases": [
            "ahidf",
            "animal husbandry infrastructure development fund",
            "animal husbandry infrastructure fund",
        ],
    },

    # --------------------------------------------------------
    # KCC
    # --------------------------------------------------------
    "kcc": {
        "name": "Kisan Credit Card",
        "file_name": "Credit facility for farmers.pdf",
        "aliases": [
            "kcc",
            "kisan credit card",
            "kisan credit",
            "credit facility for farmers",
            "farmers credit",
        ],
    },

    # --------------------------------------------------------
    # KCC ANIMAL HUSBANDRY / FISHERIES
    # --------------------------------------------------------
    "kcc-animal-fisheries": {
        "name": "KCC for Animal Husbandry and Fisheries",
        "file_name": "KCC for animal husbandry and fisheries.pdf",
        "aliases": [
            "kcc animal husbandry",
            "kcc for animal husbandry",
            "kcc fisheries",
            "kcc for fisheries",
            "kcc animal husbandry fisheries",
            "kisan credit card animal husbandry",
            "kisan credit card fisheries",
        ],
    },

    # --------------------------------------------------------
    # NBHM
    # --------------------------------------------------------
    "nbhm": {
        "name": "National Beekeeping and Honey Mission",
        "file_name": "National Beekeeping and Honey Mission.pdf",
        "aliases": [
            "nbhm",
            "national beekeeping and honey mission",
            "beekeeping mission",
            "honey mission",
            "beekeeping scheme",
            "honey scheme",
        ],
    },

    # --------------------------------------------------------
    # NMEO
    # --------------------------------------------------------
    "nmeo": {
        "name": "National Mission on Edible Oils",
        "file_name": "National Mission on Edible Oils .pdf",
        "aliases": [
            "nmeo",
            "national mission on edible oils",
            "edible oils mission",
            "edible oil scheme",
            "oilseeds mission",
        ],
    },

    # --------------------------------------------------------
    # NMNF
    # --------------------------------------------------------
    "nmnf": {
        "name": "National Mission on Natural Farming",
        "file_name": "National Mission on Natural Farming.pdf",
        "aliases": [
            "nmnf",
            "national mission on natural farming",
            "natural farming",
            "natural farming mission",
            "natural farming scheme",
        ],
    },

    # --------------------------------------------------------
    # RASHtriya GOKUL MISSION
    # --------------------------------------------------------
    "rgm": {
        "name": "Rashtriya Gokul Mission",
        "file_name": "Rashtriya Gokul Mission.pdf",
        "aliases": [
            "rgm",
            "rashtriya gokul mission",
            "gokul mission",
            "gokul yojana",
        ],
    },

    # --------------------------------------------------------
    # PULSES
    # --------------------------------------------------------
    "pulses": {
        "name": "Mission for Aatmanirbharta in Pulses",
        "file_name": "Mission for Aatmanirbharta in Pulses.pdf",
        "aliases": [
            "mission for aatmanirbharta in pulses",
            "aatmanirbharta in pulses",
            "pulses mission",
            "pulse mission",
            "pulses scheme",
        ],
    },

    # --------------------------------------------------------
    # AMRIT SAROVAR
    # --------------------------------------------------------
    "amrit-sarovar": {
        "name": "Mission Amrit Sarovar",
        "file_name": "Mission Amrit Sarovar.pdf",
        "aliases": [
            "mission amrit sarovar",
            "amrit sarovar",
            "sarovar mission",
        ],
    },

    # --------------------------------------------------------
    # TEA
    # --------------------------------------------------------
    "tea": {
        "name": "Tea Development & Promotion Scheme",
        "file_name": "Tea Development & Promotion Scheme.pdf",
        "aliases": [
            "tea development promotion scheme",
            "tea development",
            "tea promotion scheme",
            "tea scheme",
        ],
    },

    # --------------------------------------------------------
    # VIBRANT VILLAGES
    # --------------------------------------------------------
    "vibrant-villages": {
        "name": "Vibrant Villages Programme",
        "file_name": "Vibrant Villages Programme.pdf",
        "aliases": [
            "vibrant villages programme",
            "vibrant villages program",
            "vibrant villages",
            "vvp",
        ],
    },

    # --------------------------------------------------------
    # LIVESTOCK INSURANCE
    # IMPORTANT: added for fact/year retrieval
    # --------------------------------------------------------
    "livestock-insurance": {
        "name": "Livestock Insurance",
        "file_name": "Livestock Insurance - An introduction.pdf",
        "aliases": [
            "livestock insurance",
            "livestock insured",
            "animal insurance",
            "cattle insurance",
            "livestock insurance scheme",
            "livestock scheme",
            "animals insured",
            "insured livestock",
        ],
    },

    # --------------------------------------------------------
    # AGRICULTURE INFRASTRUCTURE / OTHER DOCUMENTS
    # --------------------------------------------------------

    "crop-insurance": {
        "name": "Crop Insurance Schemes",
        "file_name": "Crop insurance schemes.pdf",
        "aliases": [
            "crop insurance",
            "crop insurance schemes",
            "crop insurance scheme",
            "crop insurance schemes for farmers",
            "insurance for crops",
        ],
    },

    "organic-farming": {
        "name": "Financial Assistance to Organic Farmers",
        "file_name": "Financial assistance to organic farmers.pdf",
        "aliases": [
            "financial assistance to organic farmers",
            "organic farmers",
            "organic farming assistance",
            "organic farming scheme",
            "organic farming",
        ],
    },

    "fisheries-infrastructure": {
        "name": "Fisheries and Aquaculture Infrastructure Development Fund",
        "file_name": "Fisheries and Aquaculture Infrastructure Development Fund.pdf",
        "aliases": [
            "fisheries and aquaculture infrastructure development fund",
            "fisheries infrastructure development fund",
            "fisheries infrastructure",
            "aquaculture infrastructure",
            "faidf",
        ],
    },

    "fishermen-accident-insurance": {
        "name": "Group Accident Insurance Scheme for Fishermen",
        "file_name": "Group Accident Insurance scheme for Fishermen.pdf",
        "aliases": [
            "group accident insurance scheme for fishermen",
            "fishermen accident insurance",
            "accident insurance fishermen",
            "fishermen insurance",
        ],
    },

    "dairy-interest-subvention": {
        "name": "Interest Subvention for Dairy Sector",
        "file_name": "Interest subvention for dairy sector.pdf",
        "aliases": [
            "interest subvention for dairy sector",
            "dairy interest subvention",
            "interest subvention dairy",
            "dairy sector scheme",
        ],
    },

    "krishi-udan": {
        "name": "Krishi UDAN Scheme",
        "file_name": "Krishi UDAN scheme .pdf",
        "aliases": [
            "krishi udan",
            "krishi udan scheme",
            "krishi udaan",
            "krishi udaan scheme",
        ],
    },

    "mission-amrit-sarovar": {
        "name": "Mission Amrit Sarovar",
        "file_name": "Mission Amrit Sarovar.pdf",
        "aliases": [
            "mission amrit sarovar",
            "amrit sarovar mission",
        ],
    },

    "national-welfare-fishermen": {
        "name": "National Scheme of Welfare of Fishermen",
        "file_name": "National Scheme of Welfare of Fishermen.pdf",
        "aliases": [
            "national scheme of welfare of fishermen",
            "welfare of fishermen",
            "fishermen welfare scheme",
            "fishermen welfare",
        ],
    },

    "pacs": {
        "name": "Primary Agricultural Credit Societies",
        "file_name": "Primary Agricultural Credit Societies (PACS).pdf",
        "aliases": [
            "primary agricultural credit societies",
            "pacs",
            "primary agricultural credit society",
            "agricultural credit societies",
        ],
    },

    "dhan-dhaanya": {
        "name": "Prime Minister Dhan-Dhaanya Krishi Yojana",
        "file_name": "Prime Minister Dhan-Dhaanya Krishi Yojana.pdf",
        "aliases": [
            "prime minister dhan dhaanya krishi yojana",
            "dhan dhaanya krishi yojana",
            "dhan-dhaanya",
            "dhan dhaanya",
            "pm dhan dhaanya",
        ],
    },

    "tea-development": {
        "name": "Tea Development & Promotion Scheme",
        "file_name": "Tea Development & Promotion Scheme.pdf",
        "aliases": [
            "tea development promotion scheme",
            "tea development and promotion scheme",
            "tea development scheme",
        ],
    },

    "unique-package-farmers": {
        "name": "Unique Package for Farmers",
        "file_name": "Unique package for farmers .pdf",
        "aliases": [
            "unique package for farmers",
            "unique package farmers",
            "special package for farmers",
        ],
    },
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic scheme matching.

    Examples:
        PM-KISAN -> pm kisan
        PM_KISAN -> pm kisan
        PM KISAN -> pm kisan
    """

    if not text:
        return ""

    text = str(text).lower()

    # Replace common separators with spaces
    text = re.sub(r"[-_/]+", " ", text)

    # Remove remaining punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# SCHEME DETECTION
# ============================================================

def detect_scheme(question: str) -> Optional[Dict]:
    """
    Detect the most likely scheme mentioned in a question.

    Longest alias wins so that:
        "pm kisan maan dhan"
    is not incorrectly detected as:
        "pm kisan"

    Returns:
        {
            "key": "...",
            "name": "...",
            "file_name": "...",
            "matched_alias": "..."
        }

        or None
    """

    normalized_question = normalize_text(question)

    if not normalized_question:
        return None

    matches: List[Tuple[int, str, str, Dict]] = []

    for scheme_key, scheme in SCHEME_REGISTRY.items():

        for alias in scheme.get("aliases", []):

            normalized_alias = normalize_text(alias)

            if not normalized_alias:
                continue

            # Word-boundary style matching using tokenized text
            pattern = r"(?<!\w)" + re.escape(normalized_alias) + r"(?!\w)"

            if re.search(pattern, normalized_question):
                matches.append(
                    (
                        len(normalized_alias),
                        scheme_key,
                        normalized_alias,
                        scheme,
                    )
                )

    if not matches:
        return None

    # Longest matching alias wins
    matches.sort(key=lambda x: x[0], reverse=True)

    _, scheme_key, matched_alias, scheme = matches[0]

    return {
        "key": scheme_key,
        "name": scheme["name"],
        "file_name": scheme["file_name"],
        "matched_alias": matched_alias,
    }


# ============================================================
# DOCUMENT → SCHEME MATCHING
# ============================================================

def document_matches_scheme(
    document: Document,
    scheme: Optional[Dict],
) -> bool:
    """
    Check whether a LangChain Document belongs to a detected scheme.
    """

    if not scheme or document is None:
        return False

    metadata = document.metadata or {}

    target_file = normalize_text(scheme.get("file_name", ""))

    candidates = [
        metadata.get("file_name"),
        metadata.get("source"),
        metadata.get("filename"),
        metadata.get("file"),
    ]

    for candidate in candidates:

        if not candidate:
            continue

        candidate_normalized = normalize_text(str(candidate))

        if candidate_normalized == target_file:
            return True

        if target_file and target_file in candidate_normalized:
            return True

    return False


# ============================================================
# SCHEME PRIORITIZATION
# ============================================================

def prioritize_scheme_documents(
    documents: List[Document],
    scheme: Optional[Dict],
) -> List[Document]:
    """
    Move documents belonging to the detected scheme to the front.

    Original order is preserved within each group.
    """

    if not documents or not scheme:
        return documents

    matching = []
    non_matching = []

    for document in documents:

        if document_matches_scheme(document, scheme):
            matching.append(document)
        else:
            non_matching.append(document)

    # Add diagnostic metadata without modifying content
    for document in matching:
        document.metadata["scheme_match"] = True
        document.metadata["matched_scheme"] = scheme["name"]

    for document in non_matching:
        document.metadata.setdefault("scheme_match", False)

    return matching + non_matching


# ============================================================
# SCHEME BOOST
# ============================================================

def apply_scheme_boost(
    documents: List[Document],
    scheme: Optional[Dict],
) -> List[Document]:
    """
    Add a deterministic scheme_match flag to documents.

    This does not alter the document content.
    """

    if not documents:
        return documents

    for document in documents:

        is_match = document_matches_scheme(document, scheme)

        document.metadata["scheme_match"] = is_match

        if is_match and scheme:
            document.metadata["matched_scheme"] = scheme["name"]
            document.metadata["scheme_file"] = scheme["file_name"]

    return documents


# ============================================================
# FACT / YEAR QUERY DETECTION
# ============================================================

FACT_KEYWORDS = {
    "how much",
    "how many",
    "number",
    "total",
    "amount",
    "quantity",
    "percentage",
    "percent",
    "rate",
    "cost",
    "fund",
    "funding",
    "beneficiaries",
    "beneficiary",
    "insured",
    "insurance",
    "coverage",
    "covered",
    "target",
    "achieved",
    "allocated",
    "spent",
    "sanctioned",
    "approved",
    "financial year",
    "fy",
    "year",
}


def is_fact_or_year_question(question: str) -> bool:
    """
    Detect questions that are likely asking for a specific fact,
    number, amount, percentage, date, or year-specific statistic.

    This is intentionally broad because these queries benefit
    from additional lexical retrieval.
    """

    normalized = normalize_text(question)

    if not normalized:
        return False

    # Financial year patterns
    year_patterns = [
        r"\b20\d{2}\s*[-/]\s*\d{2}\b",
        r"\b20\d{2}\s*[-/]\s*20\d{2}\b",
        r"\bfy\s*20\d{2}\b",
        r"\bfinancial year\b",
    ]

    for pattern in year_patterns:
        if re.search(pattern, normalized):
            return True

    # Numeric question
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:lakh|crore|million|billion|%|percent)?\b", normalized):
        return True

    # Fact keywords
    for keyword in FACT_KEYWORDS:

        if keyword in normalized:
            return True

    return False


# ============================================================
# FACT QUERY EXPANSION
# ============================================================

def build_fact_query_variants(
    question: str,
    scheme: Optional[Dict] = None,
) -> List[str]:
    """
    Build additional semantic/lexical retrieval queries for
    fact and year questions.

    The goal is NOT to invent the answer.

    The queries simply increase the probability that the relevant
    fact-bearing page/chunk enters the reranking stage.
    """

    normalized = normalize_text(question)

    variants = []

    # Original question first
    variants.append(question)

    # --------------------------------------------------------
    # Year extraction
    # --------------------------------------------------------

    years = re.findall(
        r"\b20\d{2}\s*[-/]\s*(?:20)?\d{2}\b",
        normalized,
    )

    year_text = " ".join(years)

    # --------------------------------------------------------
    # Scheme name
    # --------------------------------------------------------

    scheme_name = ""

    if scheme:
        scheme_name = scheme.get("name", "")

    # --------------------------------------------------------
    # Generic fact query
    # --------------------------------------------------------

    if scheme_name and year_text:

        variants.append(
            f"{scheme_name} financial year {year_text} "
            f"total number beneficiaries insured covered amount percentage"
        )

        variants.append(
            f"{scheme_name} {year_text} "
            f"total insured number quantity coverage"
        )

    elif scheme_name:

        variants.append(
            f"{scheme_name} total number amount percentage beneficiaries "
            f"insured coverage"
        )

    elif year_text:

        variants.append(
            f"financial year {year_text} total number amount percentage "
            f"beneficiaries insured coverage"
        )

    # --------------------------------------------------------
    # Specific livestock insurance expansion
    # --------------------------------------------------------

    if scheme and scheme.get("key") == "livestock-insurance":

        if year_text:

            variants.append(
                f"Livestock Insurance livestock insured "
                f"financial year {year_text} total"
            )

            variants.append(
                f"Livestock Insurance number of livestock insured "
                f"{year_text}"
            )

        else:

            variants.append(
                "Livestock Insurance livestock insured total number"
            )

            variants.append(
                "Livestock Insurance animals insured number coverage"
            )

    # --------------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------------

    unique_variants = []

    seen = set()

    for variant in variants:

        key = normalize_text(variant)

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique_variants.append(variant)

    return unique_variants


# ============================================================
# DEBUG HELPERS
# ============================================================

def print_scheme_detection(question: str) -> None:
    """
    Print scheme detection information.
    """

    scheme = detect_scheme(question)

    print()
    print("=" * 60)
    print("SCHEME DETECTION")
    print("=" * 60)

    print(f"Question : {question}")

    if scheme:
        print(f"Scheme   : {scheme['name']}")
        print(f"File     : {scheme['file_name']}")
        print(f"Alias    : {scheme['matched_alias']}")
    else:
        print("Scheme   : None")

    print(
        f"Fact/Year Query : "
        f"{is_fact_or_year_question(question)}"
    )

    print("=" * 60)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_questions = [
        "What is PM-KISAN?",
        "What are the benefits of PM Kisan?",
        "How does KCC work?",
        "Tell me about AIF",
        "What is natural farming?",
        "How much livestock insured in financial year (2024-25)?",
        "How many animals were insured in 2024-25?",
        "What is the weather today?",
    ]

    for question in test_questions:

        print_scheme_detection(question)

        scheme = detect_scheme(question)

        variants = build_fact_query_variants(
            question,
            scheme,
        )

        print("Retrieval variants:")

        for index, variant in enumerate(variants, start=1):
            print(f"{index}. {variant}")

    print()
    print("✓ SCHEME ROUTER TEST COMPLETE")