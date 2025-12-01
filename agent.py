import os
import re
import requests
from dotenv import load_dotenv
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import (
    LiteLLMModel,
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    WebSearchTool,
    tool,
    FinalAnswerTool,
    WikipediaSearchTool,
    VisitWebpageTool,
    DuckDuckGoSearchTool
)

load_dotenv()

from langfuse import get_client
langfuse = get_client()
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


from openinference.instrumentation.smolagents import SmolagentsInstrumentor
SmolagentsInstrumentor().instrument()

model = LiteLLMModel(
    model_id="openai/Qwen/Qwen3-Coder-480B-A35B-Instruct",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    api_base="https://api.tokenfactory.nebius.com/v1/"
)

from tool_clinical_trial import ClinicalTrialsSearchTool


@tool
def search_pubmed(topic: str, author: str) -> list[str]:
    """
    Searches the PubMed database for articles related to a specific topic.
    
    Args:
        topic: The topic or keywords to search for (e.g., "CRISPR gene editing").
        author: The name of the author to search for (e.g., "Albert Einstein").

    Returns:
        A list of PubMed IDs (strings) for the top 100 articles found.
        
    Raises:
        requests.exceptions.HTTPError: If the API request fails.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    terms = []
    if topic:
        terms.append(topic)
    if author:
        terms.append(f"{author}[Author]")

    query = " AND ".join(terms)
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 1000
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    return data["esearchresult"]["idlist"]

@tool
def parse_pdf(pdf_path:str)->list[str]:
    """
    Reads a PDF file from a specified path and extracts the text content
    from every page.

    Args:
        pdf_path: The local file path (string) to the PDF document to be parsed.
                  **NOTE**: In a remote agent environment, this path must be
                  accessible by the executing process (e.g., a path to an
                  uploaded file).

    Returns:
        A list of strings, where each string is the extracted text content
        from a single page of the PDF.
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    text=list()
    for p in range(number_of_pages):
        page = reader.pages[p]
        text.append(page.extract_text())
    return text



# Create clinical trial search agent
clinical_agent = CodeAgent(
    name="clinical_agent",
    description=(
        "Retrieve and parse clinical study data for a given disease. "
        "Use ClinicalTrialsSearchTool for trials, search_pubmed for authors, and parse_pdf for full-text analysis. "
        "Return structured tables or summaries as requested."
        "Gather general or recent information from online sources. "
        "Use Wikipedia for overviews, DuckDuckGo for recent data, and VisitWebpageTool for specific URLs. "
        "Return structured summaries with sources."
        "Use the ClinicalTrialsSearchTool() for any question related to clinical trial"
    ),
    tools=[ClinicalTrialsSearchTool()],
    additional_authorized_imports=["time", "numpy", "pandas"],
    # executor_type="blaxel", #executor_type="modal",
    use_structured_outputs_internally=True,
    return_full_result=True,
    planning_interval=3,                      # V3 add structure
    model=model,
    max_steps=6,
    verbosity_level=2
)

search_online_info = CodeAgent(
    name="search_online_info",
    description=(
        "Gather general or recent information from online sources. "
        "Use Wikipedia for overviews, DuckDuckGo for recent data, and VisitWebpageTool for specific URLs. "
        "Return structured summaries with sources."
    ),
    tools=[WikipediaSearchTool(),VisitWebpageTool(max_output_length=10000),DuckDuckGoSearchTool(max_results=5),search_pubmed,parse_pdf],
    additional_authorized_imports=["time", "numpy", "pandas"],
    # use_structured_outputs_internally=True,
    # executor_type="modal",
    planning_interval=2, 
    model=model,
    max_steps=4,
    verbosity_level=2
)



manager_agent = CodeAgent(
    name="manager_agent",
    description=(
    "Most important task is to provide a complete answer to user questions based on clinical trial data and online information. "
    "Orchestrate workflow between clinical and online agents. "
    "Validate outputs, resolve conflicts, and ensure the final answer is complete and accurate."
    "rimarily use the managed agent clinical_agent for question related to clinical trials"
    ),
    tools=[FinalAnswerTool(),ClinicalTrialsSearchTool()],
    model=model,
    # managed_agents=[clinical_agent,search_online_info],
    # executor_type="modal",
    provide_run_summary=True,
    additional_authorized_imports=["time", "numpy", "pandas"],
    use_structured_outputs_internally=True,
    verbosity_level=2,
    planning_interval=3, 
    max_steps=6,
)
