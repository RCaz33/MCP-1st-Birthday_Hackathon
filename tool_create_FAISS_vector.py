# PDF parsing
from pypdf import PdfReader
from io import BytesIO

# HTTP requests
import requests

# Environment
import os
from dotenv import load_dotenv
load_dotenv()

# SerpAPI DOI lookup
import serpapi

# PubMed / Metapub
from metapub import FindIt
import xml.etree.ElementTree as ET

# FTP download
from ftplib import FTP
from urllib.parse import urlparse

# ArXiv
import arxiv
from langchain_community.retrievers import ArxivRetriever

# Regex
import re

# LangChain document
from langchain_core.documents import Document as LangchainDocument

# PDF parsing
from pypdf import PdfReader
from io import BytesIO

# HTTP requests
import requests

# XML parsing (PubMed FTP metadata)
import xml.etree.ElementTree as ET

# FTP download
from ftplib import FTP
from urllib.parse import urlparse

# ArXiv retrieval
import arxiv
from langchain_community.retrievers import ArxivRetriever

# PubMed → PDF resolution
from metapub import FindIt

# SerpAPI DOI search
import serpapi
import os
from dotenv import load_dotenv

load_dotenv()

def parse_pdf_file(path:str) -> str:

    if path.startswith("http://") or path.startswith("https://") or path.startswith("ftp://"):
        response = requests.get(path)
        response.raise_for_status()  # Ensure download succeeded
        reader = PdfReader(BytesIO(response.content))
    else:
        reader = PdfReader(path)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    
    return text

def get_paper_from_arxiv_id(doi: str):
    """
    Retrieve paper from arXiv using its arXiv ID.
    """
    client = arxiv.Client()
    search = arxiv.Search(query=doi, max_results=1)
    results = client.results(search)
    pdf_url = next(results).pdf_url
    text = parse_pdf_file(pdf_url)
    return text

def get_paper_from_arxiv_id_langchain(arxiv_id: str):
    """
    Retrieve paper from arXiv using its arXiv ID. ==> returns a Langchain Document
    """
    search = "2304.07814"
    retriever = ArxivRetriever(
        load_max_docs=2,
        get_full_documents=True,
    )
    docs = retriever.invoke(search)
    return docs

def get_paper_from_pmid(pmid:str):
    src = FindIt(pmid)
    if src.url:
        pdf_text = parse_pdf_file(src.url)
        return pdf_text
    else:
       print(src.reason)



def download_pdf_via_ftp(url: str) -> bytes:
    """
    Download a PDF file from an FTP URL and return its content as bytes.
    """
    parsed_url = urlparse(url)
    ftp_host = parsed_url.netloc
    ftp_path = parsed_url.path

    file_buffer = BytesIO()

    with FTP(ftp_host) as ftp:
        ftp.login() 
        ftp.retrbinary(f'RETR {ftp_path}', file_buffer.write)
            
    file_buffer.getvalue()
    file_buffer.seek(0)
    return file_buffer


def parse_pdf_from_pubmed_pmid(pmid: str) -> str:
    """
    Download and parse a PDF from PubMed using its PMID.
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmid}"
    response = requests.get(url)
    cleaned_string = response.content.decode('utf-8').strip()
    try:
        root = ET.fromstring(cleaned_string)
        pdf_link_element = root.find(".//link[@format='pdf']")
        ftp_url = pdf_link_element.get('href')
        file_byte = download_pdf_via_ftp(ftp_url)

        reader = PdfReader(file_byte)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        print(f"got {pmid} via ftp download")
        return text
    except Exception as e:
        print(e)

def download_pdf_from_url(url):
    """
    Download and extract text from a PDF URL
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get('content-type', '').lower()
    if 'pdf' not in content_type and not response.content.startswith(b'%PDF'):
        raise Exception(f"URL did not return a PDF (got {content_type})")
    
    reader = PdfReader(BytesIO(response.content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() #or ""
    return text


def download_paper_from_doi(doi):
    """
    Attempt to download paper from DOI with multiple fallback methods
    """
    # Clean DOI if it has prefix
    doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
    
    # Method 1: Try Unpaywall API (free, legal access)
    try:
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=your@email.com"
        response = requests.get(unpaywall_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('best_oa_location') and data['best_oa_location'].get('url_for_pdf'):
                pdf_url = data['best_oa_location']['url_for_pdf']
                text = download_pdf_from_url(pdf_url)
                print(f"Found PDF via Unpaywall: {pdf_url}")
                return text
    except Exception as e:
        print(f"Unpaywall failed: {e}")

def get_pdf_content_serpapi(doi: str) -> str:
    """
    Get the link to the paper from its DOI using SerpAPI Google Scholar search.
    """
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))
    results = client.search({
        'engine': 'google_scholar',
        'q': doi,
    })

    pdf_path = results["organic_results"][0]["link"]
    pdf_text = parse_pdf_file(pdf_path)
    return pdf_text



# Torch device detection
import torch

# Embeddings & vector store dependencies
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List, Tuple

# Progress bar
from tqdm import tqdm

class ReferenceExtractor:
    """Extract and classify references from LLM outputs."""
    
    # Regex patterns for identification
    DOI_PATTERN = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
    DOI_LOOSE = r"10\.\d{4,9}(?:/[-._;()/:A-Za-z0-9]+)?"
    PMID_PATTERN = r"\b\d{7,8}\b"
    ARXIV_NEW = r"\b\d{4}\.\d{4,5}(?:v\d+)?\b"
    ARXIV_OLD = r"\b[a-z\-]+/\d{7}\b"
    PMCID_PATTERN = r"\bPMC\d+\b"
    
    def __init__(self):
        """Initialize the extractor with compiled regex patterns."""
        self.patterns = {
            'doi': re.compile(f"({self.DOI_PATTERN})|({self.DOI_LOOSE})", re.IGNORECASE),
            'pmid': re.compile(self.PMID_PATTERN),
            'arxiv': re.compile(f"({self.ARXIV_NEW})|({self.ARXIV_OLD})", re.IGNORECASE),
            'pmcid': re.compile(self.PMCID_PATTERN, re.IGNORECASE)
        }
    
    def extract_references(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all references from text and classify them.
        
        Args:
            text: Input string that may contain references in various formats
            
        Returns:
            List of tuples: (reference_value, reference_type)
        """
        references = []
        seen = set()
        
        # First, try to parse as a list-like string
        list_refs = self._extract_from_list_format(text)
        if list_refs:
            for ref in list_refs:
                ref_type = self._classify_single_ref(ref)
                if ref not in seen:
                    references.append((ref, ref_type))
                    seen.add(ref)
            return references
        
        # If not a list format, extract using regex patterns
        for ref_type, pattern in self.patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                ref_value = match.group(0).strip()
                if ref_value not in seen:
                    references.append((ref_value, ref_type))
                    seen.add(ref_value)
        
        return references
    
    def _extract_from_list_format(self, text: str) -> List[str]:
        """
        Extract references from list-like formats.
        Handles: "id1,id2,id3" and '["id1","id2"]' and "['id1', 'id2']"
        """
        text = text.strip()
        
        # Try parsing as Python list string
        if text.startswith('[') and text.endswith(']'):
            try:
                # Remove brackets and quotes, split by comma
                cleaned = text[1:-1]
                # Handle both single and double quotes
                items = re.findall(r'["\']([^"\']+)["\']', cleaned)
                if items:
                    return [item.strip() for item in items]
            except:
                pass
        
        # Try comma-separated format (no brackets)
        if ',' in text and not any(char in text for char in ['\n', '(', ')']):
            # Check if it looks like a simple list
            if text.count(',') >= 1 and len(text) < 200:
                items = [item.strip().strip('"\'') for item in text.split(',')]
                # Filter out empty strings
                return [item for item in items if item]
        
        return []
    
    def _classify_single_ref(self, ref: str) -> str:
        """Classify a single extracted reference string."""
        ref = ref.strip().strip('"\'')
        
        # Check each pattern in priority order
        if re.match(r"10\.\d{4,9}(?:/[-._;()/:A-Za-z0-9]+)?", ref, re.IGNORECASE):
            return "doi"
                 
        if re.match(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", ref, re.IGNORECASE):
            return "doi"
        
        if re.match(r"^PMC\d+$", ref, re.IGNORECASE):
            return "pmcid"
        
        if re.match(r"^\d{4}\.\d{4,5}(?:v\d+)?$", ref):
            return "arxiv"
        
        if re.match(r"^[a-z\-]+/\d{7}$", ref, re.IGNORECASE):
            return "arxiv"
        
        if re.match(r"^\d{7,8}$", ref):
            return "pmid"
        
        return "unknown"
    

def process_ref(extr_ref:tuple[str,str]) -> str:
    """ router to use proper tool parser given reference type """
    if extr_ref[1] == "arxiv":
        return get_paper_from_arxiv_id(extr_ref[0])
    elif extr_ref[1] == "pmid":
        for tool in [get_paper_from_pmid, parse_pdf_from_pubmed_pmid]:
            try:
                return tool(extr_ref[0])
            except:
                continue
    elif extr_ref[1] == "doi":
        for tool in [get_pdf_content_serpapi, download_paper_from_doi]:
            try:
                return tool(extr_ref[0])
            except:
                continue
    elif extr_ref[1] == "pmcid":
        return parse_pdf_from_pubmed_pmid(extr_ref[0])
           
import torch
def get_device():
    """Detect the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Test if MPS actually works
            torch.zeros(1).to('mps')
            return 'mps'
        except:
            return 'cpu'
    return 'cpu'



from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings 
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

def create_vector_store_from_list_of_doi(refs :str, VECTOR_DB_PATH:str) -> str:

    VECTOR_DB_PATH = "./tmp/vector_stores" + VECTOR_DB_PATH
    
    from langchain_community.vectorstores import FAISS

    # define embedding
    device = get_device()

    embedding_name="BAAI/bge-small-en-v1.5"
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_name,
                                        model_kwargs={"device": device}, # set device acording to availaility
                                        encode_kwargs={"normalize_embeddings": True},)
    try:
        # Load the vector database from the folder
        print(f"try to load vector store from {VECTOR_DB_PATH}")
        KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
            VECTOR_DB_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True  # Required for security in newer LangChain versions
        )
        existing_reference = [doc.metadata.get("source") for doc in KNOWLEDGE_VECTOR_DATABASE.docstore._dict.values()]
        print("vectro store loaded")
    except Exception as e :
        print("FAISS load error:", e)
        KNOWLEDGE_VECTOR_DATABASE = None
        existing_reference = []
        print("no vector store found, creating a new one...")
        

    # fetch docs
    extractor = ReferenceExtractor()
    REFS = extractor.extract_references(refs) # Change here the type of IDs to DEBUG

    raw_docs=[]
    for ref in tqdm(REFS, disable=True):
        if ref[0] not in set(existing_reference):
            text = process_ref(ref)
            if text:
                raw_docs.append(LangchainDocument(page_content=text,metadata={'source':ref[0]}))
        
    recover_yield = f" *** -> {round(100*len(raw_docs)/len(REFS))}% papers downloaded"
    print(recover_yield)

    # split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                                AutoTokenizer.from_pretrained(embedding_name),
                                chunk_size=3000,
                                chunk_overlap=int(3000 / 10),
                                add_start_index=True,
                                strip_whitespace=True,
                                separators="."
                                )
    
    if raw_docs:
        docs_processed = text_splitter.split_documents(raw_docs)
        print("creating the vector store...")

        # create the vector store
        NEW_KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)

        if KNOWLEDGE_VECTOR_DATABASE :
            print("merge vector store")
            KNOWLEDGE_VECTOR_DATABASE.merge_from(NEW_KNOWLEDGE_VECTOR_DATABASE)
            KNOWLEDGE_VECTOR_DATABASE.save_local(VECTOR_DB_PATH)
        else:
            NEW_KNOWLEDGE_VECTOR_DATABASE.save_local(VECTOR_DB_PATH)

        return VECTOR_DB_PATH
    
    else:
        return f"all the data already in vector store {VECTOR_DB_PATH}"