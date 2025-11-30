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

# Reference parser & vector store tools
from tool_create_FAISS_vector import *

# Torch device detection
import torch

# Embeddings & vector store dependencies
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List, Tuple,

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