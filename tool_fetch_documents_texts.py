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

