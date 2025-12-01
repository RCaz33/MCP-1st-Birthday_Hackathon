

from langchain_community.embeddings import HuggingFaceEmbeddings 


import os



def query_vector_store(query: str, store_name: str, top_k: int = 5) -> dict:
    """
    Query a specific vector store to retreive top_k documents related to the user question. 
    Each document have metadata that is the identification of the source, it must be said clearly.

    
    Args:
        query (str): User's question
        store_name (str): Which vector store to search
        top_k (int): Number of chunks to retrieve
    
    Returns:
        dict: Retrieved context, sources, store_name
    """
    from langchain_community.vectorstores import FAISS

    VECTOR_STORE_DIRECTORY = "./tmp/vector_stores"
    if not os.path.exists(VECTOR_STORE_DIRECTORY):
        os.makedirs(VECTOR_STORE_DIRECTORY)
    vector_stores = os.listdir(VECTOR_STORE_DIRECTORY)
    print(vector_stores)
    store_path = f"{VECTOR_STORE_DIRECTORY}/{store_name}"
    if store_name not in vector_stores:
        return {"error": f"Vector store '{store_name}' not found, you must create it first with tool create faiss vector"}
    

    embedding_name="BAAI/bge-small-en-v1.5"
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_name,
                                        model_kwargs={"device": "mps"},
                                        encode_kwargs={"normalize_embeddings": True},)


    vector_store = FAISS.load_local(
        store_path, 
        embedding_model, 
        allow_dangerous_deserialization=True  
        )

    results = vector_store.similarity_search(query, top_k)
    
    context = [r.page_content for r in results]
    sources = [r.metadata["source"] for r in results]
    
    return {
        "context": context,
        "sources": sources,
        "store_name": store_name
    }