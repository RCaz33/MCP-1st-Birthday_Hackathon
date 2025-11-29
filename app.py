from agent import manager_agent
import gradio as gr
from smolagents import stream_to_gradio
import smolagents
import json
import re
import ast

agent = manager_agent


import logging
logging.info("Processing request")


# --- PATCH OpenTelemetry detach bug (generator-safe) ---
from opentelemetry.context import _RUNTIME_CONTEXT
_orig_detach = _RUNTIME_CONTEXT.detach
def _safe_detach(token):
    try:
        _orig_detach(token)
    except Exception:
        # Suppress context-var boundary errors caused by streamed generators
        pass
_RUNTIME_CONTEXT.detach = _safe_detach
# --- PATCH OpenTelemetry detach bug (generator-safe) ---


def answer_question(question):
    """Use a smolagent CodeAgent with tools to answer a question.
    The agent streams its thought process (planning steps) and the final answer.
    Args:
        question (str): The question to be answered by the agent.
    Yields:
        tuple(str, str): A tuple containing the current 'thoughts' (planning/intermediate steps)
                         and the current 'final_answer'.
    """
    thoughts = ""
    final_answer = ""
    n_tokens =0
    try:
        logging.info(f"Received question: {question}")
        for st in manager_agent.run(question,stream=True,return_full_result=True):
            if isinstance(st, smolagents.memory.PlanningStep):
                plan = st.model_output_message.content.split("## 2.")[-1]
                for m in plan.split("\n"):
                    thoughts += "\n" + m
                    yield thoughts, final_answer

            elif isinstance(st,  smolagents.memory.ToolCall):
                thoughts += f"\nTool called: {st.dict()['function']['name']}\n"
                for m in st.dict()['function']['arguments'].split("\n"):
                    thoughts += "\n" + m
                    yield thoughts, final_answer

            elif isinstance(st,  smolagents.agents.ActionOutput):
                if st.output:
                    thoughts += "\n" + str(st.output) + "\n"
                    yield thoughts, final_answer
                else:
                    thoughts += "\n****************\nNo output from action.\n****************\n"
                    yield thoughts, final_answer

            elif isinstance(st,  smolagents.memory.ActionStep):
                
                for m in st.model_output_message.content.split("\n"):
                    thoughts += m
                    yield thoughts, final_answer

                thoughts += "\n********** End fo Step " + str(st.step_number) + " : *********\n " + str(st.token_usage) + "\nStep duration" + str(st.timing) + "\n\n"
                yield thoughts, final_answer

            elif isinstance(st, smolagents.memory.FinalAnswerStep):
                final_answer = st.output
                yield thoughts, final_answer
    except GeneratorExit:
        print("Stream closed cleanly.")
        return "",""
    


# def create_rag_files(refs :list[str], VECTOR_DB_PATH:str)-> str:
#     from tool_create_FAISS_vector import create_vector_store_from_list_of_doi

#     FAISS_VECTOR_PATH = create_vector_store_from_list_of_doi(refs,VECTOR_DB_PATH)
#     return FAISS_VECTOR_PATH

def tool_clinical_trial(query_cond:str=None, query_term:str=None,query_lead:str=None,max_results: int = 5000) -> list:
    """
    Search Clinical Trials database for trials with 4 arguments.
    
    Args:
        query_cond (str): Disease or condition (e.g., 'lung cancer', 'diabetes')
        query_term (str): Other terms (e.g., 'AREA[LastUpdatePostDate]RANGE[2023-01-15,MAX]').
        query_lead (str): Searches the LeadSponsorName
        max_results (int): Number of trials to return (max: 1000)
    
    Returns:
        list(str): each string being a structured representation of a trial.
    """
    from tool_TOON_formater import TOON_formater
    try:
        max_results = int(max_results)
    except:
        max_results = 500
        
    params = {
        "query.cond": query_cond,
        "query.term":query_term,
        "query.lead":query_lead,
        "pageSize": min(max_results, 5000),
        "format": "json"
    }
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        studies = response.json().get("studies", [])
        
        structured_trials = []
        for i, study in enumerate(studies):
            structured_data = TOON_formater(study)
            structured_trials.append(structured_data)

        return structured_trials
        
    except Exception as e:
        return [f"Error searching clinical trials: {str(e)}"]
    


def create_rag(refs :str, VECTOR_DB_PATH:str)-> str:
    """Create a RAG (Retrieval-Augmented Generation) vector store from a list of DOIs.
    Args:
        refs (str): A comma-separated string of DOIs (Digital Object Identifiers).
        VECTOR_DB_PATH (str): The local path where the FAISS vector store should be saved.
    Returns:
        str: The path to the newly created FAISS vector store.
    """
    from tool_create_FAISS_vector import create_vector_store_from_list_of_doi
    FAISS_VECTOR_PATH = create_vector_store_from_list_of_doi(refs,VECTOR_DB_PATH)
    return FAISS_VECTOR_PATH



def use_rag(query: str, store_name: str, top_k: int = 5) -> str:
    """Retrieve context from a FAISS vector store based on a query.
    Args:
        query (str): The question or query string to use for retrieval.
        store_name (str): The path to the FAISS vector store to query.
        top_k (int): The number of top-k most relevant context documents to retrieve (default: 5).
    Returns:
        str: A JSON string containing the retrieved context, including the content and source (DOI).
    """
    from tool_query_FAISS_vector import query_vector_store
    context_as_dict = query_vector_store(query, store_name, top_k)
    return json.dumps(context_as_dict, indent=2)

from PIL import Image

def describe_figure(figure : Image) -> str:
    """Provide a detailed, thorough description of an image figure.
    Args:
        figure (Image): The image figure object (from PIL) to be described.
    Returns:
        description (str): A detailed textual description of the figure's content.
    """
    from tool_describe_figure import thourough_picture_description
    description = thourough_picture_description(figure)
    return description



# Create neat interface - Question Analyzer as a Blocks component
with gr.Blocks() as interface2:
    gr.Markdown("# Question Analyzer")
    gr.Markdown("""Enter a question to analyze. Examples:
    - Find the name of the sponsor that did the most studies on Alzheimer's disease in the last 10 years.
    - Provide a summary of recent clinical trials on diabetes and list 3 relevant research articles from PubMed.
    - What are the scientific paper linked to the clinical study referenced as NCT04516746?
    - How many clinical studies on cancer were completed in the last 5 years?
    - Find recent phase 3 trials for lung cancer sponsored by Pfizer
    """)
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Question", 
                placeholder="Enter your question here...",
                lines=3,
            )
            submit_btn = gr.Button("Submit", variant="primary")
            response_output = gr.Textbox(
                label="Final Answer", 
                interactive=False, 
                lines=8
            )
        with gr.Column():
            thoughts_output = gr.Textbox(
                label="LLM Thoughts/Reasoning", 
                interactive=False, 
                lines=8
            )

    
    chat_history = gr.State([])
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[thoughts_output, response_output],
        queue=True
    )


# Combine interfaces into a single tabbed interface
demo = gr.TabbedInterface(
    [interface2, 
     gr.Interface(
         fn=create_rag, 
         inputs=[gr.Textbox("list of references to include in vector store",lines=2, info="(can be DOIs, PMIDs, erxivs, ... and a mix of it)"),
                 gr.Textbox("Name of the vactore store", lines=2, placeholder="My_Diabetes_vector") ],
                 outputs=gr.Textbox("path of the vactore store"),
            api_name="create_vector_store_for_rag"),
        
         gr.Interface(
            fn=use_rag, 
            inputs=[gr.Textbox("question that needs context to answer"), 
                    gr.Textbox("Name of the vector store to use", placeholder="Diabetes, Sickel_cell_anemia, Prostate_cancer, ..")], 
            outputs=gr.Textbox("Answer with Rag"),
            api_name="use_vector_store_to_create_context"),
         gr.Interface(
            fn=tool_clinical_trial, 
            inputs=[gr.Textbox("Disease or condition (e.g., 'lung cancer', 'diabetes')"), 
                    gr.Textbox("Other terms (e.g., 'AREA[LastUpdatePostDate]RANGE[2023-01-15,MAX]'"),
                    gr.Textbox("Searches the LeadSponsorName"),
                    gr.Textbox("max results")], 
            outputs=gr.Textbox("TOON formated response"),
            api_name="use_vector_store_to_create_context"),
        gr.Interface(
            describe_figure, 
            gr.Image(type="pil"), 
            gr.Textbox(), 
            api_name="figure_description"),
    ],
    ["Use a code agent with sandbox execution equiped with clinical trial tool", 
    "Create RAG tool with FAISS vector store",
    "Query RAG tool",
    "Query clinical trial database"
    "Thourough figure description",]
)

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True)