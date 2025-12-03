from agent import manager_agent
import gradio as gr
import smolagents
import json
import re
import ast
import requests 
from PIL import Image



# Guard function to insure clinical trials topic (allows bypassing a call to the agent)
def is_clinical_question(query: str) -> bool:
    keywords = [
        "clinical", "trial", "efficacy", "phase I", "phase II", "phase III", "study", "studies", 
        "protocol", "adverse event", "safety", "randomized", "placebo",
        "biomarker", "endpoint", "inclusion", "exclusion"
    ]
    q = query.lower()
    return any(kw in q for kw in keywords)

# Use logging to debug
import logging
from datetime import datetime
now = datetime.utcnow().isoformat()
logging.info(f"Processing request {now}")

# Use langfuse to log traces
from langfuse import Langfuse, get_client, propagate_attributes
# langfuse = Langfuse(environment='Dev888')
# langfuse = get_client()
 
# --- PATCH --- In order to be able to stream Agent intenal steps to Gradio interface
# --- OpenTelemetry detach bug (generator-safe) ---
from opentelemetry.context import _RUNTIME_CONTEXT
_orig_detach = _RUNTIME_CONTEXT.detach
def _safe_detach(token):
    try:
        _orig_detach(token)
    except Exception:
        pass
_RUNTIME_CONTEXT.detach = _safe_detach
# --- OpenInference NonRecordingSpan bug ---
try:
    from openinference.instrumentation.smolagents import _wrappers
    
    _orig_finalize = _wrappers._finalize_step_span
    def _safe_finalize_step_span(span, step_log):
        # Check if span has status attribute before accessing it
        if hasattr(span, 'status') and hasattr(span.status, 'status_code'):
            return _orig_finalize(span, step_log)
        # For NonRecordingSpan, just skip finalization
        return None
    
    _wrappers._finalize_step_span = _safe_finalize_step_span
except ImportError:
    pass  # OpenInference not installed
# --- END PATCH ---


# Define Agent, 
def Agent(question, history):
    """Use a smolagent CodeAgent with tools to answer a question.
    The agent streams its thought process (planning steps) and the final answer.
    Args:
        question (str): The question to be answered by the agent.
    Yields:
        tuple(str, str, str): A tuple containing the current 'thoughts' (planning/intermediate steps)
                         ,the current 'final_answer', the history
    """
    
    if not is_clinical_question(question):
        err_msg = 'A Clinical trial keyword mas missing (e.g. "clinical", "study", "studies", "trial", "efficacy", "phase I", "phase II", "phase III",\
    "protocol", "adverse event", "safety", "randomized", "placebo","biomarker", "endpoint", "inclusion", "exclusion"'
        history.append({"question": question, "answer": err_msg})
        yield "I can only answer questions related to clinical studies.",err_msg,history
               
        return
    thoughts = ""
    final_answer = ""



    try:
        logging.info(f"Received question: {question}")
        # append history to the next question
        question_with_history = "Conversation history:\n" + str(history) + "\n\nNew user question:\n " + question

        with propagate_attributes(tags=["Development","Code Agent","Q4"],
                                  user_id="DEV_xx",
                                  session_id=f"Dev_codagent"):
            for st in manager_agent.run(question_with_history,stream=True,return_full_result=True):
                if isinstance(st, smolagents.memory.PlanningStep):
                    plan = 20*"# " + "\n# Planning " + st.plan.split("## 2. Plan")[-1]
                    for m in plan.split("\n"):
                        thoughts += "\n" + m
                        yield thoughts, final_answer, history
                        
                elif isinstance(st,  smolagents.memory.ToolCall):
                    code = 20*"-" + f"\n{st.name}\n\n" + st.dict()['function']['arguments']+ "\n"+ 20*"-"
                    for m in code.split("\n"):
                        thoughts += "\n" + m
                        yield thoughts, final_answer, history

                elif isinstance(st,  smolagents.agents.ActionOutput):
                    if not st.output:
                        thoughts +=  "\n\n\n****************\nNo output from action.\n****************\n\n"
                        yield thoughts, final_answer, history
                    else:
                        thoughts +=    "\n***********\nNow processing the output of the tool\n***********\n\n"
                        yield thoughts, final_answer, history

                elif isinstance(st,  smolagents.memory.ActionStep):
                    for chatmessage in st.model_input_messages:
                        if chatmessage.role == "assistant":
                            thoughts += "Agent plan:\n"
                            managed_agent_plan = chatmessage.content[0]['text'].split("2. Plan")[-1]
                            for l in managed_agent_plan.split("\n"):
                                thoughts += l
                            thoughts += "\n\n--> Action from agent \n" + (st.code_action if st.code_action else "") +"\n\n"
                            yield thoughts, final_answer, history
                    thoughts += "\n********** End fo Step " + str(st.step_number) + " : *********\n" + str(st.token_usage) + "\nStep duration" + str(st.timing) + "\n\n"

                    yield thoughts, final_answer, history

                elif isinstance(st, smolagents.memory.FinalAnswerStep):
                    final_answer = st.output
                    history.append({"question": question, "answer": final_answer})
                    yield thoughts, final_answer, history



    except GeneratorExit:
        print("Stream closed cleanly.")
        return "","", ""
    
    except gr.CancelledError:
        print("Request cancelled")
        return "Request cancelled","Submit new request", ""


# Set up clinical trial MCP tool with structured TOON output
def tool_clinical_trial(query_cond:str=None, query_term:str=None,query_lead:str=None,max_results: str="5") -> str:
    """
    Search Clinical Trials database for trials with 4 arguments.
    
    Args:
        query_cond (str): Disease or condition (e.g., 'lung cancer', 'diabetes')
        query_term (str): Other terms such as exact ID "NCTxxxxxxxx" or (e.g., 'AREA[LastUpdatePostDate]RANGE[2023-01-15,MAX]').
        query_lead (str): Searches the LeadSponsorName
        max_results (int): Number of trials to return (max: 1000)
    
    Returns:
        list(str): each string being a structured representation of a trial.
    """
    from tool_TOON_formater import TOON_formater

    try:
        max_results = int(max_results)
    except:
        max_results = 5
        
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
        stream = ""
        for i, study in enumerate(studies):
            structured_data = TOON_formater(study)
            structured_trials.append(structured_data)

            for l in structured_data.split("\n"):
                stream += "\n\n"+ 10*"--" + "\n" + structured_data
                yield stream

        return structured_trials
        
    except Exception as e:
        return [f"Error searching clinical trials: {str(e)}"]
    

# Set up RAG
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

# Use RAG
def use_rag(query: str, store_name: str, top_k: int = 5) -> str:
    """Retrieve context from a FAISS vector store based on a query.
    Args:
        query (str): The question or query string to use for retrieval.
        store_name (str): The path to the FAISS vector store to query.
        top_k (int): The number of top-k most relevant context documents to retrieve (default: 5).
    Returns:
        str: A TOON formated string containing the retrieved contexts, including the contents, the source and the scores.
    """
    from tool_query_FAISS_vector import query_vector_store
    context_as_dict = query_vector_store(query, store_name, top_k)
    return str(context_as_dict)

# Describe a figure with Gemini
def describe_figure(image : Image.Image) -> str:
    """Provide a detailed, thorough description of an image figure.
    Args:
        image (Image.Image): The image to describe.
    Returns:
        description (str): A detailed textual description of the figure's content.
    """
    from tool_describe_figure import thorough_picture_description
    description = thorough_picture_description(image)
    return description

# Create neat interface - Question Analyzer as a Blocks component
with gr.Blocks() as interface1:
    with gr.Row():
        with gr.Column():
            gr.Markdown("""## Gradio Application
    The server exposes a multi-tab Gradio interface for interactive operations:
    1. Question Analyzer
    Streams LLM reasoning steps, tool invocations, and intermediate thoughts
    Displays final, aggregated answer from the manager agent
    2. Create RAG Vector Store
    Upload list of DOIs/PMIDs
    Outputs path to stored FAISS index
    3. Query RAG Vector Store
    Retrieve contextual scientific documents for any question
    4. Clinical Trial Query Interface
    Calls the ClinicalTrials.gov API
    Returns TOON-formatted study objects
    5. Figure Description
    Accepts image uploads
    Produces detailed biomedical-quality captions""")
        with gr.Column():
            gr.Markdown("""## Summary
    This MCP server is a full-stack multi-agent research system with:
    Hierarchical LLM planning
    Dedicated scientific and clinical tools
    Real-time execution monitoring
    FAISS-based custom RAG infrastructure
    Integrated web search and document extraction
    A complete interactive UI for researchers or clinicians
    It is suitable for:
    Clinical evidence synthesis
    Scientific research workflows
    Medical question answering
    Literature reviews
    Automated extraction pipelines""")

with gr.Blocks() as interface2:
    gr.Markdown("# Question Analyzer")
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Question", 
                placeholder="Enter your question here...",
                lines=3,
            )
            gr.Examples(["What is the weather in LA?",
                         "What are the 5 most recent clinical study sponsored by Merck?",
                         "How many trials were completed in 2025 by AbbVie?",
                         "What are the pmids associated with the study NCT04516746?",],question_input)
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                stop_btn   = gr.Button("Stop", variant="secondary") 
            response_output = gr.Textbox(
                label="Final Answer", 
                placeholder="Copy/paste this output to the next tab 'Create RAG tool with FAISS vector store",
                interactive=False, 
                lines=8
            )
        with gr.Column():
            thoughts_output = gr.Textbox(
                label="LLM Thoughts/Reasoning", 
                placeholder="The Agent reasonning will be outputed here, use it to track its validity",
                interactive=False, 
                lines=8
            )

    chat_history = gr.State([]) # Necessary to save conversation history

    submit_evt = submit_btn.click(
        fn=Agent,
        inputs=[question_input, chat_history],
        outputs=[thoughts_output, response_output, chat_history],
        queue=True
    )
    stop_btn.click(fn=None, cancels=[submit_evt])

with gr.Blocks() as interface3:
    with gr.Row():
        with gr.Column():
            gr.Markdown('### Create a vector store. You can specify an existing vector store to update it')
            ref_input = gr.Textbox(
                label="List of references to include in vector store",
                lines=2,
                info="(can be DOIs, PMIDs, arXivs, ... and a mix of it)"
            )
            vector_name_input = gr.Textbox(
                label="Name of the vector store",
                lines=2,
                placeholder="My_Diabetes_vector"
            )
        with gr.Column():
            gr.Examples(
                examples=[
                    ["I found the folowing PMID: 40445502, 40050939, 29884191", "Ex1_vector_store"],
                    ["2501.16868, 29641991, PMC5034499, 10.1007","Ex2_vector_store" ]
                ],
                inputs=[ref_input, vector_name_input]
            )
            path_output = gr.Textbox(
                label="Path of the vector store",
                lines=4
            )
            submit_btn = gr.Button("Create Vector Store")
            submit_btn.click(
                fn=create_rag,
                inputs=[ref_input, vector_name_input],
                outputs=path_output,
                queue=True
            )

# Combined interfaces with tabs
demo = gr.TabbedInterface(
    [interface1,
        interface2, 
     interface3,
         gr.Interface(
            fn=use_rag, 
            inputs=[gr.Textbox(label="Question that needs context to answer", placeholder="What is the dose of medicine to gove an infant under type2 diabetes"), 
                    gr.Textbox(label="Name of the vector store to use", placeholder="Diabetes, Sickel_cell_anemia, Prostate_cancer, ..")], 
            outputs=gr.Textbox(label="Answer with Rag",lines=8, placeholder="Your answer will be provided here"),
            api_name="use_vector_store_to_create_context"),
         gr.Interface(
            fn=tool_clinical_trial, 
            inputs=[gr.Textbox(label="Disease or condition (e.g., 'lung cancer', 'diabetes')",placeholder="Diabetes"), 
                    gr.Textbox(label="Other terms (e.g., 'AREA[LastUpdatePostDate]RANGE[2023-01-15,MAX]'", placeholder=""),
                    gr.Textbox(label="Searches the LeadSponsorName",placeholder="Lilly OR Sanofi"),
                    gr.Textbox(label="Max results to retreive",placeholder=50)], 
            outputs=gr.Textbox(label="TOON formated response",lines=10, placeholder="Your answer will be provided here"),
            api_name="use_clinical_trial_to_create_context"),
        gr.Interface(
            describe_figure, 
            gr.Image(type="pil"), 
            gr.Textbox(), 
            api_name="figure_description"),
    ],
    ["Description",
    "Use a code agent with sandbox execution equiped with clinical trial tool", 
    "Create RAG tool with FAISS vector store",
    "Query RAG tool",
    "Query clinical trial database",
    "Thourough figure description",]
)

# Allows MCP
if __name__ == "__main__":
    demo.queue().launch(mcp_server=True)