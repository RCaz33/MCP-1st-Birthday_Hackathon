# Use Gradio Interface
Use the specialized code agent from gradio interface [track agent reasonning]
* Must Add NEBIUS_API_KEY secrets to the huggingface space in order to use the Agent
* Add LANGFUSE secrets to track

# Set-up MCP tools for a client (huggingchat)

## 1. Server side : Connect to the space and click "Utiliser via API"
![alt text](imgs/Step_1-2.jpg)
## 2. Client side : Select "Manage MCP server" on huggingchat


# --------------------------------------------------------


## 3. Server side : Choose communication type (MCP streamable HTTP)
![alt text](imgs/Step_3-4.jpg)
## 4. Client side : Click on "Add Server"


# ---------------------------------------------------------


## 5. Server side : Copy the link for the client side (MCP streamable HTTP)
![alt text](imgs/Step_5-6.jpg)
## 6. Client side : Paste the link Click in "Server URL"


# ----------------------------------------------------------




# Chat with data
 
## 7. Query LLM Without/With MCP tool changes the actions/output
## 8. The MCP tools can be called multiple times (x6) in a single request
![alt text](imgs/chat_with_data.jpg)