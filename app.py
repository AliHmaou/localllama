from llama_cpp import Llama
import logging
import sys
import gradio as gr

# Log configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def getModelInstance(repo_id="bartowski/gemma-2-2b-it-GGUF", filename="gemma-2-2b-it-Q5_K_S.gguf",
                     n_gpu_layers=-1, context_length=4096, verbose=False):
    try:
        # Loading start log
        logger.info("Starting model loading")
        logger.debug(f"Parameters: repo_id={repo_id}, filename={filename}, n_gpu_layers={n_gpu_layers}, "
                     f"context_length={context_length}, verbose={verbose}")

        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=context_length,
            verbose=verbose,
            n_gpu_layers=n_gpu_layers
        )
        
        # Success log
        logger.info("Model loaded successfully")
        return llm, filename

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise FileNotFoundError("The specified model file is not found. Check the name and path.")
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise ValueError("Error in input parameters. Check the parameter values.")
        
    except Exception as e:
        logger.error(f"Unknown error during model loading: {e}")
        raise RuntimeError("An unknown error occurred while loading the model.")

    finally:
        logger.info("End of model loading attempt.")

def getModelChatResponse(message, messages_history):
    llm_local = llm
    stream=False
    messages_history.append({"role":"user", "content":message})
    llm_response = llm_local.create_chat_completion(
        messages=messages_history,
        stream=False  # Enables streaming
    )
    # Display tokens progressively as in a chatbot
    #if stream:
    #    nb_tokens = 0
    #    for token in llm_response:
    #        nb_tokens+=1
    #        content = token['choices'][0].get('delta', {}).get('content', '')
    #        print(content, end='', flush=True)
    llm_response_content = llm_response['choices'][0]['message']
    messages_history.append(llm_response_content)
    return messages_history, str(llm_response), 

## RUN

import duckdb as ddb
# You can see and choose your model from the trending GGUF list dataset:
models_list = ddb.sql(
    """
	SELECT model_rank, repo_id, default_file_name, repo_url
	FROM 
	"https://huggingface.co/datasets/alihmaou/HUGGINGFACE_TRENDING_GGUFS_LIST/resolve/main/data/train-00000-of-00001.parquet"
	WHERE model_rank <100
    ORDER BY model_rank;"""
).to_df()

# Set the model needed by its rank number
filtered_model = models_list[models_list["model_rank"] == 15]
repo_id = filtered_model.iloc[0]["repo_id"]
filename = filtered_model.iloc[0]["default_file_name"]

logger.info(filtered_model)

n_ctx=4048
verbose=False
n_gpu_layers=0#-1
stream = True

messages_history = [{"role": "user","content": "You are a function that converts country names to iso codes. For instance if user give you France, your answer will be FRA"}]
messages_history.append({"role": "assistant","content": "SET UP AND READY"})

llm, filenameloaded = getModelInstance(repo_id,filename,n_gpu_layers,n_ctx,verbose)
logger.info("LLM loaded: "+filenameloaded)
#response = getModelChatResponse("hello", messages_history, llm, stream)

# Creating the interface
with gr.Blocks() as app:
    with gr.Tabs():
        
        # "Chat" Tab
        with gr.Tab("Chat"):
            with gr.Row():
                # Chatbot column
                with gr.Column():
                    chatbot = gr.Chatbot(type="messages", show_copy_all_button=True, value=messages_history)
                    msg = gr.Textbox(label="Chatbox",placeholder="Ask your questions in natural language about your data. You can control SQL at any time.")
                with gr.Column() as gr_chat_details: # Input for the question
                    with gr.Tab("⌨️ Model",interactive=True):
                        gr_chat_current_Model = gr.Code(label="⌨️ Model", language="markdown", value=filename)
                    with gr.Tab("🤖 Full response", interactive=True):
                        gr_chat_current_Response = gr.Code(language="markdown",label="🤖 Full response")
            msg.submit(getModelChatResponse, [msg, chatbot], [chatbot, gr_chat_current_Response])
            # Column with Markdown for details
        # "Models" Tab
        with gr.Tab("Models Top 100"):
            gr.Dataframe(value=models_list, interactive=False, label="Top 100 GGUF Models HF")
        

# Launching the application
app.launch()