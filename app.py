from llama_cpp import Llama
import logging
import sys
import gradio as gr

# Configuration des logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)] )
logger = logging.getLogger(__name__)

def getModelInstance(repo_id="bartowski/gemma-2-2b-it-GGUF", filename="gemma-2-2b-it-Q5_K_S.gguf",
                     n_gpu_layers=-1, context_length=4096, verbose=False):
    try:
        # Log de début de chargement
        logger.info("Début du chargement du modèle")
        logger.debug(f"Paramètres : repo_id={repo_id}, filename={filename}, n_gpu_layers={n_gpu_layers}, "
                     f"context_length={context_length}, verbose={verbose}")

        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=context_length,
            verbose=verbose,
            n_gpu_layers=n_gpu_layers
        )
        
        # Log de succès
        logger.info("Modèle chargé avec succès")
        return llm, filename

    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé : {e}")
        raise FileNotFoundError("Le fichier de modèle spécifié est introuvable. Vérifiez le nom et le chemin.")
        
    except ValueError as e:
        logger.error(f"Erreur de valeur : {e}")
        raise ValueError("Erreur dans les paramètres d'entrée. Vérifiez les valeurs des paramètres.")
        
    except Exception as e:
        logger.error(f"Erreur inconnue lors du chargement du modèle : {e}")
        raise RuntimeError("Une erreur inconnue est survenue lors du chargement du modèle.")

    finally:
        logger.info("Fin de la tentative de chargement du modèle.")

def getModelChatResponse( message, messages_history):
    llm_local = llm
    stream=False
    messages_history.append({"role":"user", "content":message})
    llm_response = llm_local.create_chat_completion(
        messages = messages_history,
        stream=False  # Active le streaming
    )
    # Afficher les tokens au fur et à mesure de leur génération façon chatbot
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
# You can see and choose your model from the trending GGUF list dataset :  
models_list = ddb.sql(
    """
	SELECT model_rank,repo_id, default_file_name, repo_url
	FROM 
	"https://huggingface.co/datasets/alihmaou/HUGGINGFACE_TRENDING_GGUFS_LIST/resolve/main/data/train-00000-of-00001.parquet"
	where model_rank <100
    order by model_rank;"""
).to_df()

# Filtrer pour model_rank = 17
filtered_model = models_list[models_list["model_rank"] == 15]
repo_id = filtered_model.iloc[0]["repo_id"]
filename = filtered_model.iloc[0]["default_file_name"]

logger.info(filtered_model)

n_ctx=4048
verbose=False
n_gpu_layers=0#-1
stream = True


messages_history = [{"role": "user","content": "### SYSTEM : Tu es simplificateur de texte, tu rends les textes plus faciles a lire et a comprendre pour les personnes légèrements déficientes mentales ou apprenante du langage français."}]
messages_history.append({"role": "assistant","content": "SET UP AND READY"})
#messages_history.append({"role": "user","content": "Apprend moi le mot 'manger' et ses conjugaisons"})

llm, filenameloaded = getModelInstance(repo_id,filename,n_gpu_layers,n_ctx,verbose)
logger.info("LLM chargé : "+filenameloaded)
#response = getModelChatResponse("hello", messages_history, llm, stream)

# Création de l'interface
with gr.Blocks() as app:
    with gr.Tabs():
        
        # Onglet "Chat"
        with gr.Tab("Chat"):
            with gr.Row():
                # Colonne du chatbot
                with gr.Column():
                    chatbot = gr.Chatbot(type="messages", show_copy_all_button=True, value=messages_history)
                    msg = gr.Textbox(label="Chatbox",placeholder="Posez ici vos questions en langage naturel sur vos données, vous pouvez contrôler le SQL à tout moment.")
                with gr.Column() as gr_chat_details:# Input pour la question
                    with gr.Tab("⌨️ Model",interactive=True):
                        gr_chat_current_Model = gr.Code(label="⌨️ Model", language="markdown", value=filename)
                    with gr.Tab("🤖 Full response", interactive=True):
                        gr_chat_current_Response = gr.Code(language="markdown",label="🤖 Full response")
            msg.submit(getModelChatResponse, [msg, chatbot], [chatbot, gr_chat_current_Response])
            # Colonne avec le Markdown pour le détail
        # Onglet "Models"
        with gr.Tab("Models Top 100"):
            gr.Dataframe(value=models_list, interactive=False, label="Top 100 des modèles (GGUF) HF")
        

# Lancement de l'application
app.launch()