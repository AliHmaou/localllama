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
    return llm_response['choices'][0]['message']['content']

## RUN


import duckdb as ddb
models_list = ddb.sql(
    """
	SELECT model_rank,repo_id, default_file_name, repo_url
	FROM 
	"https://huggingface.co/datasets/alihmaou/HUGGINGFACE_TRENDING_GGUFS_LIST/resolve/main/data/train-00000-of-00001.parquet"
	where model_rank <100
    order by model_rank;"""
).to_df()

repo_id=models_list[16]["repo_id"]
filename=models_list[16]["filename"]

n_ctx=4048
verbose=False
n_gpu_layers=0#-1
stream = True


messages_history = [{"role": "user","content": "### SYSTEM : Tu es un professeur de berbere marocain amazigh d'Agadir."}]
messages_history.append({"role": "assistant","content": "SET UP AND READY"})
messages_history.append({"role": "user","content": "Apprend moi le mot 'manger' et ses conjugaisons"})

llm, filenameloaded = getModelInstance(repo_id,filename,n_ctx,n_gpu_layers,verbose)
logger.info("LLM chargé : "+filenameloaded)
#response = getModelChatResponse("hello", messages_history, llm, stream)

gr.ChatInterface(getModelChatResponse, type="messages").launch()