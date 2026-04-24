#pip install chromadb
#pip install transformers
#pip install accelerate
#pip install -q -U google-genai

from dotenv import load_dotenv
import os
import time
#importo librerias de chromadb
import chromadb
#
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
#importo libreria para api de gemini
from google import genai
from google.genai import types

#cargo mi variable de api key
env_path = os.path.join(os.path.dirname(__file__), ".env")  # Asegúrate de que el archivo .env esté en el mismo directorio que este script
load_dotenv(dotenv_path=env_path)  # o simplemente load_dotenv() si ejecutas desde la raíz del proyecto
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


#directorio donde se encuentran los txts procesados
txt_dir = os.path.join(os.path.dirname(__file__), "textos")
db_dir = os.path.join(os.path.dirname(__file__), "db")

#configuracion de la base de datos vectorial
chroma_client = chromadb.PersistentClient(path=db_dir) #creo un cliente de chromadb para almacenar los embeddings de manera persistente en el directorio db
collection = chroma_client.get_collection(name="chunks_embeddings") #obtengo la colección de chromadb donde voy a almacenar los embeddings generados

#configuracion del modelo de lenguaje para generar las respuestas a usuario
model_id = "LiquidAI/LFM2.5-350M"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
#   attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
#funcion para generar la respuesta a partir de la consulta del usuario, utilizando los chunks mas relevantes obtenidos a partir
#de la consulta y el modelo de lenguaje para generar una respuesta estructurada y citada con las fuentes utilizadas
def generate_response_local(query, top_k_chunks: int = 10):

    # Obtener los top-k chunks más relevantes para reducir prompt y acelerar
    top_chunks = query_embedding(query, top_k=top_k_chunks)

    # Construir prompt con template estructurado para respuestas largas y citadas
    system_prompt = (
        "Eres un asistente experto que responde de acuerdo a los fuentes proporcionadas mediante un chunk traido de un rag. Responde basándote únicamente en las fuentes indicadas; no inventes información. "
        "Si la información es insuficiente o la pregunta es muy corta o ambigua, responde 'No hay suficiente información en las fuentes' y da una explicación breve/general sobre lo que haz . "
        "Debes responder teniendo en cuenta que eres un chatbot que responde a preguntas de usuarios, por lo que tu respuesta debe ser clara, concisa y fácil de entender. "
    )

    # Formatear las fuentes (top chunks) de forma concisa
    if top_chunks:
        sources_text = "\n\n".join([f"[{c['id']}] {c['doc']}" for c in top_chunks])
    else:
        sources_text = ""

    #print(f"Top {len(top_chunks)} chunks relevantes para la consulta:\n{sources_text}\n")

    user_prompt = (
        f"Eres un asistente experto que responde de acuerdo a los fuentes proporcionadas en chunks, estos chunks son los 10 mas relacionados con la pregunta y son extraidos de un rag que ya los pondero. Responde basándote en la informaicon de las fuentes. "
        f"Si la información es insuficiente o la pregunta es muy corta o ambigua, responde 'No hay suficiente información en las fuentes' y da una explicación breve/general sobre lo que haz . "
        f"Debes responder teniendo en cuenta que eres un chatbot que responde a preguntas de usuarios, por lo que tu respuesta debe mantener una estructura clara, concisa y fácil de entender. "
        f"Pregunta: {query}\n\n"
        f"Fuentes (top {len(top_chunks)}):\n{sources_text}\n\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    encoding = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )

    # extraer tensores y moverlos al dispositivo del modelo
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # Generación en modo inferencia y sin streaming para menor overhead
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.4,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            max_new_tokens=256,
            early_stopping=False,
        )

    # Decodificar de forma segura
    if output is not None:
        try:
            text = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception:
            text = str(output)
    else:
        text = ""

    print(f"Respuesta generada: {text}")

def generate_response_api(query, top_k_chunks: int = 10):
    top_chunks = query_embedding(query, top_k=top_k_chunks)

    # Formatear las fuentes (top chunks) de forma concisa
    if top_chunks:
        sources_text = "\n\n".join([f"[{c['id']}] {c['doc']}" for c in top_chunks])
    else:
        sources_text = ""

    #print(f"Top {len(top_chunks)} chunks relevantes para la consulta:\n{sources_text}\n")

    user_prompt = (
        f"Eres un asistente experto que responde de acuerdo a los fuentes proporcionadas en chunks, estos chunks son los 10 mas relacionados con la pregunta y son extraidos de un rag que ya los pondero. Responde basándote en la informaicon de las fuentes. "
        f"Si la información es insuficiente o la pregunta es muy corta o ambigua, responde 'No hay suficiente información en las fuentes' y da una explicación breve/general sobre lo que haz . "
        f"Debes responder teniendo en cuenta que eres un chatbot que responde a preguntas de usuarios, por lo que tu respuesta debe mantener una estructura clara, concisa y fácil de entender. "
        f"Pregunta: {query}\n\n"
        f"Fuentes (top {len(top_chunks)}):\n{sources_text}\n\n"
    )

    client = genai.Client()

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=user_prompt,
        # Si descomento habilito el pensamiento del modelo, dando mejores respuesta a coste de tiempo
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high")
        ),
    )

    print(f"Respuesta generada: {response.text}")




def query_embedding(query, top_k: int = 10):
    # Devuelve una lista de dicts [{'id': id, 'doc': document}, ...] con los top_k resultados
    result = collection.query(
        query_texts=[query], #consulta de texto que quiero vectorizar y comparar con los embeddings almacenados en la colección de chromadb
        n_results=top_k,
    )
    ids = result.get('ids', [[]])[0]
    docs = result.get('documents', [[]])[0]
    distances = result.get('distances', [[]])[0]
    out = []
    for _id, _doc, _distances in zip(ids, docs, distances):
        out.append({'id': _id, 'doc': _doc})
    return out
    

def main():
    #chunk_and_embed() #lo llamo solo la primera vez para generar los chunks y embeddings y guardarlos, luego lo comento para no generar duplicados en la base de datos de chromadb
    #while True:
    #    pregunta = input("Ingrese su pregunta: ")
    #    generate_response_api(pregunta)
    #mide los tiempos de respuesta de cada metodo para comparar
    inicio = time.time()
    print("RESPUESTA MEDIANTE API DE GEMINI:")
    generate_response_api("quien es el autor o autores?")
    generate_response_api("cual es el objetivo del proyecto?")
    generate_response_api("donde queda el rio reconquista?")
    # ----------------------
    fin = time.time()
    print(f"Tiempo de respuesta de la API: {fin - inicio} segundos")
    inicio = time.time()
    print("\nRESPUESTA MEDIANTE MODELO LOCAL:")
    generate_response_local("quien es el autor o autores?")
    generate_response_local("cual es el objetivo del proyecto?")
    generate_response_local("donde queda el rio reconquista?")
    fin = time.time()
    print(f"Tiempo de respuesta del modelo local: {fin - inicio} segundos")

if __name__ == "__main__":
    main()

