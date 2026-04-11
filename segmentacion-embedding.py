#pip  install  langchain-text-splitters
#pip install -qU langchain-huggingface
#pip install sentence-transformers
#pip install -U "langchain-core"
#pip install chromadb
#pip install transformers
#pip install accelerate

#importo las librerias de langchain para generar chunks, embeddings y almacenar los embeddings en un vector store en memoria local
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os
#importo librerias de chromadb
import chromadb
from chromadb.config import Settings
#
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

#directorio donde se encuentran los txts procesados
txt_dir = os.path.join(os.path.dirname(__file__), "textos")
db_dir = os.path.join(os.path.dirname(__file__), "db")

# Configuración del splitter
# chunk_size = 200 tokens
# chunk_overlap = 20 tokens (10% de 200)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, #cantidad maxima de caracteres por chunk
    chunk_overlap=20, #cantidad de solapamiento entre chunks (20 caracteres)
    length_function=len,  # aquí usamos len, es la deafult
)   

#configuracion de la base de datos vectorial
chroma_client = chromadb.PersistentClient(path=db_dir) #creo un cliente de chromadb para almacenar los embeddings de manera persistente en el directorio db
collection = chroma_client.get_collection(name="chunks_embeddings") #obtengo la colección de chromadb donde voy a almacenar los embeddings generados, si ya existe la colección, la obtengo, sino la creo

#configuracion del modelo de lenguaje para generar las respuestas a usuario
model_id = "LiquidAI/LFM2.5-350M"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
#   attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Note: streaming increases overhead. we won't stream tokens for faster inference.

#funcion para generar la respuesta a partir de la consulta del usuario, utilizando los chunks mas relevantes obtenidos a partir
#de la consulta y el modelo de lenguaje para generar una respuesta estructurada y citada con las fuentes utilizadas
def generate_response(query, top_k_chunks: int = 5):
    # Obtener los top-k chunks más relevantes para reducir prompt y acelerar
    top_chunks = query_embedding(query, top_k=top_k_chunks)

    # Construir prompt con template estructurado para respuestas largas y citadas
    system_prompt = (
        "Eres un asistente experto. Responde basándote únicamente en las fuentes indicadas; no inventes información. "
        "Si la información es insuficiente, responde 'No hay suficiente información en las fuentes'. "
        "Estructura la respuesta así:\n"
        "Primero un pequeño resumen ejecutivo (2–3 frases).\n"
        "Luego desarrolla en 4 párrafos (~100–150 palabras cada uno), con ejemplos.\n"
        "Finalmente, una conclusión breve.\n"
        "Debajo de todo las Fuentes: lista con IDs citados al final de cada párrafo.\n"
        "Entrega la respuesta en un formato de texto en parrafos. Al final, entre corchetes, indica el/los ID(s) de chunk usados."
    )

    # Formatear las fuentes (top chunks) de forma concisa
    if top_chunks:
        sources_text = "\n\n".join([f"[{c['id']}] {c['doc']}" for c in top_chunks])
    else:
        sources_text = ""

    print(f"Top {len(top_chunks)} chunks relevantes para la consulta:\n{sources_text}\n")

    user_prompt = (
        f"Fuentes (top {len(top_chunks)}):\n{sources_text}\n\n"
        f"Pregunta: {query}\n\n"
        f"Responde siguiendo la estructura solicitada en el system prompt: {system_prompt}"
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
            do_sample=True,
            temperature=0.2,
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

def chunk_and_embed():
    collection.delete() #elimino la colección de chromadb para evitar duplicados, solo lo hago la primera vez que genero los chunks y embeddings, luego lo comento para no eliminar los datos ya almacenados
    collection = chroma_client.create_collection(name="chunks_embeddings") #creo la colección de chromadb donde voy a almacenar los embeddings generados, si ya existe la colección, la creo, sino la obtengo
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #objeto encargado de generar los embeddings de 384 dimensiones utilizando el modelo "all-MiniLM-L6-v2" de HuggingFace(https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    id_archivo = 0
    for filename in os.listdir(txt_dir): #itero por cada txt en el directorio
        id_archivo += 1
        if filename.lower().endswith(".txt"):
            txt_path = os.path.join(txt_dir, os.path.splitext(filename)[0] + ".txt") #genero el path del txt donde voy a escribir
            with open(txt_path, "r", encoding="utf-8") as txt_file:
                texto = txt_file.read() #leo el archivo txt completo
                chunks = splitter.split_text(texto) #genero los chunks con las propiedades ya configuradas
                print(f"---Archivo numero {id_archivo} Chunks para {filename} ---")
                added = 0
                for i, c in enumerate(chunks): #itero los chunks generados para el archivo actual, luego genero y almaceno el embedding de cada chunk en el vector store con un id unico
                    id=f"{id_archivo}_{i}" #genero el id del chunk actual, donde f es el numero del archivo y i es el numero del chunk dentro de ese archivo
                    chunk_embed = embeddings.embed_query(c) #agrego el chunk vectorizado al vector store con el id
                    collection.add(ids=[id], embeddings=[chunk_embed], documents=[c])
                    added += 1
                print(f"Chunks añadidos para {filename}: {added}")

def query_embedding(query, top_k: int = 5):
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
        if _distances > 0.5:
            out.append({'id': _id, 'doc': _doc})
    return out
    

def main():
    #chunk_and_embed() #lo llamo solo la primera vez para generar los chunks y embeddings y guardarlos, luego lo comento para no generar duplicados en la base de datos de chromadb
    while True:
        pregunta = input("Ingrese su pregunta: ")
        generate_response(pregunta)

if __name__ == "__main__":
    main()

