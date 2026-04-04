#pip  install  langchain-text-splitters
#pip install -qU langchain-huggingface
#pip install sentence-transformers
#pip install -U "langchain-core"

#importo las librerias de langchain para generar chunks, embeddings y almacenar los embeddings en un vector store en memoria local
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os

#directorio donde se encuentran los txts procesados
txt_dir = os.path.join(os.path.dirname(__file__), "textos")

# Configuración del splitter
# chunk_size = 200 tokens
# chunk_overlap = 20 tokens (10% de 200)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, #cantidad maxima de caracteres por chunk
    chunk_overlap=20, #cantidad de solapamiento entre chunks (20 caracteres)
    length_function=len,  # aquí usamos len, es la deafult
)   

#funcion para generar los chunks de texto a partir de los txts procesados y luego generar los embeddings utilizando HuggingFaceEmbeddings 
# y almacenarlos en un vector store en memoria (InMemoryVectorStore) con un id unico para cada chunk generado
def chunk_and_embed():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #objeto encargado de generar los embeddings de 384 dimensiones utilizando el modelo "all-MiniLM-L6-v2" de HuggingFace(https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    vector_store = InMemoryVectorStore(embeddings) #objeto encargado de almacenar los embeddings generados en memoria local
    id_archivo = 0
    for filename in os.listdir(txt_dir): #itero por cada txt en el directorio
        id_archivo += 1
        if filename.lower().endswith(".txt"):
            txt_path = os.path.join(txt_dir, os.path.splitext(filename)[0] + ".txt") #genero el path del txt donde voy a escribir
            with open(txt_path, "r", encoding="utf-8") as txt_file:
                texto = txt_file.read() #leo el archivo txt completo
                chunks = splitter.split_text(texto) #genero los chunks con las propiedades ya configuradas
                print(f"---Archivo numero {id_archivo} Chunks para {filename} ---")
                for i, c in enumerate(chunks): #itero los chunks generados para el archivo actual, luego genero y almaceno el embedding de cada chunk en el vector store con un id unico
                    id=f"{id_archivo}_{i}" #genero el id del chunk actual, donde f es el numero del archivo y i es el numero del chunk dentro de ese archivo
                    vector_store.add_texts(texts=[c], ids=[id]) #agrego el chunk vectorizado al vector store con el id 
                    embedding_result =  vector_store.store.get(id)  # Obtener el embedding generado para el chunk actual
                    print(f"Embedding para chunk {i+1}: {embedding_result}")

def main():
    chunk_and_embed()


if __name__ == "__main__":
    main()

