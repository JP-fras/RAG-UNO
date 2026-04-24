#utilizo la liberia Pymupdf para extraer el texto de un pdf
#pip install pymupdf
#utilizo la liberia spacy para lematizar el texto extraído de los pdfs
#pip install spacy
#python -m spacy download es_core_news_sm
#pip install nltk
#pip  install  langchain-text-splitters
#pip install -qU langchain-huggingface
#pip install sentence-transformers
#pip install -U "langchain-core"
#pip install chromadb

import fitz
import os
pdf_dir = "./pdfs" #directorio donde se encuentran los pdfs
pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs") #actualizo la ruta del directorio de PDFs
txt_dir = os.path.join(os.path.dirname(__file__), "textos") #directorio donde se guardarán los txts procesados
import spacy
nlp = spacy.load('es_core_news_sm')
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
spanish_stopwords = set(stopwords.words("spanish"))
stemmer = SnowballStemmer("spanish")
import re
#importo las librerias de langchain para generar chunks, embeddings y almacenar los embeddings en un vector store en memoria local
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import chromadb

#configuracion de la base de datos vectorial
db_dir = os.path.join(os.path.dirname(__file__), "db")
chroma_client = chromadb.PersistentClient(path=db_dir) #creo un cliente de chromadb para almacenar los embeddings de manera persistente en el directorio db
collection = chroma_client.get_collection(name="chunks_embeddings") #obtengo la colección de chromadb donde voy a almacenar los embeddings generados, si ya existe la colección, la obtengo, sino la creo

#funcion para lematizar el texto extraído de los pdfs ej: "corriendo" -> "correr", "niños" -> "niño", etc
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    print(lemmatized_text[:10]) #imprimo los primeros 10 caracteres del texto lematizado para verificar que se hizo correctamente
    return lemmatized_text

#funcion para eliminar los stopwords del texto extraído de los pdfs ej:  "el", "la", "de", "y", etc
def remove_stopwords(text):
    #elimino los stopwords del texto limpio
    tokens = word_tokenize(text, language="spanish")
    filtered = [t for t in tokens if t not in spanish_stopwords]
    return ' '.join(filtered)

#funcion para aplicar stemming al texto extraído de los pdfs ej: "corriendo" -> "corr", "niños" -> "niñ", etc
def stem_text(text): 
    tokens = word_tokenize(text, language="spanish")
    stemmed = [stemmer.stem(t) for t in tokens]
    return ' '.join(stemmed)

#funcion para limpiar y normalizar el texto extraído de los pdfs
def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # elimino líneas que solo contienen números (número de página)
        if re.match(r'^\s*\d+\s*$', line):
            continue
        # elimino líneas con nombre del documento (ajustar si tienes un patrón específico)
        if re.search(r'prueba|nombre_del_documento', line, re.IGNORECASE):
            continue
        # elimino líneas de watermarks comunes
        if re.search(r'watermark|confidencial|sample', line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    # normalizar saltos de línea y espacios extra
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # espacios extra
    cleaned_text = cleaned_text.replace('\n ', '\n').replace(' \n', '\n')
    # resuelvo problemas de encoding (caracteres raros)
    cleaned_text = cleaned_text.encode('utf-8', 'ignore').decode('utf-8')
    return cleaned_text.strip()

def remove_repeated_headers(pages, min_occurrence=2):
    """
    Detecta y elimina encabezados que se repiten en varias páginas.
    Recibe una lista de strings (cada string es el texto de una página).
    """
    # obtener la primera línea no vacía de cada página
    first_lines = []
    for p in pages:
        fl = ""
        for line in p.splitlines():
            s = line.strip()
            if s:
                fl = s
                break
        first_lines.append(fl)

    # contar ocurrencias
    counts = {}
    for fl in first_lines:
        counts[fl] = counts.get(fl, 0) + 1

    # seleccionar candidatos a encabezado (aparecen al menos min_occurrence veces y son cortos)
    candidate_headers = set()
    for line, cnt in counts.items():
        if not line:
            continue
        if cnt >= min_occurrence:
            words = line.split()
            if 1 < len(words) <= 8 and len(line) <= 120:
                candidate_headers.add(line)

    # eliminar el encabezado candidato del inicio de cada página
    cleaned_pages = []
    for p, fl in zip(pages, first_lines):
        cleaned = p
        if fl in candidate_headers:
            lines = cleaned.splitlines()
            i = 0
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].strip() == fl:
                del lines[i]
                cleaned = "\n".join(lines)
        cleaned_pages.append(cleaned)
    return cleaned_pages


def chunk_and_embed():
    # Evita UnboundLocalError: usa la variable global `collection`
    global collection

    # Configuración del splitter
    # chunk_size = 200 tokens
    # chunk_overlap = 20 tokens (10% de 200)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, #cantidad maxima de caracteres por chunk
        chunk_overlap=20, #cantidad de solapamiento entre chunks (20 caracteres)
        length_function=len,  # aquí usamos len, es la deafult
    )   

    # Intento vaciar la colección existente; si falla, borro y recreo la colección
    try:
        collection.delete()
    except Exception:
        try:
            chroma_client.delete_collection(name="chunks_embeddings")
        except Exception:
            pass
        collection = chroma_client.create_collection(name="chunks_embeddings")
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


def preprocess(pdf_dir=pdf_dir, txt_dir=txt_dir):
    #loop principal para extraer el texto y guradarlo en txts separados por cada pdf
    for filename in os.listdir(pdf_dir): #itero por cada pdf en el directorio
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)#genero el path del pdf
            txt_path = os.path.join(txt_dir, os.path.splitext(filename)[0] + ".txt") #genero el path del txt donde voy a escribir
            with fitz.open(pdf_path) as doc, open(txt_path, "w", encoding="utf-8") as txt_file: #abro el pdf y el txt para escribir el texto extraído
                # extraigo texto de todas las páginas primero
                pages_raw = [page.get_text("text") for page in doc]
                # elimino encabezados repetidos (ej: nombre del autor que aparece en cada página)
                pages = remove_repeated_headers(pages_raw, min_occurrence=2)
                # proceso cada página ya sin encabezados repetidos
                for page_text in pages:
                    text = clean_text(page_text)
                    text = remove_stopwords(text)
                    text = lemmatize_text(text)
                    #opcionalmente podria haber aplicado stemming, pero en este caso no lo hago 
                    #porque la lematización ya es suficiente para normalizar el texto y el stemming 
                    #podría ser demasiado agresivo y perder información importante.
                    #text = stem_text(text) 
                    txt_file.write(text) #escribo el texto procesado en el txt
                    txt_file.write("\n\n") #separo cada pagina con un salto doble
            print(f"Texto extraído de {filename} y guardado en {txt_path}")

def main():
    preprocess() #primero extraigo el texto de los pdfs y lo guardo en txts
    chunk_and_embed() #luego genero los chunks y los embeddings de cada chunk y los guardo en la base de datos vectorial

if __name__ == "__main__":
    main()