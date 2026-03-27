#utilizo la liberia Pymupdf para extraer el texto de un pdf
#pip install pymupdf
#utilizo la liberia spacy para lematizar el texto extraído de los pdfs
#pip install spacy
#python -m spacy download es_core_news_sm
#pip install nltk

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
        #elimino caracteres especiales
        if re.search(r"[^a-záéíóúñü\s]", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    # normalizar saltos de línea y espacios extra
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # espacios extra
    cleaned_text = cleaned_text.replace('\n ', '\n').replace(' \n', '\n')
    # resuelvo problemas de encoding (caracteres raros)
    cleaned_text = cleaned_text.encode('utf-8', 'ignore').decode('utf-8')
    return cleaned_text.strip()

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    #opcionalmente podria haber aplicado stemming, pero en este caso no lo hago 
    #porque la lematización ya es suficiente para normalizar el texto y el stemming 
    #podría ser demasiado agresivo y perder información importante.
    #text = stem_text(text) 
    return text

#loop principal para extraer el texto y guradarlo en txts separados por cada pdf
for filename in os.listdir(pdf_dir): #itero por cada pdf en el directorio
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)#genero el path del pdf
        txt_path = os.path.join(txt_dir, os.path.splitext(filename)[0] + ".txt") #genero el path del txt donde voy a escribir
        with fitz.open(pdf_path) as doc, open(txt_path, "w", encoding="utf-8") as txt_file: #abro el pdf y el txt para escribir el texto extraído
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text") #obtengo el texto
                text = preprocess_text(text) #aplico el preprocesamiento al texto extraído
                txt_file.write(text) #escribo el texto procesado en el txt
                txt_file.write("\n\n") #separo cada pagina con un salto doble
        print(f"Texto extraído de {filename} y guardado en {txt_path}")