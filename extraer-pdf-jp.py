#utilizo la liberia Pymupdf para extraer el texto de un pdf
#pip install pymupdf

import fitz
import os

pdf_dir = "./pdfs" #directorio donde se encuentran los pdfs
pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs") #actualizo la ruta del directorio de PDFs
import re

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

#loop principal para extraer el texto y guradarlo en txts separados por cada pdf
for filename in os.listdir(pdf_dir): #itero por cada pdf en el directorio
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)#genero el path del pdf
        txt_path = os.path.splitext(pdf_path)[0] + ".txt" #genero el path del txt donde voy a escribir
        with fitz.open(pdf_path) as doc, open(txt_path, "w", encoding="utf-8") as txt_file: #abro el pdf y el txt para escribir el texto extraído
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text") #obtengo el texto
                text = clean_text(text) #lo limpio y normalizo con la fucnion clean_text
                txt_file.write(text) #escribo el texto limpio en el txt
                txt_file.write("\n\n") #separo cada pagina con un salto doble
        print(f"Texto extraído de {filename} y guardado en {txt_path}")

#PROBLEMAS A RESOLVER:  No reconoce imagenes/fotos/escaneos, por ej, el documento de revista ANDIS,
#el cual es un escaneo, no lo toma, solo toma archivos pdf nativos, para eso se podria usar ocr tal vez.