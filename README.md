# RAG-UNO
Repositorio de la implementacion de un RAG orientado al proyecto del rio reconquista de la Universidad Nacional del Oeste.
Desarrollado por Juan Pablo Frascino

# DOCUMENTACION
## Intro
El programa se divide en archivos .py.
El primer archivo denominado "preproceso-db.py" es el encargado del preprocesado de los textos pdf y el armado de la base de datos vectorial, esto es un archivo que corre de manera batch solamente cuando se necesite actualizar la base de conocimiento del rag
El segundo archivo denominado "consultas-rag.py" es el encargado de tomar el prompt del usuario contrastar la consulta contra la db y utilizar los modelos de llm para generar la consulta, este es el que debe correr de cara al usuario.

### Disclaimer!
Antes de utilizar, correr el requirements.txt para descargar dependencias

## Utilizacion de metodos
en construccion...