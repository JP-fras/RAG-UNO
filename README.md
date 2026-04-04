# RAG-UNO
Repositorio de la implementacion de un RAG orientado al proyecto del rio reconquista de la Universidad Nacional del Oeste.
Desarrollado por Juan Pablo Frascino

# ACTUALIZACIONES JP
## Semana 16/03
Cambios  
-Creacion de archivo .py 
-Extraccion del texto de los pdf
-Guardado en txt

## Semana 23/03
Cambios:
-Pipeline de preproceso de texto terminado 
-Leo pdf->Normalizo y limpio con regex->Elimino stopwords->Aplico lemmatizacion
-Tambien se agrego la opcion de hacer steamming pero se dejo la implementacion con lemmatizacion
-Se cambio el nombre del archivo .py a preproceso-pdf
-Ahora los textos ya procesados se guardan en una nueva carpeta llamada "/textos"

## Semana 01/04
Cambios:
-Cree el archivo "segmentancion-embedding.py" encargado de la fase 2 y 3:
* Toma los archivos .txt y los segmenta en chunks de 200 caracteres con 20 de overlaping
* Embede los chunks en un espacio vectorial de 384 dimensiones 
* Los almacena en un objeto vector store de manera local y dinamica

## PLAN
![Foto de las fases](/assets/fases-foto.ivf)
| FASE 1 | FASE 2 | FASE 3 | FASE 4 | ... |
| -------|--------|--------|---------|----|
| Preprocesado del texto desde pdf | Segmentacion | Vectorizacion y ponderacion | Base Vectorial | ... |
| COMPLETADA | COMPLETADA | COMPLETADA | en proceso| |