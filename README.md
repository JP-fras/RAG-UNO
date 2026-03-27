# RAG-UNO
Repositorio de la implementacion de un RAG orientado al proyecto del rio reconquista de la Universidad Nacional del Oeste.

# ACTUALIZACIONES JP
## Semana 16/03
Cambios
-Creacion de archivo .py 
-Extraccion del texto de los pdf
-Guardado en txt

## Semana 23/03
Cambios:
-Pipeline de preproceso de texto terminado 
Leo pdf->Normalizo y limpio con regex->Elimino stopwords(ej:  "el", "la", "de", "y", etc)->Aplico lemmatizacion(ej: "corriendo" -> "correr", "niños" -> "niño", etc)
* Tambien se agrego la opcion de hacer steamming pero se dejo la implementacion con lemmatizacion
-Se cambio el nombre del archivo .py a preproceso-pdf
-Ahora los textos ya procesados se guardan en una nueva carpeta llamada "/textos"

## PLAN
| FASE 1 | FASE 2 | FASE 3 | FASE 4 | ... |
| -------|--------|--------|---------|----|
| Preprocesado del texto desde pdf | Segmentacion y Chunking  | Vectorizacion y ponderacion | Base Vectorial | ... |
| COMPLETADA | | | | |