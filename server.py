from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import io
import contextlib
import consultas_rag as consultas

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10 # Valor predeterminado para top_k

@app.get("/health")
def health():
    if consultas is None:
        return {"status": "error", "module_loaded": False, "error": "module not loaded"}
    return {"status": "ok", "modulo cargado": True}


# Ejemplo de uso con curl desde cmd(trae los 40 chunks mas relevantes):
#curl -v -X POST "http://localhost:8000/embedding" -H "Content-Type: application/json" -d "{\"query\":\"¿Donde queda el rio reconquista?\",\"top_k\":40}"
@app.post("/embedding")
def embedding(req: QueryRequest):
    try:
        chunks = consultas.query_embedding(req.query, top_k=req.top_k)
        return {"query": req.query, "top_k": req.top_k, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejemplo de uso con curl desde cmd sin especificar top_k (trae los 10 chunks mas relevantes por defecto):
#curl -v -X POST "http://localhost:8000/generate-api" -H "Content-Type: application/json" -d "{\"query\":\"¿Donde queda el rio reconquista?\"}"
# Ejemplo de uso con curl desde cmd especificando top_k:
#curl -v -X POST "http://localhost:8000/generate-api" -H "Content-Type: application/json" -d "{\"query\":\"¿Donde queda el rio reconquista?\",\"top_k\":20}"
@app.post("/generate-api")
def generate_api(req: QueryRequest):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            consultas.generate_response_api(req.query, top_k_chunks=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    output = buf.getvalue()
    return {"query": req.query, "output": output}

#Implementacion con el modelo local, comentada por deafult debido a que el modelo local consume mucho computo para un servidor simple de handler api
#@app.post("/generate-local")
#def generate_local(req: QueryRequest):
#    buf = io.StringIO()
#    try:
#        with contextlib.redirect_stdout(buf):
#            consultas.generate_response_local(req.query, top_k_chunks=req.top_k)
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))
#    output = buf.getvalue()
#    return {"query": req.query, "output": output}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
