from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title='Cricket Inference Service', version='0.1.0')

class InferRequest(BaseModel):
    prompt: str

@app.get('/healthz')
def health():
    return {'status': 'ok'}

@app.get('/readyz')
def ready():
    return {'ready': True}

@app.post('/infer')
def infer(req: InferRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail='Prompt required')
    # TODO: load model from private HF repo using HF_TOKEN; call with req.prompt + computed stats context
    return {'answer': f"Stub answer to: {req.prompt}"}
