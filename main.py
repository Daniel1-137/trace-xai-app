"""TRACE API 1.4 — prova de conceito, apenas dados sintéticos."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from engine import demo_data, train, explain, stl_projection

MODEL = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    data = demo_data(n=500, seed=137)
    model, X, metrics, nums, cats = train(data, "Resultado")
    MODEL.update(model=model, X=X, metrics=metrics, nums=nums, cats=cats)
    yield

app = FastAPI(title="TRACE API — motor experimental", version="1.4.0", lifespan=lifespan,
              description="API para demonstração académica, treinada com dados sintéticos. Não utilizar em decisões reais de crédito.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000", "http://127.0.0.1:8000",
  "https://daniel1-137.github.io"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

class Pedido(BaseModel):
    idade: int = Field(35, ge=18, le=100)
    montante: float = Field(10000, gt=0, le=1000000)
    duracao: int = Field(24, ge=1, le=360)
    historico: int = Field(0, ge=0, le=20)
    poupanca: float = Field(5000, ge=0, le=10000000)
    estabilidade: str = Field("Estável")

@app.get("/")
def inicio():
    return FileResponse(Path(__file__).with_name("demo.html"))

@app.get("/api/health")
def health():
    return {"status": "ok", "motor": "Regressão logística", "dados": "sintéticos", "versao": "1.4.0"}

@app.get("/api/metrics")
def metrics():
    return MODEL["metrics"]

@app.get("/api/cases")
def cases():
    X = MODEL["X"]
    return [{"indice": int(i), "montante": float(row["Montante solicitado"]),
             "duracao": int(row["Duração do empréstimo"])} for i, row in X.iloc[::25].iterrows()]

@app.get("/api/case/{idx}")
def case(idx: int):
    X=MODEL["X"]
    if idx < 0 or idx >= len(X): raise HTTPException(status_code=404,detail="Caso inexistente")
    row=X.iloc[idx]
    return {"idade":int(row["Idade"]),"montante":float(row["Montante solicitado"]),
            "duracao":int(row["Duração do empréstimo"]),"historico":int(row["Histórico de incumprimento"]),
            "poupanca":float(row["Poupança disponível"]),"estabilidade":str(row["Estabilidade laboral"])}

@app.post("/api/analyze")
def analyze(pedido: Pedido):
    if pedido.estabilidade not in {"Estável", "Variável", "Temporário"}:
        raise HTTPException(status_code=422, detail="Estabilidade: Estável, Variável ou Temporário")
    fields={"Idade":pedido.idade,"Montante solicitado":pedido.montante,
            "Duração do empréstimo":pedido.duracao,"Histórico de incumprimento":pedido.historico,
            "Poupança disponível":pedido.poupanca,"Estabilidade laboral":pedido.estabilidade}
    row = pd.DataFrame([fields], columns=MODEL["X"].columns)
    model=MODEL["model"]
    probs=model.predict_proba(row)[0]
    class_positive=MODEL["metrics"]["classe_positiva"]
    idx=list(model.classes_).index(class_positive)
    prob=float(probs[idx])
    contributions, intercept, logodds, reconstructed = explain(model,row,MODEL["nums"],MODEL["cats"])
    weights,palette,color=stl_projection(contributions)
    predicted=str(model.predict(row)[0])
    ranked=sorted(contributions.items(),key=lambda x:abs(x[1]),reverse=True)
    inc=[name for name,v in ranked if v>0]
    dec=[name for name,v in ranked if v<0]
    expl=(f"O modelo classificou o pedido como {predicted.lower()}, atribuindo {prob*100:.1f}% "
          f"à classe {class_positive.lower()}. "
          f"As contribuições positivas mais relevantes para os log-odds foram: {', '.join(inc[:2]) or 'nenhuma'}. "
          f"As negativas foram: {', '.join(dec[:2]) or 'nenhuma'}. "
          "As contribuições são associativas e não demonstram causalidade.")
    payload={"entrada":fields,"previsao":predicted,"classe_positiva":class_positive,
         "probabilidade":round(prob,8),"contribuicoes_logodds":contributions,"intercepto":intercept,
         "soma_logodds":logodds,"conferencia":bool(np.isclose(prob,reconstructed)),
         "stl":{"pesos_percentuais":weights,"paleta":palette,"cor_experimental":color},
         "explicacao":expl,"metricas_teste":MODEL["metrics"],
         "aviso":"Protótipo experimental, dados de treino sintéticos; não utilizar para decisões financeiras reais."}
    payload["hash_evidencia"] = sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()
    return payload
