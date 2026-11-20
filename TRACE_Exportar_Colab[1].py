# TRACE 1.3 — Cola esta célula no final do Colab, depois de treinar o motor.
# Exporta um JSON de decisões efetivamente calculadas, para o site TRACE no GitHub.
# ATENÇÃO: os dados de exemplo são sintéticos; não publiques dados pessoais.
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

indices = [0, 25, 50, 75, 100]  # podes modificar; usa apenas índices existentes
registos = []
for i in indices:
    if i >= len(X):
        continue
    linha = X.iloc[[i]]
    contrib, intercepto, logodds, prob = explain(modelo, linha, numericas, categoricas)
    prevista = str(modelo.predict(linha)[0])
    classe_positiva = str(metricas['classe_positiva'])
    assert np.isclose(prob, modelo.predict_proba(linha)[0,1], atol=1e-8)
    pesos, paleta, cor = stl_projection(contrib)
    entrada = linha.iloc[0].to_dict()
    nomes = { 'Montante solicitado':'Montante solicitado','Duração do empréstimo':'Duração do empréstimo'}
    montante = float(entrada.get('Montante solicitado',0))
    meses = int(entrada.get('Duração do empréstimo',0))
    risco_pct = prob*100
    estado = 'danger' if risco_pct >= 60 else ('warn' if risco_pct >= 35 else 'ok')
    registos.append({
        'id':f'TR-{i:04d}', 'name':f'Caso {i}', 'amount':f'{montante:,.0f} €'.replace(',',' '),
        'months':meses, 'score':round(risco_pct,1),
        'decision':prevista, 'status':estado,
        'risk':'Elevado' if estado=='danger' else ('Moderado' if estado=='warn' else 'Baixo'),
        'items':[[str(k),round(float(v),4),'negative' if v>0 else 'positive'] for k,v in contrib.items()],
        'trace':{'classe_positiva':classe_positiva,'intercepto':float(intercepto),
                 'log_odds':float(logodds),'stl_pesos':pesos,'stl_cor':cor},
        'input':{str(k):str(v) for k,v in entrada.items()}
    })
exportacao = {
  'schema':'trace-export-v1',
  'gerado_utc':datetime.now(timezone.utc).isoformat(),
  'origem':'Motor Python TRACE — Google Colab',
  'dados':'Sintéticos' if not globals().get('USAR_CSV',False) else 'CSV fornecido pelo utilizador',
  'aviso':'Resultados de investigação. Não usar em decisões financeiras reais.',
  'cases':registos
}
saida=Path('/content/trace_data.json')
saida.write_text(json.dumps(exportacao,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print('Exportados',len(registos),'casos para',saida)
from google.colab import files
files.download(str(saida))
