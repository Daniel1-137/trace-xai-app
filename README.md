# TRACE 1.4 — API Python em tempo real

**Projeto funcional independente**. Não altera o TRACE académico `trace-xai-iade`.

## Executar no computador

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000` para usar o formulário; `http://127.0.0.1:8000/docs` para testar a API.

## Publicação (depois dos testes locais)

1. Criar um **novo serviço web Python** num alojamento que suporte FastAPI (por exemplo, Render); usar **apenas** o repositório `trace-xai-app`.
2. Adicionar os ficheiros deste pacote a uma pasta própria (`backend/`) e configurar a diretoria de trabalho dessa pasta.
3. Comando de instalação: `pip install -r requirements.txt`.
4. Comando de arranque: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Testar `/api/health`, `/docs` e `/` no URL de publicação.
6. Só depois integrar o endereço da API na interface HTML já publicada em `trace-xai-app` (sem tocar no repositório académico).

## Segurança e limitações

- Dados de treino **sintéticos**; probabilidades reais do modelo treinado, mas não validadas para concessão de crédito.
- A explicação é uma decomposição **exata dos log-odds** de regressão logística, não SHAP.
- STL é experimental, baseado no valor absoluto das contribuições, não é explicação causal nem um instrumento validado.
- Demonstração **sem autenticação**, sem persistência, limites de pedidos ou auditoria formal; não utilizar dados pessoais reais ou colocar em produção para decisões de alto risco.
- A amostra sintética é gerada no arranque a partir de uma semente fixa. Para produção, armazenar versão de modelo e artefactos imutáveis.
