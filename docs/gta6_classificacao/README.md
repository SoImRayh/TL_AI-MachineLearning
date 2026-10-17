# Classificação do hype de GTA VI

## Arquivos

- Base: `datasets/gta6_fan_expectations_2026.csv` (original, não modificada).
- Notebook: `notebooks/Gta6ClassificacaoTrabalho.ipynb`.
- Execução de ponta a ponta: `docs/gta6_classificacao/executar.py`.
- Resultados (tabelas, figuras, relatório gerado): `docs/gta6_classificacao/resultados/`.

## Tarefa

- **Alvo:** `hype_score_1to10` reagrupado em 5 classes ordenadas:

  |Classe|Notas|Registros|
  |---|---|---:|
  |Baixo|1–6|1.447|
  |Alto|7|1.025|
  |Muito alto|8|1.172|
  |Altíssimo|9|1.032|
  |Máximo|10|1.324|

  Com as 10 notas originais, a razão entre a maior e a menor classe era 662× (a nota 1 tinha 2 registros). Agrupada, ela cai para 1,4×, e todas as classes aparecem em treino, validação e teste.
- **Removidos de X:**
  - `respondent_id`: identificador da linha;
  - `overall_sentiment`: função exata da nota (1–4 → Neutral/Cautious, 5–7 → Positive, 8–10 → Very Positive), o que seria vazamento de dados;
  - `hype_score_1to10`: nota original, da qual o alvo é derivado.
- **TF-IDF:** aplicado a `most_anticipated_feature` (unigramas e bigramas, TF sublinear, norma L2). Numéricos passam por imputação pela mediana e padronização; categóricos, por one-hot. O pré-processamento é ajustado só no subconjunto de treino de cada modelo.

## Protocolo

- Divisão estratificada 60/20/20: 3.600 / 1.200 / 1.200, semente 42.
- **LR-GD:** `SGDClassifier(loss="log_loss")` nas variantes sem penalidade, Ridge, Lasso e Elastic Net; 160 épocas. O alpha é escolhido na validação dentro de {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} (figura `busca_alpha.png`).
- **AD própria:** CART/Gini implementada no notebook (`ArvoreDecisaoPropria`), com `fit` e `predict` próprios. A profundidade é escolhida pela curva de complexidade (1 a 12).
- **Consistência:** LR Ridge e AD em 30%, 60% e 90% do treino, com 3 sementes.
- **Curvas:** 4 da LR (F1 macro × nº de instâncias de treino) e 1 da AD (F1 macro × profundidade).
- **Teste:** precision, recall, F1 micro/macro/weighted e métricas por classe para os 81 modelos. Três modelos são escolhidos pela validação para as matrizes de confusão.

## Executar

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/gta6_classificacao/executar.py
```

O script executa o notebook em um kernel novo e salva as saídas no próprio notebook; leva cerca de 2 minutos. Dependências: numpy, pandas, scikit-learn, scipy, matplotlib, tabulate, nbclient, nbformat, ipykernel.
