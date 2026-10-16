# Trabalho acadêmico de classificação — GTA VI

## Arquivos de entrada e entrega

- Base original: `datasets/gta6_fan_expectations_2026.csv` (preservada).
- Notebook exclusivo: `notebooks/Gta6ClassificacaoTrabalho.ipynb`.
- Código de execução: `docs/gta6_classificacao/executar.py`.
- Testes: `docs/gta6_classificacao/verificar.py`.
- Relatório com números reais: `resultados/relatorio_resultados.md`.
- Tabelas, manifesto e figuras: `resultados/`.

Os demais notebooks não são usados como destino de escrita. O arquivo `notebooks/GtaViClassification.ipynb` já tinha alterações do usuário antes desta implementação; essas alterações não foram sobrescritas, revertidas ou atribuídas a este trabalho.

## Pendência importante: este CSV não atende integralmente ao enunciado

A auditoria confirmou 6.000 linhas, 20 colunas e identificadores únicos.

`overall_sentiment` possui apenas três classes:

|Sentimento|Quantidade|
|---|---:|
|Very Positive|3.528|
|Positive|2.190|
|Neutral/Cautious|282|

Não existem comentários ou respostas discursivas. `most_anticipated_feature` tem dez frases predefinidas e repetidas. Essas descrições são texto armazenado em strings, mas sua natureza é categórica, não genuinamente não estruturada. Concatenar os campos ou gerar comentários artificiais não resolveria essa lacuna cientificamente.

O alvo é `hype_score_1to10`: classificação das dez notas originais de hype, sem agrupamento ou criação de rótulos.

|Classe|Quantidade|
|---|---:|
|1|2|
|2|19|
|3|69|
|4|192|
|5|419|
|6|746|
|7|1.025|
|8|1.172|
|9|1.032|
|10|1.324|

O objetivo é identificar a nota de hype pelas demais respostas contemporâneas do questionário, não realizar análise textual de sentimento ou previsão temporal. `primary_platform` passa a ser atributo categórico; `overall_sentiment` continua como atributo estruturado. O alvo e `respondent_id` ficam fora de X. Sem proveniência da coleta, não se pode descartar que o sentimento seja derivado da nota: antes de usar o modelo em produção, verificar sua origem e disponibilidade. O protocolo não demonstra ausência de vazamento semântico entre respostas correlatas.

TF-IDF é aplicado às frases originais de `most_anticipated_feature`, sem fabricar comentários, com a limitação explicitamente marcada. O manifesto registra `genuine_unstructured_text=false` e `full_assignment_compliance=false`.

Para entrega estritamente aderente: obter aprovação explícita do professor para essa adaptação ou usar/complementar a base com textos reais e rótulos adequados, de fonte e licença verificadas. Não afirmar que todas as exigências do enunciado foram cumpridas. A origem/licença/representatividade da base não foi informada no CSV; não se assume coleta real.

## Método e prevenção de vazamento

### Divisão

1. Reservar um dos dois registros de nota 1 para treino e outro para teste. Essa classe não tem suporte suficiente para ocupar três partições disjuntas.
2. Estratificar as demais classes, completando 1.200 registros no teste e 1.200 na validação (20% do total em cada).
3. Treino: 3.600 registros, 60% do total.
4. Semente principal 42; conjuntos de IDs disjuntos e cobertura do arquivo inteiro verificados.

Não há nota 1 na validação. F1 macro considera sempre as dez classes, com zero para a classe ausente. Não é possível avaliar sua generalização de forma confiável com apenas dois registros. Nenhuma linha é removida ou duplicada. Os subconjuntos de treino usam alocação proporcional com mínimo de um exemplo por classe, preservando todas as dez notas e o aninhamento.

Subconjuntos estratificados e aninhados dentro do treino para sementes 42, 43 e 44:

|Parcela do treino|Instâncias|Parcela do total|
|---|---:|---:|
|10%|360|6%|
|30%|1.080|18%|
|60%|2.160|36%|
|90%|3.240|54%|
|100%|3.600|60%|

30%, 60% e 90% constituem a análise de consistência pedida; 10% e 100% fornecem pontos adicionais às curvas LR. Para cada semente, os menores subconjuntos estão contidos nos maiores. A divisão principal de treino/validação/teste não muda. As três sementes medem variação na amostragem/SGD, não substituem validação cruzada nem sustentam testes de significância.

### Representação

- TF-IDF: unigramas e bigramas, minúsculas, TF sublinear, IDF suavizado e normalização L2.
- Numéricos: imputação pela mediana e StandardScaler.
- Categóricos: categoria de ausência e one-hot; categorias desconhecidas ignoradas.
- Vocabulário, IDF, categorias, imputação e escala ajustados SOMENTE no subconjunto usado para treinar cada modelo.
- LR e AD recebem a mesma representação combinada.
- A árvore didática converte a matriz para densa; a LR trabalha com matriz esparsa.

A exploração inicial da base completa é descritiva; os cruzamentos entre atributos e alvo são feitos apenas no treino. Não há ajuste de representação com validação/teste. Textos repetidos não são documentos discursivos duplicados: são opções do questionário. Duplicações de linha e de respostas sem ID são auditadas.

### LR com gradient descent

`SGDClassifier(loss='log_loss')`: regressão logística via gradient descent estocástico (SGD), multiclasse um-contra-resto. Não confundir com o solver padrão de `LogisticRegression`, nem com softmax multinomial.

- Sem penalidade: `penalty=None`, alpha=0.
- Ridge: penalidade L2, alpha=0,001.
- Lasso: penalidade L1, alpha=0,001.
- Elastic Net: L1/L2, alpha=0,001 e l1_ratio=0,5.
- 160 épocas com `partial_fit`, taxa inicial 0,01, `invscaling` com potência 0,25, média de pesos.
- Sem parada por validação, sem warm-start entre tamanhos, sem busca de hiperparâmetros e sem balanceamento artificial.

O orçamento de épocas é fixo e não implica convergência garantida. Se o professor exigir GD em lote especificamente, confirmar a aceitação de SGD. Os coeficientes nulos são registrados para inspecionar o efeito das penalidades.

### AD própria

A classe `ArvoreDecisaoPropria` está integralmente no notebook, em célula com tag `arvore-propria`. Não usa a árvore do scikit-learn para treinar ou predizer.

- CART binária multiclasse com impureza Gini.
- Busca exata dos cortes entre valores distintos de todos os atributos. Usa ponto médio numericamente seguro e, se o arredondamento alcançar o limite superior, o valor inferior, preservando a partição e evitando folhas vazias.
- Contagens cumulativas para calcular a impureza ponderada dos filhos.
- Limites por profundidade, pureza, amostras por folha/divisão e ganho mínimo.
- Raiz na profundidade zero; desempate determinístico.
- `fit`, `predict_proba` e `predict` próprios; proporções empíricas nas folhas.
- Profundidade 8 fixa na consistência; profundidades 1 a 12 na curva.
- Altura real e número de nós reportados, pois altura limite e efetivamente alcançada podem diferir.

### Experimentos, cinco curvas e teste

- 60 casos de curvas LR: quatro penalidades × cinco tamanhos × três sementes.
- Nove casos adicionais AD de consistência: três tamanhos × três sementes.
- 12 casos AD da curva de complexidade: profundidades 1 a 12.
- Total: 81 modelos/configurações, todos avaliados no teste.
- Consistência: LR Ridge reutilizada dos nove casos correspondentes das curvas e AD profundidade 8 nos mesmos subconjuntos; médias, desvios, extremos, gaps e diferenças pareadas.
- Quatro figuras LR mostram treino e validação em F1 macro, média ± desvio-padrão amostral entre sementes, com quantidade de exemplos no eixo X.
- Quinta figura AD mostra treino/validação em F1 macro com profundidade no eixo X, conforme solicitado.

Seleção congelada antes de usar o teste: LR sem penalidade com treino completo/semente 42; família regularizada com melhor média de validação no treino completo (representada pela semente 42); melhor profundidade AD por validação, com desempate favorável à menor profundidade. A seleção busca três modelos metodologicamente significativos, não os três maiores resultados no teste. Não se incorpora validação ao treino final.

Métricas de teste em todos os casos: accuracy; precision, recall e F1 macro/micro/weighted; relatórios com precision/recall/F1/suporte por classe. Em multiclasse de rótulo único, micro F1 = accuracy. Divisão por zero retorna 0. Baseline separado prediz a classe mais frequente no treino. Três matrizes de confusão de contagens aparecem no notebook; matrizes normalizadas por classe real também são exportadas em CSV.

## Resultados e interpretação

Consultar o relatório gerado, não números fixados manualmente dentro do código. Na execução verificada:

|Selecionado|F1 macro no teste|F1 micro no teste|
|---|---:|---:|
|LR sem regularização, treino completo, semente 42|0,3474|0,5225|
|LR Elastic Net, treino completo, semente 42|0,3508|0,5333|
|AD própria, profundidade 8|0,3876|0,4867|
|Baseline majoritário|0,0362|0,2208|

Os três selecionados superam o baseline em F1 macro e micro. A LR Elastic Net tem o maior micro F1 entre eles; a AD tem o maior macro F1. Nenhum reconhece as notas 1 e 2 no teste; as LR também não reconhecem a nota 3. O resultado agregado não elimina a dificuldade nas classes minoritárias.

As quatro curvas LR melhoraram o F1 macro médio de validação entre 360 e 3.600 exemplos. A AD elevou o escore de treino com a profundidade, mas a melhor profundidade de validação foi 8, não 12. Persistem gaps treino–validação e limitações nas notas raras. Os resultados não permitem inferências causais, temporais ou sobre toda a população de fãs.

## Execução e verificação

O notebook foi executado de ponta a ponta com o alvo hype e contém saídas reais, incluindo nove imagens embutidas: uma EDA, cinco curvas e três matrizes de 10 × 10. São 13 células de código e zero saídas de erro. Os nove testes cobrem treino/predição da árvore, cortes com valores extremos/adjacentes, altura/desempate, classe única/matriz esparsa, entradas inválidas, vocabulário sem vazamento, imagens/execução, alvo e exclusão de features, classe rara, subconjuntos e integridade dos resultados/partições.

Na raiz do repositório, usando o ambiente existente:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/gta6_classificacao/executar.py
    .venv/bin/python docs/gta6_classificacao/verificar.py

O executor inicia um kernel Python novo com o mesmo interpretador que o chamou, usa transporte IPC local, salva as saídas apenas no notebook exclusivo e interrompe ao primeiro erro. Nenhum servidor Jupyter ou kernel global é instalado. A execução pode sobrescrever somente as saídas do notebook exclusivo e os artefatos em `resultados/`.

Dependências e versões exatas usadas estão em `requirements-execucao.txt`. Esse arquivo registra o ambiente verificado, não prova de compatibilidade com qualquer versão de Python/SO. Para reproduzir em ambiente separado com Python 3.14:

    python3.14 -m venv .venv-gta6-trabalho
    .venv-gta6-trabalho/bin/python -m pip install -r docs/gta6_classificacao/requirements-execucao.txt
    .venv-gta6-trabalho/bin/python docs/gta6_classificacao/executar.py
    .venv-gta6-trabalho/bin/python docs/gta6_classificacao/verificar.py

## Artefatos

|Arquivo/padrão em resultados/|Conteúdo|
|---|---|
|manifesto.json|Hash do CSV, classes, parâmetros, versões, aderência, seleção e contagens|
|auditoria_colunas.csv|Tipos, cardinalidades e ausências|
|distribuicao_classes.csv|Distribuição do alvo|
|particoes_por_classe.csv / particoes_ids.csv|Estratificação e IDs de treino/validação/teste|
|tamanhos_treino.csv / subconjuntos_ids.csv|Tamanhos e IDs de cada subconjunto/semente|
|treino_validacao_todos_casos.csv|Escores, gaps e complexidade dos 81 modelos|
|consistencia_resumo.csv / consistencia_pareada.csv|Médias/desvios e comparação pareada|
|selecao_antes_teste.csv|Três modelos e critérios congelados antes de avaliar teste|
|metricas_teste_todos_casos.csv|Métricas agregadas de todos os 81 casos|
|metricas_teste_por_classe.csv|810 linhas: dez classes × 81 modelos|
|baseline_teste.csv|Métricas do baseline majoritário|
|confusao_N_contagens.csv / confusao_N_normalizada.csv|Três matrizes, contagens e normalização por classe real|
|predicoes_selecionados.csv|IDs, classe real e predições dos três selecionados|
|relatorio_resultados.md|Discussão e números derivados da execução|
|figuras/eda_classes.png|Distribuição de classes|
|figuras/curva_1_lr.png … curva_4_lr.png|Quatro curvas LR|
|figuras/curva_5_ad.png|Curva de profundidade AD|
|figuras/matriz_confusao_1.png … matriz_confusao_3.png|Três matrizes de confusão|

## Referências técnicas oficiais

- TF-IDF: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
- SGD/logistic loss/penalties: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
- Vazamento: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
- Precision/recall/F1: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

As referências explicam técnicas; não validam a proveniência do CSV nem corrigem sua ausência de texto livre.
