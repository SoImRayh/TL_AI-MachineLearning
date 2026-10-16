# Resultados da execução — GTA VI

## Aderência
Adaptação ao CSV: 6000 registros e 10 classes nativas de hype_score_1to10 (notas de 1 a 10). NÃO há texto livre; TF-IDF aplicado às dez descrições categóricas de most_anticipated_feature. overall_sentiment possui três classes, não cinco. Aprovação do professor ou dados textuais reais ainda são necessários para aderência literal. A nota 1 tem dois registros: um no treino e outro no teste, sem suporte na validação. F1 macro considera todas as dez classes.

## Protocolo
Treino=3600; validação=1200; teste=1200. Sementes=[42, 43, 44]; épocas SGD=160; altura limite AD=12. 81 configurações avaliadas; cinco curvas; três matrizes de confusão.

Consistência: 1.080, 2.160 e 3.240 registros; três repetições por algoritmo/tamanho. Subconjuntos aninhados, pré-processamento ajustado em cada subconjunto, teste fora da seleção.

## Métricas dos três modelos selecionados
|Modelo|Precision macro|Recall macro|F1 micro|F1 macro|Gap treino-validação|
|---|---:|---:|---:|---:|---:|
|curva_lr&#124;LR sem regularizacao&#124;seed=42&#124;fraction=1.0&#124;depth=None|0.3490|0.3808|0.5225|0.3474|0.0427|
|curva_lr&#124;LR Elastic Net&#124;seed=42&#124;fraction=1.0&#124;depth=None|0.3594|0.3848|0.5333|0.3508|0.0299|
|curva_ad&#124;AD propria&#124;seed=42&#124;fraction=1.0&#124;depth=8|0.3854|0.3980|0.4867|0.3876|0.1339|

Baseline majoritário: F1 micro=0.2208; F1 macro=0.0362.

## Interpretação das curvas
- LR sem regularizacao: validação média F1 macro de 0.2701 (360 exemplos) para 0.3378 (3.600), diferença +0.0677.
- LR Ridge: validação média F1 macro de 0.2690 (360 exemplos) para 0.3384 (3.600), diferença +0.0694.
- LR Lasso: validação média F1 macro de 0.2712 (360 exemplos) para 0.3376 (3.600), diferença +0.0664.
- LR Elastic Net: validação média F1 macro de 0.2703 (360 exemplos) para 0.3409 (3.600), diferença +0.0706.
- AD: F1 macro de treino 0.1132 → 0.5484; validação 0.1132 → 0.3300 entre profundidades 1 e 12. Comparar com a profundidade selecionada, não escolher pela maior altura.

## Consistência
familia,fracao_treino,n_treino,treino_media,validacao_media,validacao_desvio,validacao_min,validacao_max,gap_medio
AD propria,0.3,1080,0.5278515541690827,0.35631033790074684,0.011490171661578895,0.34315322160239564,0.3643693563993734,0.17154121626833585
AD propria,0.6,2160,0.5180187941496192,0.37131812219168564,0.019415415931016932,0.3525447844131432,0.39131750596754217,0.14670067195793363
AD propria,0.9,3240,0.49636713454646325,0.3693698063063042,0.007340290645670081,0.36220976658028253,0.37687796059983736,0.12699732824015905
LR Ridge,0.3,1080,0.4060720731301826,0.32453290560583087,0.00573757415245602,0.31809696150045497,0.32911239326962627,0.08153916752435177
LR Ridge,0.6,2160,0.3920722171723096,0.3335178669335351,0.003377426369711347,0.3298493174796918,0.3364981423416743,0.05855435023877446
LR Ridge,0.9,3240,0.38408419931558674,0.3378226064047692,0.006007149029651485,0.3315788358405064,0.34356123323318993,0.046261592910817495

O desvio apresentado é entre três amostragens do mesmo holdout, não intervalo de confiança nem prova de significância.

## Erros dos selecionados
### curva_lr|LR sem regularizacao|seed=42|fraction=1.0|depth=None
Confusões principais: 9 → 10: 97; 9 → 8: 95; 6 → 7: 91
Recall por classe: 1=0.0000; 2=0.0000; 3=0.0000; 4=1.0000; 5=0.2500; 6=0.3087; 7=0.8195; 8=0.6709; 9=0.0680; 10=0.6906

### curva_lr|LR Elastic Net|seed=42|fraction=1.0|depth=None
Confusões principais: 9 → 10: 94; 9 → 8: 93; 6 → 7: 93
Recall por classe: 1=0.0000; 2=0.0000; 3=0.0000; 4=1.0000; 5=0.2262; 6=0.2953; 7=0.8537; 8=0.6709; 9=0.0922; 10=0.7094

### curva_ad|AD propria|seed=42|fraction=1.0|depth=8
Confusões principais: 9 → 8: 77; 9 → 10: 74; 7 → 6: 68
Recall por classe: 1=0.0000; 2=0.0000; 3=0.5714; 4=0.6842; 5=0.2976; 6=0.4094; 7=0.6000; 8=0.6111; 9=0.2670; 10=0.5396

## Conclusão
Os números acima medem associação contemporânea das respostas com a nota de hype. Não demonstram análise de sentimento em texto livre. Acurácia/micro F1 devem ser lidos com macro F1 e recalls; a classe majoritária pode dominar. Regularização e mais amostras não garantem melhora. Os escores de teste não foram usados para selecionar os três modelos.

## Reprodutibilidade
{
  "python": "3.14.4",
  "numpy": "2.5.2",
  "pandas": "3.0.5",
  "scikit_learn": "1.9.0",
  "scipy": "1.18.1"
}

Consultar README.md da documentação para execução, método, artefatos e pendências de aderência.