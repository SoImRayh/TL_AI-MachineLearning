# Resultados da execução — GTA VI

## Configuração
6000 registros; alvo classe_hype com 5 classes (Baixo: notas 1–6, Alto: nota 7, Muito alto: nota 8, Altíssimo: nota 9, Máximo: nota 10). Removidos de X: respondent_id (identificador), overall_sentiment (derivado da nota) e hype_score_1to10.

Treino=3600; validação=1200; teste=1200. Sementes=[42, 43, 44]; épocas SGD=160; alpha escolhido: {'LR sem regularizacao': 0.0, 'LR Elastic Net': 0.0001, 'LR Lasso': 1e-05, 'LR Ridge': 0.0001}; profundidade AD escolhida=5 (limite 12). 81 configurações avaliadas no teste.

## Busca de alpha (validação, F1 macro)
| familia        |   alpha |   treino_f1_macro |   validacao_f1_macro |   coeficientes_zero_pct |
|:---------------|--------:|------------------:|---------------------:|------------------------:|
| LR Ridge       |  1e-05  |            0.4019 |               0.3614 |                       0 |
| LR Ridge       |  0.0001 |            0.4001 |               0.3619 |                       0 |
| LR Ridge       |  0.001  |            0.3974 |               0.3571 |                       0 |
| LR Ridge       |  0.01   |            0.3723 |               0.3397 |                       0 |
| LR Ridge       |  0.1    |            0.2751 |               0.2649 |                       0 |
| LR Lasso       |  1e-05  |            0.4001 |               0.363  |                       0 |
| LR Lasso       |  0.0001 |            0.3996 |               0.3602 |                       0 |
| LR Lasso       |  0.001  |            0.3814 |               0.3482 |                       0 |
| LR Lasso       |  0.01   |            0.327  |               0.3229 |                       0 |
| LR Lasso       |  0.1    |            0.272  |               0.2856 |                       0 |
| LR Elastic Net |  1e-05  |            0.4006 |               0.3614 |                       0 |
| LR Elastic Net |  0.0001 |            0.4003 |               0.3635 |                       0 |
| LR Elastic Net |  0.001  |            0.391  |               0.3593 |                       0 |
| LR Elastic Net |  0.01   |            0.3439 |               0.3329 |                       0 |
| LR Elastic Net |  0.1    |            0.2522 |               0.2616 |                       0 |

## Métricas dos três modelos selecionados (teste)
|Modelo|Precision macro|Recall macro|F1 micro|F1 macro|Gap treino-validação|
|---|---:|---:|---:|---:|---:|
|curva_lr&#124;LR sem regularizacao&#124;seed=42&#124;fraction=1.0&#124;depth=None|0.3337|0.3956|0.4425|0.3395|0.0404|
|curva_lr&#124;LR Elastic Net&#124;seed=42&#124;fraction=1.0&#124;depth=None|0.3289|0.3945|0.4417|0.3370|0.0368|
|curva_ad&#124;AD propria&#124;seed=42&#124;fraction=1.0&#124;depth=5|0.4198|0.4180|0.4400|0.4159|0.0384|

Baseline majoritário: F1 micro=0.2408; F1 macro=0.0776.

## Curvas
- LR sem regularizacao: validação F1 macro 0.3580 (360 exemplos) → 0.3622 (3600); gap final 0.0391.
- LR Ridge: validação F1 macro 0.3581 (360 exemplos) → 0.3623 (3600); gap final 0.0380.
- LR Lasso: validação F1 macro 0.3583 (360 exemplos) → 0.3627 (3600); gap final 0.0378.
- LR Elastic Net: validação F1 macro 0.3572 (360 exemplos) → 0.3638 (3600); gap final 0.0363.
- AD: treino 0.2180 → 0.6168; validação 0.2130 → 0.3859 (profundidade 1 → 12); melhor validação na profundidade 5.

## Consistência
| familia    |   fracao_treino |   n_treino |   treino_media |   validacao_media |   validacao_desvio |   validacao_min |   validacao_max |   gap_medio |
|:-----------|----------------:|-----------:|---------------:|------------------:|-------------------:|----------------:|----------------:|------------:|
| AD propria |             0.3 |       1080 |         0.464  |            0.3754 |             0.0275 |          0.3582 |          0.4071 |      0.0886 |
| AD propria |             0.6 |       2160 |         0.4336 |            0.3648 |             0.0231 |          0.3492 |          0.3914 |      0.0688 |
| AD propria |             0.9 |       3240 |         0.4594 |            0.4136 |             0.0076 |          0.4049 |          0.4189 |      0.0459 |
| LR Ridge   |             0.3 |       1080 |         0.4469 |            0.3638 |             0.0174 |          0.3463 |          0.3811 |      0.0831 |
| LR Ridge   |             0.6 |       2160 |         0.4186 |            0.3593 |             0.0058 |          0.3528 |          0.3639 |      0.0593 |
| LR Ridge   |             0.9 |       3240 |         0.4044 |            0.3654 |             0.0074 |          0.361  |          0.374  |      0.039  |

## Erros dos selecionados
### curva_lr|LR sem regularizacao|seed=42|fraction=1.0|depth=None
Confusões principais: Altíssimo → Máximo: 111; Alto → Baixo: 111; Muito alto → Máximo: 87
Recall por classe: Baixo=0.8754; Alto=0.0390; Muito alto=0.2393; Altíssimo=0.0773; Máximo=0.7472

### curva_lr|LR Elastic Net|seed=42|fraction=1.0|depth=None
Confusões principais: Alto → Baixo: 112; Altíssimo → Máximo: 111; Muito alto → Máximo: 87
Recall por classe: Baixo=0.8754; Alto=0.0341; Muito alto=0.2308; Altíssimo=0.0773; Máximo=0.7547

### curva_ad|AD propria|seed=42|fraction=1.0|depth=5
Confusões principais: Altíssimo → Muito alto: 72; Alto → Baixo: 70; Máximo → Altíssimo: 69
Recall por classe: Baixo=0.6886; Alto=0.2732; Muito alto=0.4145; Altíssimo=0.2271; Máximo=0.4868

## Reprodutibilidade
{
  "python": "3.13.9",
  "numpy": "2.4.1",
  "pandas": "2.3.3",
  "scikit_learn": "1.8.0",
  "scipy": "1.17.0"
}