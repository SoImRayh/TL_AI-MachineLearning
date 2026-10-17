"""Classificação do hype de GTA VI — LR-GD (SGD) x Árvore de Decisão própria.

Execução (na raiz do repositório ou em qualquer pasta dentro dele):
    python code/gta6_classificacao.py

Dependências: numpy pandas scipy scikit-learn matplotlib tabulate
Saídas: métricas no terminal + figuras/CSVs em docs/gta6_classificacao/resultados/
"""
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # salva as figuras sem abrir janelas


def display(obj):
    import pandas as pd
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        with pd.option_context('display.max_rows', 200, 'display.max_columns', 30, 'display.width', 200):
            print(obj, end='\n\n')
    else:
        print(obj)


def Markdown(text):
    return text


def secao(titulo):
    print('\n' + '=' * 80 + '\n' + titulo + '\n' + '=' * 80)

secao('Classificação do hype de GTA VI')
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

# Funciona com execução iniciada na raiz ou na pasta notebooks.
ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents]
            if (p / 'datasets/gta6_fan_expectations_2026.csv').is_file())
DATASET = ROOT / 'datasets/gta6_fan_expectations_2026.csv'
OUT = ROOT / 'docs/gta6_classificacao/resultados'
FIG = OUT / 'figuras'
FIG.mkdir(parents=True, exist_ok=True)
NOTA = 'hype_score_1to10'
TARGET = 'classe_hype'
SEED = 42
SEEDS = [42, 43, 44]
FRACTIONS = [0.10, 0.30, 0.60, 0.90, 1.00]
MAX_DEPTH = 12
EPOCHS = 160
ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
VERSIONS = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
            'scipy': scipy.__version__}
print(VERSIONS)
plt.rcParams.update({'figure.figsize': (9, 5), 'axes.grid': True})

raw = pd.read_csv(DATASET)
# respondent_id é só o identificador da linha: conferimos que é único e o removemos.
assert raw.respondent_id.is_unique
df = raw.drop(columns='respondent_id').reset_index(drop=True)
assert len(df) >= 2000 and not df[NOTA].isna().any()
print('Dimensões (sem ID):', df.shape)
display(df.head())

secao('2. Estudo introdutório')
schema = pd.DataFrame({'tipo': df.dtypes.astype(str), 'valores_unicos': df.nunique(),
                       'ausentes': df.isna().sum(), 'ausentes_pct': df.isna().mean() * 100})
display(schema)
display(df.describe().T)
print('Linhas duplicadas:', int(df.duplicated().sum()))
display(df.most_anticipated_feature.value_counts().rename('quantidade').to_frame())
schema.to_csv(OUT / 'auditoria_colunas.csv')

secao('2.1 Vazamento: `overall_sentiment` é derivado da nota')
leak = pd.crosstab(df[NOTA], df.overall_sentiment)
display(leak)
# Cada nota tem um único sentimento -> o sentimento é determinado pela nota.
assert (leak.gt(0).sum(axis=1) == 1).all()
# Nenhum outro atributo categórico tem essa propriedade.
for column in df.select_dtypes(exclude='number').columns.drop('overall_sentiment'):
    assert not (pd.crosstab(df[NOTA], df[column]).gt(0).sum(axis=1) == 1).all(), column
leak.to_csv(OUT / 'vazamento_overall_sentiment.csv')

secao('2.2 Reagrupamento do alvo em 5 classes')
FAIXAS = {'Baixo': range(1, 7), 'Alto': [7], 'Muito alto': [8], 'Altíssimo': [9], 'Máximo': [10]}
CLASSES = np.array(list(FAIXAS))
mapa = {nota: classe for classe, notas in FAIXAS.items() for nota in notas}
df[TARGET] = df[NOTA].map(mapa)
assert not df[TARGET].isna().any() and df[TARGET].nunique() == 5

original = df[NOTA].value_counts().sort_index()
agrupado = df[TARGET].value_counts().reindex(CLASSES)
display(agrupado.rename('quantidade').to_frame().assign(pct=lambda t: (t.quantidade / len(df) * 100).round(1)))
print('Razão maior/menor classe — original: %.0fx; agrupado: %.2fx'
      % (original.max() / original.min(), agrupado.max() / agrupado.min()))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
original.plot.bar(ax=axes[0], color='gray')
axes[0].set(title='Notas originais (1 a 10)', xlabel='hype_score_1to10', ylabel='Respondentes')
agrupado.plot.bar(ax=axes[1], color='steelblue')
axes[1].set(title='Alvo reagrupado (5 classes)', xlabel='classe_hype', ylabel='Respondentes')
for ax in axes:
    ax.tick_params(axis='x', rotation=0)
fig.tight_layout()
fig.savefig(FIG / 'eda_classes.png', dpi=150)
plt.close(fig)
agrupado.rename('quantidade').to_csv(OUT / 'distribuicao_classes.csv')

secao('3. Partições')
train_val, test = train_test_split(df, test_size=0.20, stratify=df[TARGET], random_state=SEED)
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[TARGET], random_state=SEED)
assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
sets = [set(frame.index) for frame in [train, val, test]]
assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
assert set.union(*sets) == set(df.index)
split_table = pd.DataFrame({name: frame[TARGET].value_counts().reindex(CLASSES, fill_value=0)
                           for name, frame in [('treino', train), ('validacao', val), ('teste', test)]})
display(split_table)
split_table.to_csv(OUT / 'particoes_por_classe.csv')
pd.concat([pd.DataFrame({'linha': frame.index, 'particao': name})
           for name, frame in [('treino', train), ('validacao', val), ('teste', test)]]
          ).to_csv(OUT / 'particoes_ids.csv', index=False)

REMOVIDOS = [TARGET, NOTA, 'overall_sentiment']
FEATURES = [c for c in df if c not in REMOVIDOS]
assert not set(REMOVIDOS) & set(FEATURES)
X_train, y_train = train[FEATURES], train[TARGET].to_numpy()
X_val, y_val = val[FEATURES], val[TARGET].to_numpy()
print('Atributos usados:', FEATURES)


def subconjuntos_aninhados(frame, seed):
    pools = {1.0: frame}
    current = frame
    for fraction in sorted([f for f in FRACTIONS if f < 1], reverse=True):
        size = int(round(len(frame) * fraction))
        current, _ = train_test_split(current, train_size=size, stratify=current[TARGET],
                                      random_state=seed)
        pools[fraction] = current
    for small, large in zip(FRACTIONS[:-1], FRACTIONS[1:]):
        assert set(pools[small].index) <= set(pools[large].index)
    for fraction, sample in pools.items():
        assert len(sample) == round(len(frame) * fraction)
        assert set(sample[TARGET]) == set(CLASSES)
    return pools

POOLS = {seed: subconjuntos_aninhados(train, seed) for seed in SEEDS}
size_table = pd.DataFrame([{'fracao_treino': f, 'n_treino': len(POOLS[42][f]),
                           'fracao_total': len(POOLS[42][f]) / len(df)} for f in FRACTIONS])
display(size_table)
size_table.to_csv(OUT / 'tamanhos_treino.csv', index=False)
pd.concat([pd.DataFrame({'linha': sample.index, 'semente': seed, 'fracao_treino': fraction})
           for seed, pools in POOLS.items() for fraction, sample in pools.items()]
         ).to_csv(OUT / 'subconjuntos_ids.csv', index=False)

secao('4. Representação TF-IDF + dados estruturados')
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def criar_preprocessador(frame):
    """Ajustar SOMENTE no subconjunto de treino de cada experimento."""
    numeric = frame.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric and c != 'most_anticipated_feature']
    transformers = [('texto', TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                     sublinear_tf=True, smooth_idf=True, norm='l2', min_df=1),
                     'most_anticipated_feature')]
    if numeric:
        transformers.append(('numericos', Pipeline([
            ('imputar', SimpleImputer(strategy='median')),
            ('escala', StandardScaler())]), numeric))
    if categorical:
        transformers.append(('categoricos', Pipeline([
            ('imputar', SimpleImputer(strategy='constant', fill_value='Ausente')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical))
    return ColumnTransformer(transformers, sparse_threshold=1.0)

processor_demo = criar_preprocessador(X_train)
Z_demo = processor_demo.fit_transform(X_train)
vocab = processor_demo.named_transformers_['texto']
display(pd.DataFrame({'termo': vocab.get_feature_names_out(), 'idf': vocab.idf_}).head(20))
print('Atributos combinados:', Z_demo.shape[1], 'Vocabulário TF-IDF:', len(vocab.vocabulary_))
print('Exemplos de frases originais:', X_train.most_anticipated_feature.head(3).tolist())

secao('5. Árvore de decisão implementada do zero')
import numpy as np
from scipy import sparse


class ArvoreDecisaoPropria:
    """CART multiclasse: busca exata de cortes, Gini e predição próprias.

    Raiz na profundidade zero. Empates: menor índice de atributo/corte/classe.
    Não utiliza sklearn.tree. Conversão densa é limitada a esta base pequena.
    """

    def __init__(self, max_depth=12, min_samples_leaf=8, min_samples_split=2,
                 min_impurity_decrease=1e-10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

    @staticmethod
    def _matrix(X):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError('X deve ser matriz 2D finita, previamente imputada')
        return X

    def fit(self, X, y):
        X = self._matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
            raise ValueError('y deve ser vetor não vazio alinhado a X')
        if self.max_depth < 0 or self.min_samples_leaf < 1 or self.min_samples_split < 2:
            raise ValueError('Parâmetros de profundidade/amostras inválidos')
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)
        self.node_count_ = 0
        self.depth_ = 0
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self.root_ = self._grow(X, encoded, 0, len(y))
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _grow(self, X, y, depth, n_total):
        self.node_count_ += 1
        self.depth_ = max(self.depth_, depth)
        n = len(y)
        counts = np.bincount(y, minlength=self.n_classes_)
        impurity = 1.0 - np.sum((counts / n) ** 2)
        node = {'proba': counts / n, 'depth': depth, 'samples': n,
                'gini': impurity, 'feature': None}
        if (depth >= self.max_depth or impurity <= 1e-15
                or n < max(self.min_samples_split, 2 * self.min_samples_leaf)):
            return node
        best_gain = self.min_impurity_decrease
        best = None
        # Todas as fronteiras entre valores distintos são avaliadas.
        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind='stable')
            values = X[order, feature]
            positions = np.flatnonzero(values[:-1] < values[1:]) + 1
            positions = positions[(positions >= self.min_samples_leaf)
                                  & (n - positions >= self.min_samples_leaf)]
            if not len(positions):
                continue
            cumulative = np.cumsum(np.eye(self.n_classes_)[y[order]], axis=0)
            left = cumulative[positions - 1]
            right = counts - left
            # Gini ponderado = 1 - (sum(L_k²)/nL + sum(R_k²)/nR)/n.
            weighted = 1 - (np.sum(left ** 2, axis=1) / positions
                            + np.sum(right ** 2, axis=1) / (n - positions)) / n
            gains = impurity - weighted
            i = int(np.argmax(gains))
            if gains[i] > best_gain + 1e-14:
                pos = int(positions[i])
                a, b = float(values[pos - 1]), float(values[pos])
                # Evita overflow com sinais opostos e arredondamento para b.
                threshold = a / 2 + b / 2 if a < 0 < b else a + (b - a) / 2
                if not a <= threshold < b:
                    threshold = a
                best_gain = float(gains[i])
                best = feature, float(threshold)
        if best is None:
            return node
        feature, threshold = best
        mask = X[:, feature] <= threshold
        self.feature_importances_[feature] += n / n_total * best_gain
        node.update(feature=feature, threshold=threshold,
                    left=self._grow(X[mask], y[mask], depth + 1, n_total),
                    right=self._grow(X[~mask], y[~mask], depth + 1, n_total))
        return node

    def predict_proba(self, X):
        if not hasattr(self, 'root_'):
            raise ValueError('Treine a árvore antes da predição')
        X = self._matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Quantidade de atributos diferente do treino')
        probabilities = np.empty((len(X), self.n_classes_))
        for i, row in enumerate(X):
            node = self.root_
            while node['feature'] is not None:
                node = node['left'] if row[node['feature']] <= node['threshold'] else node['right']
            probabilities[i] = node['proba']
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


secao('6. Regressão logística por gradiente descendente')
PENALTIES = {'LR sem regularizacao': None, 'LR Ridge': 'l2',
             'LR Lasso': 'l1', 'LR Elastic Net': 'elasticnet'}
BEST_ALPHA = {'LR sem regularizacao': 0.0}


def treinar_lr(Z, y, family, seed, alpha=None):
    penalty = PENALTIES[family]
    alpha = BEST_ALPHA[family] if alpha is None else alpha
    model = SGDClassifier(loss='log_loss', penalty=penalty,
                          alpha=0.0 if penalty is None else alpha, l1_ratio=0.5,
                          learning_rate='invscaling', eta0=0.01, power_t=0.25,
                          average=True, shuffle=False, random_state=seed,
                          early_stopping=False, n_jobs=1)
    rng = np.random.default_rng(seed)
    for epoch in range(EPOCHS):
        order = rng.permutation(len(y))
        model.partial_fit(Z[order], y[order], classes=CLASSES)
    return model


def metricas(y, prediction):
    scores = {'accuracy': accuracy_score(y, prediction)}
    for avg in ['micro', 'macro', 'weighted']:
        scores[f'precision_{avg}'] = precision_score(y, prediction, labels=CLASSES, average=avg, zero_division=0)
        scores[f'recall_{avg}'] = recall_score(y, prediction, labels=CLASSES, average=avg, zero_division=0)
        scores[f'f1_{avg}'] = f1_score(y, prediction, labels=CLASSES, average=avg, zero_division=0)
    return scores


# Busca de alpha: treino completo, avaliação na validação.
processor_full = criar_preprocessador(X_train)
Z_full = processor_full.fit_transform(X_train)
Z_val_full = processor_full.transform(X_val)
alpha_rows = []
for family, penalty in PENALTIES.items():
    if penalty is None:
        continue
    for alpha in ALPHAS:
        model = treinar_lr(Z_full, y_train, family, SEED, alpha=alpha)
        alpha_rows.append({'familia': family, 'alpha': alpha,
                           'treino_f1_macro': f1_score(y_train, model.predict(Z_full), average='macro'),
                           'validacao_f1_macro': f1_score(y_val, model.predict(Z_val_full), average='macro'),
                           'coeficientes_zero_pct': float(np.mean(model.coef_ == 0) * 100)})
alpha_search = pd.DataFrame(alpha_rows)
for family, group in alpha_search.groupby('familia'):
    best = group.sort_values(['validacao_f1_macro', 'alpha'], ascending=[False, False]).iloc[0]
    BEST_ALPHA[family] = float(best.alpha)
display(alpha_search.round(4).assign(alpha=alpha_search.alpha.map('{:g}'.format)))
print('Alpha escolhido por penalidade:', BEST_ALPHA)
alpha_search.to_csv(OUT / 'busca_alpha.csv', index=False)

fig, ax = plt.subplots()
for family, group in alpha_search.groupby('familia', sort=False):
    line, = ax.plot(group.alpha, group.validacao_f1_macro, 'o-', label=family)
    ax.plot(BEST_ALPHA[family], group.set_index('alpha').loc[BEST_ALPHA[family], 'validacao_f1_macro'],
            '*', ms=16, color=line.get_color())
ax.set(xscale='log', title='Busca de alpha — F1 macro na validação (★ = escolhido)',
       xlabel='alpha', ylabel='F1 macro (validação)')
ax.legend(); fig.tight_layout()
fig.savefig(FIG / 'busca_alpha.png', dpi=150)
plt.close(fig)
records, fitted = [], {}


def registrar(model, processor, sample, family, seed, fraction, kind, depth=None):
    key = f'{kind}|{family}|seed={seed}|fraction={fraction}|depth={depth}'
    Z = processor.transform(sample[FEATURES])
    metadata = {'modelo_id': key, 'familia': family, 'semente': seed,
                'fracao_treino': fraction, 'n_treino': len(sample),
                'experimento': kind, 'max_depth': depth,
                'alpha': BEST_ALPHA.get(family)}
    train_metrics = metricas(sample[TARGET], model.predict(Z))
    val_metrics = metricas(y_val, model.predict(processor.transform(X_val)))
    row = dict(metadata, **{'treino_' + k: v for k, v in train_metrics.items()},
               **{'validacao_' + k: v for k, v in val_metrics.items()})
    if family == 'AD propria':
        row.update(altura_real=model.depth_, nos=model.node_count_)
    else:
        row['coeficientes_zero_pct'] = float(np.mean(model.coef_ == 0) * 100)
    records.append(row)
    fitted[key] = (model, processor)
    return key

start = time.perf_counter()
# Curva de complexidade da AD: treino completo, profundidades 1 até a altura limite.
for depth in range(1, MAX_DEPTH + 1):
    tree = ArvoreDecisaoPropria(max_depth=depth, min_samples_leaf=8).fit(Z_full, y_train)
    registrar(tree, processor_full, train, 'AD propria', SEED, 1.0, 'curva_ad', depth)
tree_curve = pd.DataFrame(records).sort_values('max_depth')
# Profundidade da consistência: melhor F1 macro de validação (empate -> menor profundidade).
BEST_DEPTH = int(tree_curve.sort_values(['validacao_f1_macro', 'max_depth'],
                                        ascending=[False, True]).iloc[0].max_depth)
print('Profundidade escolhida para a AD:', BEST_DEPTH)

# Curvas da LR (alpha já escolhido) e consistência da AD nos mesmos subconjuntos.
for seed in SEEDS:
    for fraction in FRACTIONS:
        sample = POOLS[seed][fraction]
        processor = criar_preprocessador(sample[FEATURES])
        Z = processor.fit_transform(sample[FEATURES])
        y = sample[TARGET].to_numpy()
        for family in PENALTIES:
            model = treinar_lr(Z, y, family, seed)
            registrar(model, processor, sample, family, seed, fraction, 'curva_lr')
        if fraction in [0.30, 0.60, 0.90]:
            tree = ArvoreDecisaoPropria(max_depth=BEST_DEPTH, min_samples_leaf=8).fit(Z, y)
            registrar(tree, processor, sample, 'AD propria', seed, fraction, 'consistencia_ad', BEST_DEPTH)
    print('Semente concluída:', seed)

results = pd.DataFrame(records)
results['gap_f1_macro'] = results.treino_f1_macro - results.validacao_f1_macro
assert len(results) == 81 and results.modelo_id.is_unique
results.to_csv(OUT / 'treino_validacao_todos_casos.csv', index=False)
print('Modelos treinados:', len(results), 'Tempo (s):', round(time.perf_counter() - start, 2))
display(results.head())

secao('7. Consistência: 30%, 60% e 90% da partição de treino')
consistency = results[((results.experimento == 'curva_lr') & (results.familia == 'LR Ridge')
                       & results.fracao_treino.isin([0.30, 0.60, 0.90]))
                      | (results.experimento == 'consistencia_ad')].copy()
assert len(consistency) == 18
summary = consistency.groupby(['familia', 'fracao_treino', 'n_treino']).agg(
    treino_media=('treino_f1_macro', 'mean'), validacao_media=('validacao_f1_macro', 'mean'),
    validacao_desvio=('validacao_f1_macro', 'std'),
    validacao_min=('validacao_f1_macro', 'min'), validacao_max=('validacao_f1_macro', 'max'),
    gap_medio=('gap_f1_macro', 'mean')).reset_index()
display(summary)
paired = consistency.pivot(index=['semente', 'fracao_treino'], columns='familia', values='validacao_f1_macro')
paired['diferenca_AD_menos_LR'] = paired['AD propria'] - paired['LR Ridge']
display(paired)
summary.to_csv(OUT / 'consistencia_resumo.csv', index=False)
paired.to_csv(OUT / 'consistencia_pareada.csv')

secao('8. Cinco gráficos de curvas')
learning_plots = []
for index, family in enumerate(PENALTIES, start=1):
    subset = results[(results.experimento == 'curva_lr') & (results.familia == family)]
    aggregated = subset.groupby('n_treino')[['treino_f1_macro', 'validacao_f1_macro']].agg(['mean', 'std'])
    fig, ax = plt.subplots()
    x = aggregated.index.to_numpy()
    for column, label, color in [('treino_f1_macro', 'Treino', 'tab:blue'),
                                 ('validacao_f1_macro', 'Validação', 'tab:orange')]:
        mean = aggregated[column]['mean'].to_numpy()
        std = aggregated[column]['std'].to_numpy()
        ax.plot(x, mean, 'o-', label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)
    alpha_txt = '' if PENALTIES[family] is None else f' (alpha={BEST_ALPHA[family]:g})'
    ax.set(title=family + alpha_txt + ' — média ± desvio (3 sementes)',
           xlabel='Instâncias usadas da partição de treino', ylabel='F1 macro', ylim=(0, 1))
    ax.legend(); fig.tight_layout()
    name = f'curva_{index}_lr.png'
    fig.savefig(FIG / name, dpi=150); learning_plots.append(name)
    plt.close(fig)

tree_curve = results[results.experimento == 'curva_ad'].sort_values('max_depth')
fig, ax = plt.subplots()
ax.plot(tree_curve.max_depth, tree_curve.treino_f1_macro, 'o-', label='Treino')
ax.plot(tree_curve.max_depth, tree_curve.validacao_f1_macro, 'o-', label='Validação')
ax.set(title='AD própria — curva de complexidade', xlabel='Profundidade máxima permitida',
       ylabel='F1 macro', ylim=(0, 1), xticks=range(1, MAX_DEPTH + 1))
ax.legend(); fig.tight_layout()
name = 'curva_5_ad.png'
fig.savefig(FIG / name, dpi=150); learning_plots.append(name)
plt.close(fig)
assert len(learning_plots) == 5
display(tree_curve[['max_depth', 'altura_real', 'nos', 'treino_f1_macro', 'validacao_f1_macro', 'gap_f1_macro']])

secao('9. Seleção de três modelos antes do teste')
lr_full = results[(results.experimento == 'curva_lr') & (results.fracao_treino == 1.0)]
rank_regularized = lr_full[lr_full.familia != 'LR sem regularizacao'].groupby('familia').validacao_f1_macro.mean().sort_values(ascending=False, kind='stable')
best_regularized = rank_regularized.index[0]
selected = [lr_full[(lr_full.familia == family) & (lr_full.semente == SEED)].iloc[0].modelo_id
            for family in ['LR sem regularizacao', best_regularized]]
selected.append(tree_curve.sort_values(['validacao_f1_macro', 'max_depth'], ascending=[False, True]).iloc[0].modelo_id)
assert len(set(selected)) == 3
selection = results[results.modelo_id.isin(selected)].copy()
selection['criterio'] = selection.familia.map({
    'LR sem regularizacao': 'referencia sem penalidade; treino 100%; seed 42',
    best_regularized: 'melhor familia regularizada pela media de validacao; treino 100%; seed 42',
    'AD propria': 'melhor profundidade na validacao; empate favorece menor altura'})
selection.to_csv(OUT / 'selecao_antes_teste.csv', index=False)
display(selection[['modelo_id', 'validacao_f1_macro', 'criterio']])
SELECTION_FROZEN = True

secao('10. Avaliação final no teste de todos os casos')
assert SELECTION_FROZEN
X_test, y_test = test[FEATURES], test[TARGET].to_numpy()
test_rows, per_class, predictions = [], [], {}
for _, row in results.iterrows():
    key = row.modelo_id
    model, processor = fitted[key]
    pred = model.predict(processor.transform(X_test))
    predictions[key] = pred
    test_rows.append({'modelo_id': key, **metricas(y_test, pred)})
    report = classification_report(y_test, pred, labels=CLASSES, output_dict=True, zero_division=0)
    for label in CLASSES:
        per_class.append({'modelo_id': key, 'classe': label, **report[str(label)]})
test_metrics = results.merge(pd.DataFrame(test_rows), on='modelo_id', validate='one_to_one')
assert len(test_metrics) == len(results)
assert np.allclose(test_metrics.f1_micro, test_metrics.accuracy)
test_metrics.to_csv(OUT / 'metricas_teste_todos_casos.csv', index=False)
pd.DataFrame(per_class).to_csv(OUT / 'metricas_teste_por_classe.csv', index=False)
columns = ['familia', 'experimento', 'fracao_treino', 'max_depth', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro']
# Mostrar TODAS as configurações de curvas; repetições resumidas por média.
display(test_metrics[test_metrics.experimento == 'curva_lr'].groupby(['familia', 'fracao_treino'])[
    ['precision_macro', 'recall_macro', 'f1_micro', 'f1_macro']].agg(['mean', 'std']))
display(test_metrics[test_metrics.experimento == 'curva_ad'][columns])
baseline = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train), 1)), y_train)
baseline_scores = metricas(y_test, baseline.predict(np.zeros((len(test), 1))))
pd.DataFrame([baseline_scores]).to_csv(OUT / 'baseline_teste.csv', index=False)
print('Baseline majoritário:'); display(pd.DataFrame([baseline_scores]))
display(test_metrics[test_metrics.modelo_id.isin(selected)][columns])

secao('11. Três matrizes de confusão e análise de erros')
error_summaries = []
for index, key in enumerate(selected, start=1):
    matrix = confusion_matrix(y_test, predictions[key], labels=CLASSES)
    assert matrix.shape == (len(CLASSES), len(CLASSES)) and matrix.sum() == len(test)
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(OUT / f'confusao_{index}_contagens.csv')
    pd.DataFrame(normalized, index=CLASSES, columns=CLASSES).to_csv(OUT / f'confusao_{index}_normalizada.csv')
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=0)
    ax.set_title(key); ax.set_xlabel('Classe predita'); ax.set_ylabel('Classe real')
    ax.grid(False); fig.tight_layout()
    fig.savefig(FIG / f'matriz_confusao_{index}.png', dpi=150)
    plt.close(fig)
    error_pairs = sorted([(int(matrix[i, j]), str(CLASSES[i]), str(CLASSES[j]))
                          for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j and matrix[i, j] > 0], reverse=True)
    top = error_pairs[:3]
    recalls = dict(zip(CLASSES.tolist(), np.diag(normalized).tolist()))
    error_summaries.append({'modelo_id': key, 'principais_erros': top, 'recall_por_classe': recalls})
    print('Principais confusões (quantidade, real, predita):', top)
    display(pd.DataFrame({'classe': CLASSES, 'recall': np.diag(normalized), 'suporte': matrix.sum(axis=1)}))
# Exportar predições dos três selecionados para auditoria individual.
prediction_table = test[[NOTA, TARGET]].rename_axis('linha').reset_index()
for index, key in enumerate(selected, start=1):
    prediction_table[f'predicao_modelo_{index}'] = predictions[key]
prediction_table.to_csv(OUT / 'predicoes_selecionados.csv', index=False)

secao('12. Relatório de resultados')
selected_metrics = test_metrics.set_index('modelo_id').loc[selected]
lines = ['# Resultados da execução — GTA VI', '',
         '## Configuração',
         f'{len(df)} registros; alvo {TARGET} com 5 classes ({", ".join(f"{k}: notas {min(v)}–{max(v)}" if len(v) > 1 else f"{k}: nota {v[0]}" for k, v in FAIXAS.items())}). '
         f'Removidos de X: respondent_id (identificador), overall_sentiment (derivado da nota) e {NOTA}.', '',
         f'Treino={len(train)}; validação={len(val)}; teste={len(test)}. Sementes={SEEDS}; épocas SGD={EPOCHS}; '
         f'alpha escolhido: {BEST_ALPHA}; profundidade AD escolhida={BEST_DEPTH} (limite {MAX_DEPTH}). '
         f'{len(results)} configurações avaliadas no teste.', '',
         '## Busca de alpha (validação, F1 macro)', alpha_search.round(4).assign(alpha=alpha_search.alpha.map('{:g}'.format)).to_markdown(index=False), '',
         '## Métricas dos três modelos selecionados (teste)',
         '|Modelo|Precision macro|Recall macro|F1 micro|F1 macro|Gap treino-validação|',
         '|---|---:|---:|---:|---:|---:|']
for key, row in selected_metrics.iterrows():
    safe_key = key.replace('|', '&#124;')
    lines.append(f'|{safe_key}|{row.precision_macro:.4f}|{row.recall_macro:.4f}|{row.f1_micro:.4f}|{row.f1_macro:.4f}|{row.gap_f1_macro:.4f}|')
lines += ['', f'Baseline majoritário: F1 micro={baseline_scores["f1_micro"]:.4f}; '
          f'F1 macro={baseline_scores["f1_macro"]:.4f}.', '', '## Curvas']
for family in PENALTIES:
    curve = results[(results.experimento == 'curva_lr') & (results.familia == family)]
    first = curve[curve.fracao_treino == min(FRACTIONS)]
    last = curve[curve.fracao_treino == 1]
    lines.append(f'- {family}: validação F1 macro {first.validacao_f1_macro.mean():.4f} ({first.n_treino.iloc[0]} exemplos) → '
                 f'{last.validacao_f1_macro.mean():.4f} ({last.n_treino.iloc[0]}); gap final {last.gap_f1_macro.mean():.4f}.')
lines += [f'- AD: treino {tree_curve.iloc[0].treino_f1_macro:.4f} → {tree_curve.iloc[-1].treino_f1_macro:.4f}; '
          f'validação {tree_curve.iloc[0].validacao_f1_macro:.4f} → {tree_curve.iloc[-1].validacao_f1_macro:.4f} '
          f'(profundidade 1 → {MAX_DEPTH}); melhor validação na profundidade {BEST_DEPTH}.', '',
          '## Consistência', summary.round(4).to_markdown(index=False), '',
          '## Erros dos selecionados']
for item in error_summaries:
    lines += [f'### {item["modelo_id"]}',
              'Confusões principais: ' + '; '.join(f'{real} → {pred}: {count}' for count, real, pred in item['principais_erros']),
              'Recall por classe: ' + '; '.join(f'{k}={v:.4f}' for k, v in item['recall_por_classe'].items()), '']
lines += ['## Reprodutibilidade', json.dumps(VERSIONS, indent=2)]
report_text = '\n'.join(lines)
(OUT / 'relatorio_resultados.md').write_text(report_text, encoding='utf-8')
manifest = {'dataset': str(DATASET.relative_to(ROOT)),
            'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
            'shape_sem_id': list(df.shape), 'target': TARGET, 'faixas': {k: list(v) for k, v in FAIXAS.items()},
            'removidos_de_X': REMOVIDOS + ['respondent_id'],
            'split_sizes': {'train': len(train), 'validation': len(val), 'test': len(test)},
            'seeds': SEEDS, 'fractions': FRACTIONS, 'epochs': EPOCHS, 'alphas_testados': ALPHAS,
            'alpha_escolhido': BEST_ALPHA, 'profundidade_escolhida': BEST_DEPTH,
            'tree_depth_limit': MAX_DEPTH, 'configurations': len(results),
            'learning_plots': learning_plots, 'selected_model_ids': selected,
            'selection_uses_test': False, 'features': FEATURES, 'versions': VERSIONS}
(OUT / 'manifesto.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

print('Artefatos gravados em:', OUT)

secao('Checklist do enunciado')

secao('RESUMO — MÉTRICAS NO TESTE')
cols = ['precision_macro', 'recall_macro', 'precision_micro', 'recall_micro', 'f1_micro', 'f1_macro']
print('\nCurvas LR (média das 3 sementes por tamanho de treino):')
display(test_metrics[test_metrics.experimento == 'curva_lr'].groupby(['familia', 'n_treino'])[cols].mean().round(4))
print('Curva AD (profundidade):')
display(test_metrics[test_metrics.experimento == 'curva_ad'].set_index('max_depth')[cols].round(4))
print('Consistência (30/60/90% do treino, média das 3 sementes):')
display(test_metrics[((test_metrics.experimento == 'curva_lr') & (test_metrics.familia == 'LR Ridge')
                      & test_metrics.fracao_treino.isin([0.3, 0.6, 0.9])) | (test_metrics.experimento == 'consistencia_ad')]
        .groupby(['familia', 'fracao_treino'])[cols].mean().round(4))
print('Três modelos selecionados:')
display(test_metrics.set_index('modelo_id').loc[selected, cols].round(4))
print('Baseline:', {k: round(baseline_scores[k], 4) for k in cols})
print('\nFiguras e CSVs em:', OUT)
