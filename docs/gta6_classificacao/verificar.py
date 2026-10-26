"""Testes da árvore própria extraída do notebook, sem executar os experimentos."""
from pathlib import Path
import json
import unittest
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'notebooks/Gta6ClassificacaoTrabalho.ipynb'


def carregar_arvore():
    assert NOTEBOOK.exists(), 'Notebook exclusivo e árvore própria ainda não implementados'
    notebook = json.loads(NOTEBOOK.read_text())
    cells = [c for c in notebook['cells'] if 'arvore-propria' in c.get('metadata', {}).get('tags', [])]
    assert len(cells) == 1, 'Esperada uma célula com a implementação completa da árvore'
    namespace = {}
    exec(''.join(cells[0]['source']), namespace)
    return namespace['ArvoreDecisaoPropria']


class TestArvore(unittest.TestCase):
    def test_cortes_finitos_extremos_e_adjacentes(self):
        cls = carregar_arvore()
        pairs = [(-1e308, 1e308), (1e308, np.nextafter(1e308, np.inf)),
                 (0.0, np.nextafter(0.0, 1.0))]
        for a, b in pairs:
            with self.subTest(a=a, b=b), np.errstate(all='raise'):
                x = np.array([[a], [b]])
                model = cls(max_depth=1, min_samples_leaf=1).fit(x, ['a', 'b'])
                np.testing.assert_array_equal(model.predict(x), ['a', 'b'])
                self.assertTrue(np.isfinite(model.root_['threshold']))
                np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)

    def test_treina_prediz_multiclasse(self):
        cls = carregar_arvore()
        x = np.arange(15, dtype=float).reshape(-1, 1)
        y = np.repeat(['a', 'b', 'c', 'd', 'e'], 3)
        model = cls(max_depth=6, min_samples_leaf=1).fit(x, y)
        np.testing.assert_array_equal(model.predict(x), y)
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)

    def test_limite_profundidade_e_empate(self):
        cls = carregar_arvore()
        x = np.arange(6).reshape(-1, 1)
        y = np.array(['b', 'a', 'b', 'a', 'b', 'a'])
        model = cls(max_depth=0, min_samples_leaf=1).fit(x, y)
        self.assertEqual(model.depth_, 0)
        self.assertEqual(model.node_count_, 1)
        np.testing.assert_array_equal(model.predict(x), ['a'] * 6)

    def test_esparsa_constante_e_classe_unica(self):
        cls = carregar_arvore()
        x = sparse.csr_matrix(np.ones((8, 2)))
        model = cls(min_samples_leaf=1).fit(x, ['z'] * 8)
        np.testing.assert_array_equal(model.predict(x), ['z'] * 8)
        self.assertEqual(model.node_count_, 1)

    def test_erros_de_entrada(self):
        cls = carregar_arvore()
        with self.assertRaises(ValueError):
            cls().predict([[0]])
        with self.assertRaises(ValueError):
            cls().fit([[np.nan]], ['a'])
        with self.assertRaises(ValueError):
            cls().fit([[0]], [])
        with self.assertRaises(ValueError):
            cls().fit([[0]], ['a']).predict([[0, 1]])


class TestPreprocessamento(unittest.TestCase):
    def test_vocabulario_nao_aprende_validacao(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = [c for c in notebook['cells'] if 'preprocessamento' in c.get('metadata', {}).get('tags', [])]
        self.assertEqual(len(cells), 1, 'Pré-processamento isolado ainda não implementado')
        namespace = {}
        exec(''.join(cells[0]['source']), namespace)
        import pandas as pd
        train = pd.DataFrame({'most_anticipated_feature': ['driving physics', 'story depth'], 'age': [20, 30]})
        val = pd.DataFrame({'most_anticipated_feature': ['sentinelvalidation'], 'age': [40]})
        processor = namespace['criar_preprocessador'](train)
        processor.fit(train)
        vocabulary = dict(processor.named_transformers_['texto'].vocabulary_)
        processor.transform(val)
        self.assertNotIn('sentinelvalidation', vocabulary)
        self.assertEqual(processor.named_transformers_['texto'].vocabulary_, vocabulary)


class TestEntrega(unittest.TestCase):
    def test_alvo_hype_e_features_sem_vazamento(self):
        import pandas as pd
        notebook = json.loads(NOTEBOOK.read_text())
        namespace = {'__name__': '__test__'}
        setup = next(c for c in notebook['cells'] if c['cell_type'] == 'code')
        # Inspeciona a atribuição sem disparar saídas ou criar artefatos.
        import ast
        statements = ast.parse(''.join(setup['source'])).body
        target = next(s.value.value for s in statements if isinstance(s, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'TARGET' for t in s.targets))
        self.assertEqual(target, 'hype_score_1to10')
        namespace.update(df=pd.read_csv(ROOT / 'datasets/gta6_fan_expectations_2026.csv'),
                         TARGET=target, SEED=42, SEEDS=[42, 43, 44],
                         FRACTIONS=[0.10, 0.30, 0.60, 0.90, 1.0], np=np,
                         pd=pd, display=lambda *args: None)
        namespace['CLASSES'] = np.sort(namespace['df'][target].unique())
        from sklearn.model_selection import train_test_split
        from tempfile import TemporaryDirectory
        namespace['train_test_split'] = train_test_split
        split_cell = notebook['cells'][5]
        with TemporaryDirectory(dir=ROOT / '.venv') as directory:
            namespace['OUT'] = Path(directory)
            exec(''.join(split_cell['source']), namespace)
        self.assertNotIn(target, namespace['FEATURES'])
        self.assertNotIn('respondent_id', namespace['FEATURES'])
        self.assertIn('primary_platform', namespace['FEATURES'])
        self.assertEqual(set(namespace['train'][target]), set(range(1, 11)))
        self.assertEqual(namespace['train'][target].eq(1).sum(), 1)
        self.assertEqual(namespace['val'][target].eq(1).sum(), 0)
        self.assertEqual(namespace['test'][target].eq(1).sum(), 1)

    def test_graficos_embutidos_e_execucao(self):
        notebook = json.loads(NOTEBOOK.read_text())
        code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertTrue(all(c['execution_count'] is not None for c in code_cells))
        outputs = [o for c in code_cells for o in c['outputs']]
        self.assertFalse(any(o['output_type'] == 'error' for o in outputs))
        images = [o for o in outputs if 'image/png' in o.get('data', {})]
        self.assertEqual(len(images), 9, 'EDA + cinco curvas + três matrizes precisam estar embutidas')

    def test_resultados_completos_e_particoes(self):
        import pandas as pd
        out = ROOT / 'docs/gta6_classificacao/resultados'
        manifest = json.loads((out / 'manifesto.json').read_text())
        self.assertEqual(manifest['target'], 'hype_score_1to10')
        self.assertEqual(manifest['classes'], list(range(1, 11)))
        self.assertFalse(manifest['full_assignment_compliance'])
        scores = pd.read_csv(out / 'metricas_teste_todos_casos.csv')
        self.assertEqual(len(scores), 81)
        self.assertEqual(scores.modelo_id.nunique(), 81)
        self.assertEqual(len(pd.read_csv(out / 'metricas_teste_por_classe.csv')), 810)
        self.assertEqual(len(pd.read_csv(out / 'selecao_antes_teste.csv')), 3)
        partitions = pd.read_csv(out / 'particoes_ids.csv')
        self.assertEqual(partitions.respondent_id.nunique(), 6000)
        self.assertEqual(partitions.particao.value_counts().to_dict(),
                         {'treino': 3600, 'validacao': 1200, 'teste': 1200})
        for index in range(1, 4):
            matrix = pd.read_csv(out / f'confusao_{index}_contagens.csv', index_col=0)
            self.assertEqual(matrix.shape, (10, 10))
            self.assertEqual(matrix.to_numpy().sum(), 1200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
