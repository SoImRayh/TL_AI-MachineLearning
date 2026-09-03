# AI Projects

Repositório para projetos acadêmicos e pessoais de Inteligência Artificial e Machine Learning.

## Estrutura

- `datasets/`: bases de dados locais e arquivos de entrada.
- `images/`: imagens usadas nos projetos, notebooks e documentação.
- `notebooks/`: notebooks Jupyter para exploração, estudos e experimentos.
- `projects/`: projetos independentes, organizados por tema.
- `setup_venv.sh`: cria o ambiente virtual e instala as dependências básicas.

## Configuração rápida

```bash
./setup_venv.sh
source .venv/bin/activate
```

Para iniciar o Jupyter Lab:

```bash
jupyter lab
```

## Bibliotecas instaladas

O script instala Python, NumPy, pandas, scikit-learn, SciPy, Matplotlib, Seaborn e JupyterLab.

Os arquivos em `datasets/` e `images/` são ignorados pelo Git por padrão para evitar o versionamento acidental de arquivos grandes. A estrutura dos diretórios é preservada por arquivos `.gitkeep`.
