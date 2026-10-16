"""Executa apenas o notebook exclusivo em kernel Python novo e salva saídas."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'notebooks/Gta6ClassificacaoTrabalho.ipynb'
notebook = nbformat.read(PATH, as_version=4)
nbformat.validate(notebook)
manager = KernelManager(kernel_name='python3', transport='ipc')
spec = manager.kernel_spec
assert spec is not None
spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, km=manager, timeout=1800, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT)}},
                        allow_errors=False)
# Usa exatamente o ambiente Python que chamou este script, sem instalar kernel
# global nem alterar configuração de outros notebooks.
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    if client.kc is not None:
        client.kc.stop_channels()
    nbformat.write(notebook, PATH)
errors = [output for cell in notebook.cells if cell.cell_type == 'code'
          for output in cell.outputs if output.output_type == 'error']
assert not errors
assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code')
print(f'Executado e salvo: {PATH}; células de código: '
      f'{sum(cell.cell_type == "code" for cell in notebook.cells)}; erros: {len(errors)}')
