# innovatenow_ml_collaboration

Projeto de exemplo da InnovateNow Tech para praticar **colaboração com Git** usando um workflow básico de branches, commits, merge e **resolução de conflito** (Tarefa 2) e, em seguida, evoluir para **fundamentos de Python aplicados a MLOps** com **módulos reutilizáveis**, **classes**, **type hints** e **testes unitários** (Tarefa 3).

Na **Tarefa 2**, foi adicionada uma etapa simples de **preparação de dados** via um módulo separado (`data_preprocessing.py`) e sua integração ao `main.py`.

Na **Tarefa 3**, foi criado um módulo reutilizável para **divisão de dados em treino e teste** (`DataSplitter`), organizado em estrutura `src/`, com **tratativa de erros**, **type hints** e **testes com pytest**, além da integração no `main.py`.

## Estrutura do projeto

- `main.py` — Script principal do projeto (integra data prep + data splitting).
- `data_preprocessing.py` — Funções para carregar/criar dados dummy e pré-processar.
- `src/`
  - `utils/`
    - `data_splitter.py` — Classe `DataSplitter` para split de `pd.DataFrame` em treino/teste.
- `tests/`
  - `test_data_splitter.py` — Testes unitários para a classe `DataSplitter`.
- `requirements.txt` — Dependências Python (inclui `pytest`).
- `.gitignore` — Arquivos/pastas ignorados pelo Git (inclui `venv/`, `dummy_data.csv` e `__pycache__/` em `src/` e `tests/`).

## Requisitos

- Python 3.8+
- Git
- Ambiente virtual (recomendado)

## Configuração do ambiente

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como executar

Com o `venv` ativado, a partir da **raiz do repositório**:

```bash
python main.py
```

### O que acontece ao executar?

- O `main.py` executa o fluxo principal do projeto.
- O módulo `data_preprocessing.py` é importado e usado.
- Se o arquivo `dummy_data.csv` **não existir**, ele será **criado automaticamente** para fins de teste.
- O dataset é carregado e passa por um pré-processamento simples (ex.: padronização de uma feature numérica).
- Em seguida, o `DataSplitter` (em `src/utils/data_splitter.py`) é utilizado para dividir os dados em **treino** e **teste** usando `sklearn.model_selection.train_test_split`.

> Observação: `dummy_data.csv` é um artefato gerado localmente e por isso está no `.gitignore` (não deve ser versionado).

## Como rodar os testes (pytest)

A partir da raiz do projeto, com o `venv` ativado:

```bash
pytest -q
```

Ou, alternativamente:

```bash
python -m pytest -q
```

### Observação sobre imports (estrutura `src/`)

Os testes e o `main.py` importam a classe assim:

- `from src.utils.data_splitter import DataSplitter`

Por isso, execute `python main.py` e `pytest` **a partir da raiz do repositório** (onde existe a pasta `src/`). Se o seu ambiente reclamar de import, uma solução simples é definir o `PYTHONPATH` apontando para a raiz do projeto ao executar (exemplo Linux/macOS):

```bash
PYTHONPATH=. pytest -q
```

## Tarefa 3 — Fundamentos de Python para MLOps (Classes e Módulos)

Nesta tarefa, o objetivo foi criar um módulo reutilizável para uma etapa fundamental de pipelines de ML: **divisão de dados em treino e teste**, com boas práticas de engenharia:

- **Classe** para encapsular a lógica (`DataSplitter`)
- **Módulo reutilizável** em `src/utils/`
- **Type hints** em entradas e saídas
- **Tratativa básica de erros**
- **Testes unitários** com `pytest`
- Integração com o `main.py`

### Implementações principais

- **Nova feature branch**: `feat/data-splitter-module`
- **Novo módulo**: `src/utils/data_splitter.py`
  - Classe `DataSplitter`:
    - Recebe `pd.DataFrame` no construtor
    - Faz cópia do dataframe para evitar modificação do original
    - Método `split(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]`
    - Validações:
      - DataFrame vazio
      - `test_size` inválido (fora de `(0, 1)`)
      - Tipo de entrada inválido (não-DataFrame)
- **Integração no `main.py`**
  - Import do `DataSplitter`
  - Uso do `DataSplitter` para dividir um DataFrame de exemplo (ex.: vindo do `load_data()`)
- **Testes**: `tests/test_data_splitter.py`
  - Verifica:
    - Retorno de dois DataFrames
    - Soma dos tamanhos = tamanho original
    - Proporção aproximada do `test_size`
    - Reprodutibilidade via `random_state`
    - Erros para DataFrame vazio, `test_size` inválido e entrada não-DataFrame
- **Atualizações de housekeeping**
  - `.gitignore`: adicionados `src/__pycache__/` e `tests/__pycache__/`
  - `requirements.txt`: garantido `pytest` (e dependências como `pandas` e `scikit-learn`)

## Tarefa 2 — Workflow Git e Resolução de Conflito (resumo)

Esta tarefa simula colaboração real: uma feature é desenvolvida em uma branch separada enquanto mudanças diferentes são feitas na `main`, causando um conflito proposital.

### Passos executados (alto nível)

1. Repo novo e vazio no GitHub
2. Clone e setup inicial (conteúdo da Tarefa 1)
3. Criação de branch `feat/add-data-prep`
4. Implementação do `data_preprocessing.py` + integração no `main.py`
5. Mudança conflitante em `main.py` diretamente na `main`
6. Merge com conflito e resolução preservando as duas intenções
7. Ajustes finais: `.gitignore`, README e push

## Histórico de commits esperado (referência)

### Tarefa 2

- `feat: Initial project setup from Task 1`
- `feat: Add data preprocessing module` (na `feat/add-data-prep`)
- `feat: Integrate data preprocessing into main script` (na `feat/add-data-prep`)
- `feat: Add conflicting message to main to simulate parallel work` (na `main`)
- `Merge feat/add-data-prep into main with conflict resolution` (na `main`)
- `chore: Add dummy_data.csv to .gitignore`
- `docs: Update README with project setup and Git workflow for Task 2`

### Tarefa 3 (exemplo de sequência)

- `feat: Create DataSplitter module in src/utils`
- `test: Add unit tests for DataSplitter`
- `feat: Integrate DataSplitter into main`
- `chore: Update .gitignore for src/tests __pycache__`
- `chore: Add pytest to requirements.txt`
- `docs: Update README for Task 3 (module + tests)`
- `Merge feat/data-splitter-module into main`
