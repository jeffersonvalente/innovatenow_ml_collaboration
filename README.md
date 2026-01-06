# innovatenow_ml_collaboration

Projeto de exemplo da InnovateNow Tech para praticar **colaboração com Git** usando um workflow básico de branches, commits, merge e **resolução de conflito**.

Este repositório foi inicializado a partir do conteúdo da **Tarefa 1 (`innovatenow_ml_env`)** e, nesta **Tarefa 2**, foi adicionada uma etapa simples de **preparação de dados** via um módulo separado (`data_preprocessing.py`) e sua integração ao `main.py`.

## Estrutura do projeto

- `main.py` — Script principal do projeto (agora chama o módulo de data prep).
- `data_preprocessing.py` — Funções para carregar/criar dados dummy e pré-processar.
- `requirements.txt` — Dependências Python.
- `.gitignore` — Arquivos/pastas ignorados pelo Git (inclui `venv/` e `dummy_data.csv`).

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
.env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como executar

Com o `venv` ativado:

```bash
python main.py
```

### O que acontece ao executar?

- O `main.py` executa o fluxo principal do projeto.
- O módulo `data_preprocessing.py` é importado e usado pelo `main.py`.
- Se o arquivo `dummy_data.csv` **não existir**, ele será **criado automaticamente** para fins de teste.
- O dataset é carregado e passa por um pré-processamento simples (ex.: padronização de uma feature numérica).

> Observação: `dummy_data.csv` é um artefato gerado localmente e por isso está no `.gitignore` (não deve ser versionado).

## Tarefa 2 — Workflow Git e Resolução de Conflito (resumo)

Esta tarefa simula colaboração real: uma feature é desenvolvida em uma branch separada enquanto mudanças diferentes são feitas na `main`, causando um conflito proposital.

### Passos executados

1. **Repo novo e vazio no GitHub**
   - Repositório criado como `innovatenow_ml_collaboration` (sem README/.gitignore/licença inicialmente).

2. **Clone e setup inicial**
   - Clone do repo vazio.
   - Cópia dos arquivos da Tarefa 1 (exceto `venv/`).
   - Commit na branch `main`:
     - `feat: Initial project setup from Task 1`
   - Push para o remoto.

3. **Criação da feature branch**
   - Branch criada a partir da `main`:
     - `feat/add-data-prep`

4. **Implementação na feature branch**
   - Criação do módulo:
     - `data_preprocessing.py`
   - Commit:
     - `feat: Add data preprocessing module`
   - Integração no `main.py` (chamada das funções de load/preprocess no final do arquivo)
   - Commit:
     - `feat: Integrate data preprocessing into main script`

5. **Mudança conflitante na main**
   - Volta para `main`.
   - Alteração deliberada no `main.py` no mesmo trecho final (para simular outro dev trabalhando em paralelo).
   - Commit:
     - `feat: Add conflicting message to main to simulate parallel work`

6. **Merge com conflito**
   - Merge de `feat/add-data-prep` para `main`:
     - `git merge feat/add-data-prep`
   - Conflito gerado em `main.py`.

7. **Como o conflito foi resolvido**
   - O arquivo `main.py` foi aberto e as marcações do Git removidas:
     - `<<<<<<< HEAD`
     - `=======`
     - `>>>>>>> feat/add-data-prep`
   - A resolução preservou **as duas intenções**:
     - a integração do módulo `data_preprocessing.py`
     - a mensagem adicionada na `main` para simular execução principal
   - Finalização do merge:
     - `git add main.py`
     - `git commit` (merge commit)

8. **Ajustes finais**
   - `dummy_data.csv` adicionado ao `.gitignore`
   - Commit:
     - `chore: Add dummy_data.csv to .gitignore`
   - README atualizado (este arquivo)
   - Commit:
     - `docs: Update README with project setup and Git workflow for Task 2`
   - Push final da `main` para o remoto.

## Histórico de commits esperado (referência)

- `feat: Initial project setup from Task 1`
- `feat: Add data preprocessing module` (na `feat/add-data-prep`)
- `feat: Integrate data preprocessing into main script` (na `feat/add-data-prep`)
- `feat: Add conflicting message to main to simulate parallel work` (na `main`)
- `Merge feat/add-data-prep into main with conflict resolution` (na `main`)
- `chore: Add dummy_data.csv to .gitignore`
- `docs: Update README with project setup and Git workflow for Task 2`
