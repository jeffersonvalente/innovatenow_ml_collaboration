# innovatenow_ml_collaboration

Projeto de exemplo da **InnovateNow Tech** para evolução progressiva em **MLOps**, cobrindo desde a configuração inicial de ambiente até a criação de **módulos reutilizáveis**, **classes com type hints** e **testes unitários**, seguindo um fluxo realista de tarefas incrementais.

O projeto foi desenvolvido em **4 tarefas**, cada uma construindo sobre a anterior, mantendo **continuidade cronológica**, histórico de commits limpo e boas práticas de engenharia.

---

## ✅ Tarefa 1 — Configuração Inicial do Ambiente de Desenvolvimento

### Contexto
Primeiro contato com o time de MLOps da InnovateNow Tech. O foco é garantir um ambiente consistente para evitar problemas de dependência ("funciona na minha máquina").

### Objetivos
- Criar ambiente virtual Python
- Gerenciar dependências
- Utilizar Git para versionamento básico

### Implementações
- Criação do ambiente virtual (`venv`)
- Instalação de `pandas` e `scikit-learn`
- Geração do `requirements.txt`
- Inicialização de repositório Git
- Criação de `.gitignore`
- Criação de `main.py` exibindo versões das bibliotecas

### Estrutura Inicial
```text
innovatenow_ml_env/
├── venv/
├── .gitignore
├── main.py
└── requirements.txt
```

### Execução
```bash
python main.py
```

---

## ✅ Tarefa 2 — Controle de Versão e Colaboração com Git

### Contexto
Simulação de colaboração em equipe usando **branches**, **merges** e **resolução de conflitos**.

### Objetivos
- Trabalhar com feature branches
- Criar commits granulares
- Resolver conflitos de merge

### Implementações
- Criação do repositório `innovatenow_ml_collaboration`
- Cópia do conteúdo da Tarefa 1
- Criação da branch `feat/add-data-prep`
- Novo módulo `data_preprocessing.py`
- Integração no `main.py`
- Simulação e resolução manual de conflito
- Merge da feature branch na `main`

### Novo módulo
- `data_preprocessing.py`
  - `load_data()`
  - `preprocess_data()`

---

## ✅ Tarefa 3 — Fundamentos de Python para MLOps (Classes e Módulos)

### Contexto
Introdução à modularização real de pipelines de ML usando **classes**, **tipagem estática** e **testes unitários**.

### Objetivos
- Criar módulos reutilizáveis
- Encapsular lógica em classes
- Introduzir testes automatizados

### Implementações
- Criação da branch `feat/data-splitter-module`
- Estrutura `src/` com `utils/`
- Classe `DataSplitter`
- Testes unitários com `pytest`
- Configuração de `pytest.ini`
- Integração no `main.py`

### Estrutura
```text
innovatenow_ml_collaboration/
├── src/
│   └── utils/
│       └── data_splitter.py
├── tests/
│   └── test_data_splitter.py
├── main.py
├── requirements.txt
├── pytest.ini
├── .gitignore
└── venv/
```

### Execução de Testes
```bash
pytest -q
```

---

## ✅ Tarefa 4 — Manipulação de Dados com Pandas (DataProcessor)

### Contexto
Simulação de uma etapa real de **engenharia de features**, limpeza e transformação de dados antes de modelos de ML.

### Objetivos
- Tratamento de valores ausentes
- Normalização de features numéricas
- Codificação de variáveis categóricas
- Continuidade de boas práticas de testes

### Implementações
- Criação da branch `feat/data-processor-module`
- Novo módulo `data_processor.py`
- Classe `DataProcessor` com métodos:
  - `handle_missing_values()`
  - `normalize_features()`
  - `encode_categorical()`
- Testes unitários em `tests/test_data_processor.py`
- Integração completa no `main.py`

### Estrutura Atual
```text
innovatenow_ml_collaboration/
├── src/
│   └── utils/
│       ├── data_splitter.py
│       └── data_processor.py
├── tests/
│   ├── test_data_splitter.py
│   └── test_data_processor.py
├── main.py
├── data_preprocessing.py
├── requirements.txt
├── pytest.ini
├── .gitignore
└── venv/
```

### Execução do Pipeline Principal
```bash
python main.py
```

### Execução dos Testes
```bash
pytest -q
```

---

## ✅ Boas Práticas Aplicadas

- Commits seguindo **Conventional Commits**
- Código modular e reutilizável
- Uso consistente de `type hints`
- Testes unitários cobrindo casos de sucesso e erro
- Estrutura profissional baseada em projetos reais de MLOps

---

## 📌 Observações Finais

Este repositório representa um **crescimento progressivo e realista** em MLOps, desde setup inicial até engenharia de dados testável, refletindo práticas usadas em ambientes profissionais.

👉 Ideal como base para:
- Pipelines de ML mais complexos
- Integração futura com modelos
- CI/CD e automação
