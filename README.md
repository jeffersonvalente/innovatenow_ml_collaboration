# innovatenow_ml_env

## Descrição

Este projeto tem como objetivo configurar um ambiente inicial de desenvolvimento em Python para projetos de Machine Learning, seguindo boas práticas de isolamento de dependências e controle de versão com Git.

O repositório demonstra:
- Uso de ambiente virtual (`venv`)
- Gerenciamento de dependências com `pip`
- Estrutura básica de um projeto Python
- Versionamento inicial com Git

## Estrutura do Projeto

```
innovatenow_ml_env/
├── .gitignore
├── main.py
├── requirements.txt
└── venv/
```

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- Python 3.8 ou superior
- Git
- Acesso à internet para instalação de pacotes

Para verificar:
```bash
python --version
git --version
```

## Configuração do Ambiente

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd innovatenow_ml_env
```

### 2. Criar o ambiente virtual

```bash
python -m venv venv
```

### 3. Ativar o ambiente virtual

Linux / macOS:
```bash
source venv/bin/activate
```

Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

### 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

## Dependências

As dependências do projeto estão listadas no arquivo `requirements.txt`. As principais bibliotecas utilizadas são:

- pandas
- scikit-learn

## Execução do Projeto

Com o ambiente virtual ativado, execute:

```bash
python main.py
```

A saída esperada inclui:
- A versão instalada do pandas
- A versão instalada do scikit-learn
- Uma mensagem confirmando que o ambiente foi configurado com sucesso

## Observações

- O diretório do ambiente virtual (`venv/`) está corretamente ignorado no versionamento via `.gitignore`
- Todo o código e arquivos de configuração relevantes estão versionados
- O histórico de commits reflete cada etapa da configuração do ambiente

## Referências

- Python venv: https://docs.python.org/3/library/venv.html
- pip freeze: https://pip.pypa.io/en/stable/cli/pip_freeze/
- Git: https://git-scm.com/docs
- Aprender Git (interativo): https://learngitbranching.js.org/
