# 🔎 Motor de Inferência — Classificação Automática de Setores

Programa em Python que lê **textos**, usa um **modelo de Machine Learning**
(TF-IDF + Regressão Logística) para classificar cada texto em um dos setores e
encaminha o resultado para a área correta:

- 🟦 **comercial**
- 🟩 **financeiro**
- 🟨 **auditoria**
- 🟥 **suporte**

O que o modelo **não souber** classificar com segurança é enviado para **manual**
(revisão humana). O resultado é registrado em um **relatório** e o arquivo é movido
para a pasta do setor correspondente.

---

## 🧠 Como funciona

```
entrada/*.txt
   |
   v
normalizacao do texto (minusculas, espacos)
   |
   v
TF-IDF  ->  Regressao Logistica   (modelo salvo em modelos/)
   |
   v
Regra "nao sei -> manual"?  (confianca e margem)
   |            |
   | nao        | sim
   v            v
motor de regras (experta)   ->   manual
   |
   v
relatorio (relatorios/)  +  move arquivo para processado/<setor>/
```

### Regra do "não sei" (manual)

Um texto vai para `manual` quando o modelo fica **inseguro**:

- confiança da classe vencedora **< 0,55**, **ou**
- margem entre a 1ª e a 2ª classe **< 0,15**

Os limiares ficam no topo do arquivo `motor_inferencia.py` e podem ser ajustados.

---

## 🗂 Para que serve cada pasta

| Pasta / arquivo         | Descrição                                                        |
|-------------------------|------------------------------------------------------------------|
| `entrada/`              | Onde você coloca os arquivos a processar (`.txt`).               |
| `processado/<setor>/`   | Arquivos já classificados e movidos (`comercial`, `financeiro`, `auditoria`, `suporte`, `manual`). |
| `relatorios/`           | Um relatório `.txt` por arquivo processado (probabilidades + decisões). |
| `modelos/`              | Modelo treinado (`modelo.joblib` e `vectorizer.joblib`).         |
| `dados/`                | `exemplos_treino.json` — frases rotuladas usadas no treino.       |
| `regras/regras.py`      | Motor de regras (forward chaining) com **experta**.              |
| `motor_inferencia.py`   | **Motor de inferência** (arquivo principal).                     |
| `treinar_modelo.py`     | Script para treinar/avaliar o modelo.                           |
| `pipeline.py`           | Ponto de entrada — usa o motor de inferência.                   |

---

## ✅ Requisitos

- **Python 3.10+** (testado em Python 3.13)
- Bibliotecas:

| Pacote          | Para quê                          | Obrigatório |
|-----------------|-----------------------------------|-------------|
| `scikit-learn`  | TF-IDF + Regressão Logística      | Sim         |
| `joblib`        | Salvar/carregar o modelo          | Sim         |
| `numpy`         | Dependência do scikit-learn       | Sim         |
| `experta`       | Motor de regras (forward chaining)| Sim         |
| `pytesseract`   | OCR de imagem                     | Só p/ OCR   |
| `pdf2image`     | OCR de PDF                        | Só p/ OCR   |
| `pillow`        | Leitura de imagens                | Só p/ OCR   |

> ⚠️ **OCR:** para ler arquivos `.png/.jpg/.jpeg/.pdf` é preciso ter o
> **Tesseract** e o **Poppler** instalados no sistema (não são pacotes Python).
> Sem eles, use arquivos `.txt`.

### Instalação

```powershell
pip install scikit-learn joblib numpy experta

# Opcional (apenas para OCR de imagem/PDF):
pip install pytesseract pdf2image pillow
```

---

## 🚀 Como usar

### 1) Coloque os arquivos em `entrada/`

Salve o texto a classificar como um arquivo `.txt` dentro da pasta `entrada/`.
Exemplo: `entrada/chamado_001.txt`.

### 2) (Opcional) Treine o modelo

O modelo já é treinado automaticamente na 1ª execução. Para treinar/re-treinar e
ver as métricas:

```powershell
python treinar_modelo.py
```

### 3) Rode o motor de inferência

```powershell
python pipeline.py
```

(ou, equivalentemente, `python motor_inferencia.py`)

O programa vai:
1. Ler cada arquivo de `entrada/`;
2. Classificar o setor;
3. Gravar um relatório em `relatorios/`;
4. Mover o arquivo para `processado/<setor>/`.

### 4) Classificar um texto direto (sem arquivo)

```powershell
python pipeline.py --texto "quero um orçamento para 50 licenças do sistema"
```

### 5) Forçar re-treino antes de rodar

```powershell
python pipeline.py --treinar
```

---

## ⚙️ Configuração

Tudo que costuma precisar de ajuste está no topo de `motor_inferencia.py`:

```python
SETORES = ["comercial", "financeiro", "auditoria", "suporte"]

LIMIAR_CONFIANCA = 0.55   # confianca minima para confiar no modelo
MARGEM_MINIMA = 0.15      # diferenca minima entre 1a e 2a classe
```

- **Baixar `LIMIAR_CONFIANCA`** → o modelo decide mais casos (mais automação,
  menos `manual`). **Subir** → mais textos vão para `manual` (mais conservador).
- **Ajustar `MARGEM_MINIMA`** → controla quão "decisivo" o modelo precisa ser
  quando duas classes ficam próximas.

### Classes e destinos

| Classe (ML) | Ação inferida (regras)     | Pasta de destino          |
|-------------|----------------------------|---------------------------|
| `comercial` | `ENCAMINHAR_COMERCIAL`     | `processado/comercial/`   |
| `financeiro`| `PROCESSAR`                | `processado/financeiro/`  |
| `auditoria` | `ENCAMINHAR_AUDITORIA`     | `processado/auditoria/`   |
| `suporte`   | `ENCAMINHAR_SUPORTE`       | `processado/suporte/`     |
| (inseguro)  | `REVISAR_MANUAL`           | `processado/manual/`      |

---

## 🧩 Camada de regras (`regras/regras.py`)

Depois do ML, um **motor de regras com forward chaining** (`experta`) refina a
decisão com base em **categoria + confiança + evidências extraídas** (valor, data,
CNPJ por regex). Exemplos de regras:

- Financeiro com **valor ≥ R$ 10.000** e confiança ≥ 0,75 → **`AUDITAR`** (prioridade ALTA).
- Financeiro com **valor ≥ R$ 2.000** → **`PROCESSAR`**.
- Financeiro **sem valor extraído** → **`REVISAR_MANUAL`** (possível falha de OCR).
- Auditoria / comercial / suporte com confiança adequada → encaminhar ao setor.

> Compatibilidade: no Python 3.13, o `experta` precisa de um pequeno ajuste
> (`frozendict` usa `collections.Mapping`). Isso **já está corrigido** no topo de
> `regras/regras.py` — não é preciso fazer nada.

---

## 📄 Exemplo de relatório

Cada arquivo gera `relatorios/<nome>.relatorio.txt`:

```
RELATORIO DO MOTOR DE INFERENCIA
Arquivo: financeiro_01
Destino: processado\financeiro

CLASSIFICACAO (ML)
Categoria: financeiro
Confianca: 0.960
Distribuicao de probabilidades:
  financeiro: 0.960
  comercial: 0.020
  auditoria: 0.011
  suporte: 0.010

Origem da decisao: regras

EVIDENCIAS EXTRAIDAS (regex/OCR)
Valor total: 3480.0
Data: 10/02/2026
CNPJ: 12.345.678/0001-90

DECISOES INFERIDAS (MOTOR DE REGRAS)
- acao=PROCESSAR prioridade=MEDIA motivo=Financeiro com valor moderado e confianca adequada
```

---

## ➕ Adicionar novos exemplos de treino

1. Abra `dados/exemplos_treino.json`;
2. Acrescente objetos no array `exemplos`:

```json
{ "texto": "frase de exemplo do setor", "classe": "comercial" }
```

3. Re-treine:

```powershell
python treinar_modelo.py
```

> Quanto mais exemplos reais por setor, melhor a classificação. Mantenha um
> equilíbrio entre as classes (a mesma quantidade por setor ajuda).

---

## 🧪 Exemplo rápido de ponta a ponta

```powershell
# 1. Cria uma entrada
"Solicito auditoria interna nos lançamentos contábeis do trimestre" | Out-File -Encoding utf8 entrada\teste.txt

# 2. Roda
python pipeline.py

# 3. Resultado esperado:
#    PROCESSADO: teste.txt
#    Categoria (ML): auditoria ...
#    Destino: processado\auditoria
```

---

## 🛠 Solução de problemas

| Sintoma | Causa provável | Solução |
|---------|----------------|---------|
| `FileNotFoundError: Modelo não encontrado` | `modelos/` vazio | rode `python treinar_modelo.py` ou `python pipeline.py --treinar` |
| `ModuleNotFoundError: No module named 'sklearn'` | dependência ausente | `pip install scikit-learn joblib numpy experta` |
| `AttributeError: module 'collections' has no attribute 'Mapping'` | `experta`/`frozendict` antigo no Python 3.13 | já corrigido em `regras/regras.py`; garanta que usa essa versão |
| Muitos textos caindo em `manual` | limiar exigente / poucas amostras | reduza `LIMIAR_CONFIANCA` ou adicione mais exemplos |
| `OCR indisponível para .png/.pdf` | Tesseract/Poppler não instalados | instale-os, ou use arquivos `.txt` |
| Acentos aparecendo como `Ã©` no terminal | encoding do console | é só exibição; os arquivos são gravados em UTF-8 |

---

## ⚠️ Limitações

- Classificação é **texto puro** (`.txt` funciona sempre; imagem/PDF dependem de OCR).
- O modelo é **estatístico**: em textos claros a confiança costuma ficar alta
  (≈ 0,9); em textos ambíguos/curtos ela cai (≈ 0,3) e o documento vai para
  `manual` — que é o comportamento seguro esperado.
- A regra de **auditoria por valor elevado** (R$ ≥ 10.000) pode encaminhar um
  documento financeiro para `auditoria`; ajuste/remova a regra em
  `regras/regras.py` se não for o desejado.

---

## 📌 Resumo dos comandos

| Comando | O que faz |
|---------|-----------|
| `python pipeline.py` | Processa tudo em `entrada/` |
| `python pipeline.py --treinar` | Re-treina e processa |
| `python pipeline.py --texto "..."` | Classifica um texto direto |
| `python treinar_modelo.py` | Só treina/avalia o modelo |
| `python motor_inferencia.py` | Mesmo que `pipeline.py` |
