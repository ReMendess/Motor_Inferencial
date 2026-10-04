"""Motor de inferência de setores (comercial, financeiro, auditoria, suporte).

Lê textos da pasta entrada/, usa um modelo treinado salvo em modelos/
(TF-IDF + Regressão Logística) para classificar o setor e aplica a regra
"o que não souber vai para manual" quando o modelo fica inseguro.

Fluxo:
    entrada/*.txt -> normalização -> TF-IDF -> LogisticRegression
    -> [regra do manual] -> base de regras (experta) -> relatório -> processado/<setor>/
"""
from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import joblib

from regras.regras import Decisao, Documento, MotorInferencial

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
PASTA_ENTRADA = "entrada"
PASTA_PROCESSADO = "processado"
PASTA_RELATORIOS = "relatorios"
PASTA_MODELOS = "modelos"
PASTA_DADOS = "dados"

MODELO_PATH = os.path.join(PASTA_MODELOS, "modelo.joblib")
VEC_PATH = os.path.join(PASTA_MODELOS, "vectorizer.joblib")
EXEMPLOS_PATH = os.path.join(PASTA_DADOS, "exemplos_treino.json")

SETORES = ["comercial", "financeiro", "auditoria", "suporte"]

LIMIAR_CONFIANCA = 0.55
MARGEM_MINIMA = 0.15

DESTINOS: Dict[str, str] = {s: os.path.join(PASTA_PROCESSADO, s) for s in SETORES + ["manual"]}

ACAO_PARA_SETOR = {
    "AUDITAR": "auditoria",
    "ENCAMINHAR_AUDITORIA": "auditoria",
    "ENCAMINHAR_COMERCIAL": "comercial",
    "PROCESSAR": "financeiro",
    "ENCAMINHAR_SUPORTE": "suporte",
    "REVISAR_MANUAL": "manual",
}

PRIORIDADE_RANK = {"ALTA": 3, "MEDIA": 2, "BAIXA": 1}


@dataclass
class ResultadoML:
    categoria: str
    confianca: float
    probs: List[Tuple[str, float]]


@dataclass
class Extracao:
    valor_total: Optional[float] = None
    data: Optional[str] = None
    cnpj: Optional[str] = None


# --------------------------------------------------------------------------- #
# Utilidades de texto
# --------------------------------------------------------------------------- #
def normalizar_texto(txt: str) -> str:
    txt = txt.replace("\ufeff", "").replace("\n", " ")
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    txt = txt.lower()
    return re.sub(r"\s+", " ", txt).strip()


def extrair_campos(texto: str) -> Extracao:
    """Extrai evidências estruturadas (valor, data, CNPJ) via regex."""
    texto = texto.replace("\n", " ")
    m_valor = re.search(
        r"r\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2}))",
        texto,
        flags=re.IGNORECASE,
    )
    valor = None
    if m_valor:
        bruto = m_valor.group(1).replace(".", "").replace(",", ".")
        try:
            valor = float(bruto)
        except ValueError:
            valor = None

    m_data = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", texto)
    m_cnpj = re.search(r"\b(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})\b", texto)

    return Extracao(
        valor_total=valor,
        data=m_data.group(1) if m_data else None,
        cnpj=m_cnpj.group(1) if m_cnpj else None,
    )


def carregar_exemplos(caminho: str = EXEMPLOS_PATH):
    """Lê o arquivo JSON de exemplos e devolve (textos, rótulos)."""
    with open(caminho, "r", encoding="utf-8") as f:
        payload = json.load(f)
    dados = [e["texto"] for e in payload["exemplos"]]
    labels = [e["classe"] for e in payload["exemplos"]]
    return dados, labels

# --------------------------------------------------------------------------- #
# Treino e carregamento do modelo
# --------------------------------------------------------------------------- #
def treinar_modelo(caminho_exemplos: str = EXEMPLOS_PATH, salvar: bool = True):
    """Treina TF-IDF + Regressão Logística e (opcionalmente) salva em disco."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    dados, labels = carregar_exemplos(caminho_exemplos)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform([normalizar_texto(t) for t in dados])
    clf = LogisticRegression(max_iter=2000, C=10, class_weight="balanced")
    clf.fit(X, labels)

    if salvar:
        os.makedirs(PASTA_MODELOS, exist_ok=True)
        joblib.dump(clf, MODELO_PATH)
        joblib.dump(vec, VEC_PATH)
    return clf, vec, dados, labels


@lru_cache(maxsize=1)
def carregar_modelo():
    """Carrega (e memoiza) o modelo salvo em modelos/."""
    if not (os.path.exists(MODELO_PATH) and os.path.exists(VEC_PATH)):
        raise FileNotFoundError(
            "Modelo não encontrado em modelos/. Treine com python treinar_modelo.py."
        )
    return joblib.load(MODELO_PATH), joblib.load(VEC_PATH)


def classificar_texto(texto: str, clf=None, vec=None) -> ResultadoML:
    """Classifica o texto e devolve categoria vencedora + distribuição de probabilidades."""
    if clf is None or vec is None:
        clf, vec = carregar_modelo()
    X = vec.transform([normalizar_texto(texto)])
    probs = clf.predict_proba(X)[0]
    pares = sorted(
        [(c, float(p)) for c, p in zip(clf.classes_, probs)],
        key=lambda x: x[1],
        reverse=True,
    )
    return ResultadoML(categoria=pares[0][0], confianca=pares[0][1], probs=pares)


# --------------------------------------------------------------------------- #
# Base de regras (experta) -> decisões e destino
# --------------------------------------------------------------------------- #
def inferir_decisao(categoria: str, confianca: float, extracao: Extracao) -> List[dict]:
    """Roda a base de regras (forward chaining) sobre o fato Documento."""
    motor = MotorInferencial()
    motor.reset()
    motor.declare(Documento(
        categoria=categoria,
        confianca=confianca,
        valor_total=extracao.valor_total,
        data=extracao.data,
        cnpj=extracao.cnpj,
    ))
    motor.run()
    decisoes = [
        {"acao": f["acao"], "prioridade": f["prioridade"], "motivo": f["motivo"]}
        for f in motor.facts.values()
        if isinstance(f, Decisao)
    ]
    decisoes.sort(key=lambda d: PRIORIDADE_RANK.get(d["prioridade"], 0), reverse=True)
    return decisoes


def selecionar_destino(decisoes: List[dict]) -> str:
    """Mapeia a decisão de maior prioridade para a pasta de destino."""
    if not decisoes:
        return DESTINOS["manual"]
    return DESTINOS.get(ACAO_PARA_SETOR.get(decisoes[0]["acao"], "manual"), DESTINOS["manual"])


def decidir_setor(resultado: ResultadoML, extracao: Extracao) -> dict:
    """Aplica a regra do manual e, se o modelo estiver seguro, a base de regras."""
    p1 = resultado.probs[0][1]
    p2 = resultado.probs[1][1] if len(resultado.probs) > 1 else 0.0
    margem = p1 - p2

    if p1 < LIMIAR_CONFIANCA or margem < MARGEM_MINIMA:
        motivo = (
            f"Modelo inseguro (confianca={p1:.2f}, margem={margem:.2f}); "
            f"limiares confianca>={LIMIAR_CONFIANCA} e margem>={MARGEM_MINIMA}"
        )
        return {
            "origem": "regra_manual",
            "decisoes": [{"acao": "REVISAR_MANUAL", "prioridade": "ALTA", "motivo": motivo}],
            "destino": DESTINOS["manual"],
        }

    decisoes = inferir_decisao(resultado.categoria, resultado.confianca, extracao)
    return {
        "origem": "regras",
        "decisoes": decisoes,
        "destino": selecionar_destino(decisoes),
    }

# --------------------------------------------------------------------------- #
# Leitura de arquivos (txt e, se disponível, OCR)
# --------------------------------------------------------------------------- #
def ler_texto_de_arquivo(caminho: str) -> str:
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".txt":
        with open(caminho, "r", encoding="utf-8-sig", errors="replace") as f:
            return f.read()

    if ext in (".png", ".jpg", ".jpeg"):
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(caminho), lang="por")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"OCR indisponivel para {ext}: {e}")

    if ext == ".pdf":
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Leitura de PDF indisponivel: {e}")
        return "\n".join(
            pytesseract.image_to_string(pg, lang="por") for pg in convert_from_path(caminho)
        )

    raise ValueError(f"Tipo de arquivo nao suportado: {ext}")


# --------------------------------------------------------------------------- #
# Relatorio e processamento
# --------------------------------------------------------------------------- #
def gerar_relatorio(base: str, texto: str, resultado: ResultadoML,
                    extracao: Extracao, info: dict) -> str:
    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    caminho = os.path.join(PASTA_RELATORIOS, f"{base}.relatorio.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("RELATORIO DO MOTOR DE INFERENCIA\n")
        f.write(f"Arquivo: {base}\n")
        f.write(f"Destino: {info['destino']}\n\n")

        f.write("CLASSIFICACAO (ML)\n")
        f.write(f"Categoria: {resultado.categoria}\n")
        f.write(f"Confianca: {resultado.confianca:.3f}\n")
        f.write("Distribuicao de probabilidades:\n")
        for c, p in resultado.probs:
            f.write(f"  {c}: {p:.3f}\n")
        f.write(f"\nOrigem da decisao: {info['origem']}\n\n")

        f.write("EVIDENCIAS EXTRAIDAS (regex/OCR)\n")
        f.write(f"Valor total: {extracao.valor_total}\n")
        f.write(f"Data: {extracao.data}\n")
        f.write(f"CNPJ: {extracao.cnpj}\n\n")

        f.write("DECISOES INFERIDAS (MOTOR DE REGRAS)\n")
        for d in info["decisoes"]:
            f.write(f"- acao={d['acao']} prioridade={d['prioridade']} motivo={d['motivo']}\n")
        f.write("\n")

        f.write("TRECHO DO TEXTO PROCESSADO (AMOSTRA)\n")
        f.write(normalizar_texto(texto)[:600] + "\n")
    return caminho


def _imprimir_resumo(nome: str, resultado: ResultadoML, extracao: Extracao, info: dict) -> None:
    print(f"\nPROCESSADO: {nome}")
    print(f"  Categoria (ML): {resultado.categoria}  confianca: {resultado.confianca:.3f}")
    print(f"  Origem da decisao: {info['origem']}")
    print(f"  Valor: {extracao.valor_total}  Data: {extracao.data}  CNPJ: {extracao.cnpj}")
    for d in info["decisoes"]:
        print(f"  -> acao={d['acao']} prioridade={d['prioridade']} ({d['motivo']})")
    print(f"  Destino: {info['destino']}")


def processar_arquivo(caminho: str, mover: bool = True) -> dict:
    nome = os.path.basename(caminho)
    base = os.path.splitext(nome)[0]

    texto = ler_texto_de_arquivo(caminho)
    resultado = classificar_texto(texto)
    extracao = extrair_campos(texto)
    info = decidir_setor(resultado, extracao)

    os.makedirs(info["destino"], exist_ok=True)
    gerar_relatorio(base, texto, resultado, extracao, info)

    if mover:
        shutil.move(caminho, os.path.join(info["destino"], nome))

    _imprimir_resumo(nome, resultado, extracao, info)
    return {"arquivo": nome, "resultado": resultado, "extracao": extracao, **info}


def classificar_texto_direto(texto: str) -> dict:
    """Classifica um texto informado diretamente (sem arquivo fisico)."""
    preparar_pastas()
    resultado = classificar_texto(texto)
    extracao = extrair_campos(texto)
    info = decidir_setor(resultado, extracao)
    _imprimir_resumo("(texto direto)", resultado, extracao, info)
    return {"resultado": resultado, "extracao": extracao, **info}


def preparar_pastas() -> None:
    for p in [PASTA_ENTRADA, PASTA_PROCESSADO, PASTA_RELATORIOS, PASTA_MODELOS, PASTA_DADOS]:
        os.makedirs(p, exist_ok=True)
    for p in DESTINOS.values():
        os.makedirs(p, exist_ok=True)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Motor de inferencia: classifica textos por setor "
            "(comercial, financeiro, auditoria, suporte) e envia o resto para manual."
        )
    )
    parser.add_argument("--texto", help="Classifica um texto informado direto (sem arquivo).")
    parser.add_argument("--treinar", action="store_true",
                        help="Forca o treino do modelo antes de rodar.")
    args = parser.parse_args(argv)

    preparar_pastas()

    if args.treinar or not (os.path.exists(MODELO_PATH) and os.path.exists(VEC_PATH)):
        treinar_modelo()
        carregar_modelo.cache_clear()
        print("Modelo treinado e salvo em modelos/.")

    if args.texto:
        classificar_texto_direto(args.texto)
        return

    arquivos = [
        a for a in (os.path.join(PASTA_ENTRADA, n) for n in os.listdir(PASTA_ENTRADA))
        if os.path.isfile(a)
    ]
    if not arquivos:
        print("Nenhum arquivo em entrada/. Adicione arquivos .txt e execute novamente.")
        return

    for caminho in sorted(arquivos):
        try:
            processar_arquivo(caminho)
        except Exception as e:  # noqa: BLE001
            print(f"\nERRO ao processar {os.path.basename(caminho)}: {e}")
            os.makedirs(DESTINOS["manual"], exist_ok=True)
            shutil.move(caminho, os.path.join(DESTINOS["manual"], os.path.basename(caminho)))


if __name__ == "__main__":
    main()
