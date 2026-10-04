import collections
import collections.abc

# Compatibilidade: a dependência frozendict (usada pelo experta) referencia
# collections.Mapping, removido no Python 3.10+. Reexpomos os ABCs necessários.
for _nome in ("Mapping", "MutableMapping", "Sequence", "MutableSequence",
              "Iterable", "Iterator", "Set", "MutableSet", "Callable"):
    if not hasattr(collections, _nome):
        setattr(collections, _nome, getattr(collections.abc, _nome))

from experta import KnowledgeEngine, Fact, Rule, P, MATCH

class Documento(Fact):
    """Fato principal com informações do documento processado."""
    pass

class Decisao(Fact):
    """Fatos inferidos pelo motor, usados para guiar ações."""
    pass


class MotorInferencial(KnowledgeEngine):
    """
    Motor inferencial com forward chaining.
    A base de regras usa categoria, confiança e evidências extraídas (OCR/regex)
    para inferir ações e níveis de prioridade.
    """

    @Rule(Documento(categoria="financeiro",
                    confianca=P(lambda x: x >= 0.75),
                    valor_total=P(lambda v: v is not None and v >= 10000)))
    def financeiro_alto_valor(self):
        self.declare(Decisao(acao="AUDITAR", prioridade="ALTA",
                             motivo="Financeiro com valor elevado e alta confiança"))

    @Rule(Documento(categoria="financeiro",
                    confianca=P(lambda x: x >= 0.55),
                    valor_total=P(lambda v: v is not None and v >= 2000)))
    def financeiro_medio_valor(self):
        self.declare(Decisao(acao="PROCESSAR", prioridade="MEDIA",
                             motivo="Financeiro com valor moderado e confiança adequada"))

    @Rule(Documento(categoria="suporte", confianca=P(lambda x: x >= 0.55)))
    def suporte_padrao(self):
        self.declare(Decisao(acao="ENCAMINHAR_SUPORTE", prioridade="MEDIA",
                             motivo="Suporte identificado com confiança adequada"))

    @Rule(Documento(categoria="comercial", confianca=P(lambda x: x >= 0.55)))
    def comercial_padrao(self):
        self.declare(Decisao(acao="ENCAMINHAR_COMERCIAL", prioridade="MEDIA",
                             motivo="Comercial identificado com confiança adequada"))

    @Rule(Documento(categoria="auditoria", confianca=P(lambda x: x >= 0.55)))
    def auditoria_padrao(self):
        self.declare(Decisao(acao="ENCAMINHAR_AUDITORIA", prioridade="ALTA",
                             motivo="Auditoria identificada com confiança adequada"))

    @Rule(Documento(categoria="financeiro", confianca=P(lambda x: x >= 0.55)))
    def financeiro_padrao(self):
        self.declare(Decisao(acao="PROCESSAR", prioridade="MEDIA",
                             motivo="Financeiro identificado com confiança adequada"))

    @Rule(Documento(confianca=P(lambda x: x < 0.55)))
    def baixa_confianca(self):
        self.declare(Decisao(acao="REVISAR_MANUAL", prioridade="ALTA",
                             motivo="Baixa confiança do modelo, revisão necessária"))

    @Rule(Documento(categoria="financeiro",
                    valor_total=P(lambda v: v is None)),
          salience=10)
    def financeiro_sem_valor(self):
        self.declare(Decisao(acao="REVISAR_MANUAL", prioridade="ALTA",
                             motivo="Financeiro sem valor extraído, possível falha de OCR"))

    @Rule(Documento(categoria=MATCH.cat,
                    confianca=MATCH.conf),
          Decisao(acao=MATCH.acao, prioridade=MATCH.prio, motivo=MATCH.motivo))
    def consolidar(self, cat, conf, acao, prio, motivo):
        pass