"""Pipeline de processamento automatizado (ponto de entrada do projeto).

O projeto evoluiu para o motor de inferência em ``motor_inferencia.py``.
Este módulo mantém o ponto de entrada original e delega o trabalho a ele:
classifica os textos de ``entrada/`` por setor (comercial, financeiro,
auditoria, suporte) usando o modelo salvo em ``modelos/`` e envia o que o
modelo não souber para ``processado/manual/``.

Uso:
    python pipeline.py                  # processa tudo em entrada/
    python pipeline.py --treinar        # re-treina o modelo e processa
    python pipeline.py --texto "texto"  # classifica um texto direto
"""
from __future__ import annotations

from motor_inferencia import main


if __name__ == "__main__":
    main()