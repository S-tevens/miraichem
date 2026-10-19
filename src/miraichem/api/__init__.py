"""Web API behind the MiraiChem app (``miraichem serve``)."""

from miraichem.api.app import create_app

__all__ = ["create_app"]
