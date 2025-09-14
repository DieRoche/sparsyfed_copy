from flwr.server.client_proxy import ClientProxy
from typing import Any


class CidAboveNCriterion:
    """Custom criterion to select clients with cid > 20."""

    def select(self, client: ClientProxy) -> bool:
        # Assuming the client's cid is an integer or can be converted to an integer
        return int(client.cid) < 10
