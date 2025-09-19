import uuid
import requests
from .base import BasePaymentProvider
from utils.payments import get_provider_config_for_method


class CampayProvider(BasePaymentProvider):
    """
    Provider pour CamPay (Collect/Withdraw).
    Méthode d’initiation : POST /api/collect/
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        self.config = get_provider_config_for_method(payment_method)

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        # 1) Générer une référence unique pour l’idempotence
        ext_ref = str(uuid.uuid4())

        # 2) Construire l’URL & en-têtes
        url = f"{self.config['base_url']}/api/collect/"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.config['token']}"
        }

        # 3) Préparer le payload
        payload = {
            "amount": str(int(amount)),  # entier ou chaîne
            "currency": "XAF",
            "from": phone,  # ex: 2376xxxxxxxx
            "description": "Abonnement CIS",
            "external_reference": ext_ref,
            "external_user": str(user.id)
        }

        # 4) Appel HTTP
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        return resp

    def parse_callback(self, data):
        """
        Si vous gérez un callback POST de Campay, vous pouvez extraire ici :
        statut, reference, amount…
        """
        return {
            "transaction_id": data.get("reference"),
            "status": data.get("status").upper()
        }
