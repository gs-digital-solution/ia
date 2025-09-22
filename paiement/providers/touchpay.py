import uuid
import requests
from requests.auth import HTTPDigestAuth
from .base import BasePaymentProvider
from utils.payments import get_provider_config_for_method

class TouchpayProvider(BasePaymentProvider):
    """
    Provider pour Touchpay (MTN, Orange, Wave…).
    On ne force pas d’indicatif, on ne retire que les caractères non numériques.
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        self.config = get_provider_config_for_method(payment_method)

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        # 0) Nettoyage minimal : on garde seulement les chiffres
        phone_clean = ''.join(c for c in phone if c.isdigit())
        print(">>> TouchpayProvider cleaned phone:", phone_clean)

        # 1) Identifiant client unique
        client_id = str(uuid.uuid4()).replace('-', '')[:16]

        # 2) Construction de l’URL (hard-codée pour l’hôte)
        base = self.config['base_url'].rstrip('/')
        url  = (
            f"{base}/sec/touchpayapi/"
            f"{self.config['agence']}/transaction"
            f"?loginAgent={self.config['login_agent']}"
            f"&passwordAgent={self.config['password_agent']}"
        )

        # 3) Préparation du payload
        payload = {
            "idFromClient": client_id,
            "additionalInfos": {
                "recipientEmail":     user.gmail or "",
                "recipientFirstName": user.first_name or "",
                "recipientLastName":  user.last_name or "",
                "destinataire":       phone_clean
            },
            "amount":          int(amount),
            "callback":        callback_url,
            "recipientNumber": phone_clean,
            "serviceCode":     self.config['service_code'],
        }
        print(">>> TouchpayProvider URL   :", url)
        print(">>> TouchpayProvider PAYLOAD:", payload)

        # 4) En-têtes et authentification digest
        headers = {"Content-Type": "application/json"}
        auth    = HTTPDigestAuth(
            self.config['username'],
            self.config['password']
        )

        # 5) Exécution de la requête
        response = requests.put(
            url,
            json=payload,
            headers=headers,
            auth=auth,
            timeout=30
        )
        print(">>> TouchpayProvider RESPONSE HTTP", response.status_code, response.text)
        return response