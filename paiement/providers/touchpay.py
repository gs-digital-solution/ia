import uuid
import requests
from requests.auth import HTTPDigestAuth
from .base import BasePaymentProvider
from utils.payments import get_provider_config_for_method  # <-- on importe la nouvelle fonction

class TouchpayProvider(BasePaymentProvider):
    """
    Provider pour l'agrégateur Touchpay (MTN, Orange, Wave…).
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        # On récupère toute la config depuis .env via utils/payments.py
        self.config = get_provider_config_for_method(payment_method)

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Lance la requête PUT /transaction vers Touchpay.
        """

        # 1) Identifiant client unique
        client_id = str(uuid.uuid4()).replace('-', '')[:16]

        # 2) Construction de l'URL (hard-codée pour l'hôte, dynamic pour l'agence & agents)
        url = (
            "https://apidist.gutouch.net/apidist/sec/touchpayapi/"
            f"{self.config['agence']}/transaction"
            f"?loginAgent={self.config['login_agent']}"
            f"&passwordAgent={self.config['password_agent']}"
        )

        # 3) Choix du serviceCode depuis le modèle ou la config
        service_code = (
            getattr(self.payment_method, 'service_code', None)
            or self.config.get('service_code')
        )

        # 4) Préparation du payload JSON (attention à additionalInfos)
        payload = {
            "idFromClient": client_id,
            "additionalInfos": {
                "recipientEmail": user.gmail or "",
                "recipientFirstName": user.first_name or "",
                "recipientLastName": user.last_name or "",
                "destinataire": phone
            },
            "amount": int(amount),
            "callback": callback_url,
            "recipientNumber": phone,
            "serviceCode": service_code,
        }

        # 5) Headers et authentification digest
        headers = {"Content-Type": "application/json"}
        auth = HTTPDigestAuth(
            self.config['username'],
            self.config['password']
        )

        # 6) Exécution de la requête
        try:
            response = requests.put(
                url,
                json=payload,
                headers=headers,
                auth=auth,
                timeout=30
            )
        except Exception as e:
            # En cas d’erreur réseau/authentification
            print("Erreur TouchpayProvider:", e)
            raise

        return response