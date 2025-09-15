import uuid
import requests
from requests.auth import HTTPDigestAuth
from .base import BasePaymentProvider
from utils.payments import get_touchpay_config_for_method

class TouchpayProvider(BasePaymentProvider):
    """
    Provider pour l'agrégateur Touchpay (MTN, Orange, Wave…).
    Récupère la config sensible via utils.get_touchpay_config_for_method
    et utilise le champ 'service_code' de PaymentMethod si défini.
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        self.config = get_touchpay_config_for_method(payment_method)

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Lance la requête PUT /transaction vers Touchpay.
        :param amount: montant (int ou Decimal)
        :param phone: numéro du destinataire
        :param abonnement: instance SubscriptionType
        :param user: instance CustomUser
        :param callback_url: URL de callback complet
        :return: requests.Response
        """
        # 1) Identifiant client unique
        client_id = str(uuid.uuid4()).replace('-', '')[:16]

        # 2) Construction URL
        url = (
            f"https://api.gutouch.net/sec/touchpayapi/"
            f"{self.config['agence']}/transaction"
            f"?loginAgent={self.config['login_agent']}"
            f"&passwordAgent={self.config['password_agent']}"
        )

        # 3) ServiceCode : d’abord depuis le modèle, sinon fallback .env
        service_code = (
            getattr(self.payment_method, 'service_code', None)
            or self.config.get('service_code')
        )

        # 4) Payload JSON
        payload = {
            "idFromClient": client_id,
            "additionnalInfos": {
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

        headers = {"Content-Type": "application/json"}
        auth = HTTPDigestAuth(
            self.config['username'],
            self.config['password']
        )

        # 5) Exécution de la requête
        try:
            response = requests.put(
                url,
                json=payload,
                headers=headers,
                auth=auth,
                timeout=30
            )
        except Exception as e:
            # En cas d’erreur réseau / auth
            print("Erreur TouchpayProvider:", e)
            raise

        return response