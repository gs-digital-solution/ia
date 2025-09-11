import requests
from .base import BasePaymentProvider
from utils.payments import get_touchpay_config_for_method

class TouchpayProvider(BasePaymentProvider):
    """
    Provider pour l'agrégateur Touchpay (MTN, Orange, Wave…).
    La config sensible est récupérée via utils.get_touchpay_config_for_method.
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        self.config = get_touchpay_config_for_method(payment_method)

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Envoie la requête PUT /transaction au endpoint Touchpay.
        Retourne l'objet requests.Response.
        """
        # Génération d'un identifiant client unique
        import uuid
        client_id = str(uuid.uuid4()).replace('-', '')[:16]

        # Construction de l'URL
        url = (
            f"https://apidist.gutouch.net/apidist/sec/touchpayapi/"
            f"{self.config['agence']}/transaction"
            f"?loginAgent={self.config['login_agent']}"
            f"&passwordAgent={self.config['password_agent']}"
        )
        # Corps JSON
        payload = {
            "idFromClient": client_id,
            "additionnalInfos": {
                "recipientEmail": user.gmail or "",
                "recipientFirstName": getattr(user, 'first_name', ''),
                "recipientLastName": getattr(user, 'last_name', ''),
                "destinataire": phone
            },
            "amount": int(amount),
            "callback": callback_url,
            "recipientNumber": phone,
            "serviceCode": self.config["service_code"],
        }
        headers = {"Content-Type": "application/json"}
        auth = requests.auth.HTTPDigestAuth(
            self.config['username'], self.config['password']
        )

        try:
            response = requests.put(
                url, json=payload, headers=headers, auth=auth, timeout=30
            )
        except Exception as e:
            # Log d'erreur et on remonte
            print("Erreur lors de la requête Touchpay :", e)
            raise

        return response
