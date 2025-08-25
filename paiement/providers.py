from utils.payments import get_touchpay_config_for_method
import requests

class TouchpayProvider:
    def __init__(self, payment_method):  # Instance PaymentMethod
        self.config = get_touchpay_config_for_method(payment_method)
        self.payment_method = payment_method

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        import uuid
        client_id = str(uuid.uuid4()).replace('-', '')[:16]
        url = (
            f"https://apidist.gutouch.net/apidist/sec/touchpayapi/"
            f"{self.config['agence']}/transaction?loginAgent={self.config['login_agent']}&passwordAgent={self.config['password_agent']}"
        )
        data = {
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
        auth = requests.auth.HTTPDigestAuth(self.config['username'], self.config['password'])
        try:
            response = requests.put(url, json=data, headers=headers, auth=auth, timeout=30)
        except Exception as e:
            print("Erreur lors de la requête Touchpay :", e)
            raise
        return response