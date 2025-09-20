import uuid, requests
from requests.auth import HTTPDigestAuth
from .base import BasePaymentProvider
from utils.payments import get_provider_config_for_method

class TouchpayProvider(BasePaymentProvider):
    def __init__(self, payment_method):
        super().__init__(payment_method)
        self.config = get_provider_config_for_method(payment_method)

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        client_id = str(uuid.uuid4()).replace('-', '')[:16]
        base = self.config['base_url'].rstrip('/')
        url  = (
            f"{base}/sec/touchpayapi/"
            f"{self.config['agence']}/transaction"
            f"?loginAgent={self.config['login_agent']}"
            f"&passwordAgent={self.config['password_agent']}"
        )
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
            "serviceCode": self.config['service_code'],
        }
        headers = {"Content-Type": "application/json"}
        auth    = HTTPDigestAuth(self.config['username'], self.config['password'])
        return requests.put(url, json=payload, headers=headers, auth=auth, timeout=30)