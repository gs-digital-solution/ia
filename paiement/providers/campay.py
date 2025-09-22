import uuid
import requests
from .base import BasePaymentProvider
from utils.payments import get_provider_config_for_method

class CampayProvider(BasePaymentProvider):
    """
    Provider pour CamPay (Collect/Withdraw).
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        self.config = get_provider_config_for_method(payment_method)

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        # 0) Normalisation spécifique Campay (forçage de '237')
        num = ''.join(c for c in phone if c.isdigit())
        if num.startswith('0'):
            num = '237' + num[1:]
        elif not num.startswith('237'):
            num = '237' + num
        phone = num
        print(">>> CampayProvider phone used:", phone)

        ext_ref = str(uuid.uuid4())
        base    = self.config['base_url'].rstrip('/')
        url     = f"{base}/api/collect/"
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Token {self.config['perm_token']}"
        }
        payload = {
            "amount":             str(int(amount)),
            "currency":           "XAF",
            "from":               phone,
            "description":        "Abonnement CIS",
            "external_reference": ext_ref,
            "external_user":      str(user.id),
            "callback":           callback_url,
        }
        print(">>> CampayProvider collect URL   :", url)
        print(">>> CampayProvider collect PAYLOD:", payload)

        return requests.post(url, json=payload, headers=headers, timeout=30)

    def parse_callback(self, data):
        return {
            "transaction_id": data.get("reference"),
            "status":         data.get("status").upper()
        }