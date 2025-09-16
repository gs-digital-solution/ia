import uuid
import logging
import requests
from requests.auth import HTTPDigestAuth

from .base import BasePaymentProvider
from utils.payments import get_touchpay_config_for_method

logger = logging.getLogger('paiement')

class TouchpayProvider(BasePaymentProvider):
    """
    Provider pour l'agrégateur Touchpay (MTN, Orange, etc.).
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        self.config = get_touchpay_config_for_method(payment_method)

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        # 1) Identifiant client unique
        client_id = str(uuid.uuid4()).replace('-', '')[:16]

        # 2) Récupère partner_id (obligatoire dans la config .env)
        partner_id = self.config.get('partner_id')
        if not partner_id:
            raise ValueError("Partner ID manquant dans la config Touchpay")

        # 3) Construction de l’URL et des params en GET
        url = f"https://api.gutouch.net/sec/touchpayapi/{partner_id}/transaction"
        params = {
            "loginAgent":    self.config['login_agent'],
            "passwordAgent": self.config['password_agent'],
        }

        # Pour debug, vous verrez l’URL et les params dans les logs
        logger.debug(f"TouchpayProvider → URL: {url} params: {params}")

        # 4) Payload JSON
        service_code = self.payment_method.service_code or self.config.get('service_code')
        payload = {
            "idFromClient":      client_id,
            "additionnalInfos": {
                "recipientEmail":      user.gmail or "",
                "recipientFirstName":  user.first_name or "",
                "recipientLastName":   user.last_name or "",
                "destinataire":        phone
            },
            "amount":            int(amount),
            "callback":          callback_url,
            "recipientNumber":   phone,
            "serviceCode":       service_code,
        }

        headers = {"Content-Type": "application/json"}
        auth    = HTTPDigestAuth(self.config['username'], self.config['password'])

        # 5) Exécution de la requête PUT
        response = requests.put(
            url,
            params=params,
            json=payload,
            headers=headers,
            auth=auth,
            timeout=30
        )

        # Log du retour brut
        logger.debug(f"TouchpayProvider ← status {response.status_code}, body: {response.text}")

        return response