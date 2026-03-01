import uuid
import logging
from urllib.parse import urlencode
from .base import BasePaymentProvider

logger = logging.getLogger('paiement')


class IkeePayProvider(BasePaymentProvider):
    """
    Provider pour iKeePay - Mode IFRAME
    Documentation: https://www.ikeepay.com/checkout/v1/inline
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        # Récupérer les clés depuis les variables d'env
        self.public_key = payment_method.extra_config.get('public_key') if payment_method.extra_config else None
        self.secret_key = payment_method.extra_config.get('secret_key') if payment_method.extra_config else None
        self.checkout_url = "https://www.ikeepay.com/checkout/v1/inline"

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Pour iKeePay, on retourne l'URL de l'iframe à charger dans Flutter
        Note: callback_url n'est pas utilisé par iKeePay (pas de webhook)
        """
        logger.debug(f"→ IkeePayProvider.initiate_payment for abonnement {abonnement.id}")

        # 1) Générer un order_id unique
        order_id = f"IKEEPAY_{str(uuid.uuid4())[:12]}"

        # 2) iKeePay travaille en USD mais convertit automatiquement
        # On envoie le montant en FCFA, iKeePay fera la conversion
        amount_in_fcfa = int(amount)

        # 3) Construire les paramètres pour l'iframe
        params = {
            'pk': self.public_key,
            'sk': self.secret_key,
            'amount': str(amount_in_fcfa),
            'currency': 'USD',  # iKeePay utilise USD comme devise de base
            'order_id': order_id,
            'email': user.email or user.gmail or '',
            'customer_name': user.get_full_name() or user.username,
            'customer_phone': phone,
            'description': f"Abonnement {abonnement.nom}",
        }

        # 4) Nettoyer les paramètres vides
        params = {k: v for k, v in params.items() if v}

        # 5) Construire l'URL complète
        iframe_url = f"{self.checkout_url}?{urlencode(params)}"

        logger.debug(f"IkeePay iframe URL generated: {iframe_url[:100]}...")

        # 6) Retourner les informations pour la transaction
        return {
            'type': 'IFRAME',
            'iframe_url': iframe_url,
            'transaction_id': order_id,
            'status': 'PENDING_IFRAME',  # Nouveau statut
            'amount': amount_in_fcfa,
            'nom_abonnement': abonnement.nom,
            'payment_method_name': self.payment_method.nom_affiche,
            'raw_response': {
                'iframe_url': iframe_url,
                'order_id': order_id
            }
        }

    def parse_callback(self, data):
        """
        iKeePay n'envoie pas de webhook.
        Cette méthode ne sera pas utilisée, mais on la garde pour l'interface.
        """
        return {
            'transaction_id': data.get('order_id'),
            'status': 'PENDING_IFRAME',
            'raw_response': data
        }