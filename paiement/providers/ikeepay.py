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
        # Récupérer les clés depuis extra_config
        extra = payment_method.extra_config or {}
        self.public_key = extra.get('public_key')
        self.secret_key = extra.get('secret_key')
        self.checkout_url = "https://www.ikeepay.com/checkout/v1/inline"

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Pour iKeePay, on retourne l'URL de l'iframe à charger dans Flutter
        Convertit le montant FCFA → USD selon le taux configuré par pays
        """
        logger.debug(f"→ IkeePayProvider.initiate_payment for abonnement {abonnement.id}")

        # 1) Générer un order_id unique
        order_id = f"IKEEPAY_{str(uuid.uuid4())[:12]}"

        # 2) Récupérer le taux de conversion depuis extra_config (configurable par pays)
        extra = self.payment_method.extra_config or {}
        taux_conversion = extra.get('taux_conversion', 600)  # 600 par défaut (1 USD = 600 FCFA)

        # 3) Convertir FCFA → USD
        amount_in_fcfa = int(amount)
        amount_in_usd = amount_in_fcfa / taux_conversion
        amount_in_usd = round(amount_in_usd, 2)  # Arrondir à 2 décimales

        logger.debug(f"Conversion: {amount_in_fcfa} FCFA → {amount_in_usd} USD (taux: {taux_conversion})")

        # 4) Construire les paramètres pour l'iframe (AVEC montant en USD)
        params = {
            'pk': self.public_key,
            'sk': self.secret_key,
            'amount': str(amount_in_usd),  # ← Maintenant en USD
            'currency': 'USD',
            'order_id': order_id,
            'email': user.email or user.gmail or '',
            'customer_name': user.get_full_name() or user.username,
            'customer_phone': phone,
            'description': f"Abonnement {abonnement.nom}",
        }

        # 5) Nettoyer les paramètres vides
        params = {k: v for k, v in params.items() if v}

        # 6) Construire l'URL complète
        iframe_url = f"{self.checkout_url}?{urlencode(params)}"

        logger.debug(f"IkeePay iframe URL generated: {iframe_url[:100]}...")

        # 7) Retourner les informations pour la transaction
        return {
            'type': 'IFRAME',
            'iframe_url': iframe_url,
            'transaction_id': order_id,
            'status': 'PENDING_IFRAME',
            'amount': amount_in_fcfa,  # On garde le montant FCFA pour l'affichage
            'amount_usd': amount_in_usd,  # Optionnel: montant en USD pour debug
            'taux_conversion': taux_conversion,
            'nom_abonnement': abonnement.nom,
            'payment_method_name': self.payment_method.nom_affiche,
            'raw_response': {
                'iframe_url': iframe_url,
                'order_id': order_id,
                'amount_usd': amount_in_usd,
                'taux_conversion': taux_conversion
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