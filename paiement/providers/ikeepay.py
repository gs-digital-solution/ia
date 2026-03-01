import uuid
import requests
import logging
from .base import BasePaymentProvider
from utils.payments import get_provider_config_for_method

logger = logging.getLogger('paiement')


class IkeePayProvider(BasePaymentProvider):
    """
    Provider pour iKeePay en mode API directe
    Utilise une paire de clés (public/secret) pour tous les pays
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        self.config = get_provider_config_for_method(payment_method)
        self.public_key = self.config['public_key']
        self.secret_key = self.config['secret_key']
        self.base_url = self.config['base_url']

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Appelle l'API iKeePay pour initier un paiement
        Documentation: POST /api/v1/payments
        """
        logger.debug(f"→ IkeePayProvider.initiate_payment for {phone}")

        # 1) Nettoyage du numéro (garde uniquement les chiffres)
        phone_clean = ''.join(c for c in phone if c.isdigit())

        # 2) Déterminer la devise selon le pays de la méthode
        # Récupéré depuis extra_config dans l'admin
        currency = self.payment_method.extra_config.get('currency',
                                                        'XOF') if self.payment_method.extra_config else 'XOF'

        # 3) Générer une référence unique
        reference = f"IKEEPAY_{str(uuid.uuid4())[:12]}"

        # 4) Mapping du provider iKeePay selon l'opérateur de la méthode
        # (défini dans l'admin : MTN, ORANGE, MOOV, WAVE, etc.)
        provider_map = {
            'MTN': 'MTN',
            'ORANGE': 'ORANGE',
            'MOOV': 'MOOV',
            'WAVE': 'WAVE',
            'CELPAY': 'CELPAY',
            'MOOV AFRICA': 'MOOV',
            'MTN MONEY': 'MTN',
            'ORANGE MONEY': 'ORANGE',
        }
        provider = provider_map.get(
            self.payment_method.operateur.upper(),
            self.payment_method.operateur.upper()
        )

        # 5) Construction de la requête vers iKeePay
        url = f"{self.base_url}/api/v1/payments"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.secret_key}"  # La clé secrète pour l'authentification
        }

        payload = {
            "amount": int(amount),
            "currency": currency,
            "payment_method": "mobile_money",
            "provider": provider,
            "phone": phone_clean,
            "callback_url": callback_url,  # Notre webhook
            "reference": reference,
            "description": f"Abonnement {abonnement.nom}",
            "customer": {
                "email": user.email or user.gmail or "",
                "name": user.get_full_name() or user.username,
                "phone": phone_clean
            },
            "public_key": self.public_key  # Optionnel, selon leur API
        }

        logger.debug(f"IkeePay URL: {url}")
        logger.debug(f"IkeePay payload: {payload}")

        try:
            # 6) Appel API à iKeePay
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30
            )

            logger.debug(f"IkeePay response status: {response.status_code}")
            logger.debug(f"IkeePay response body: {response.text}")

            # 7) Traitement de la réponse
            if response.status_code in [200, 201, 202]:
                try:
                    data = response.json()
                except:
                    data = {"raw_text": response.text}

                # iKeePay peut retourner différentes structures
                transaction_id = data.get('reference') or data.get('id') or data.get('transaction_id') or reference

                # Retourne le résultat pour la transaction
                return {
                    'transaction_id': transaction_id,
                    'status': 'PROCESSING',  # En attente du webhook
                    'raw_response': data,
                    'reference': reference
                }
            else:
                # Erreur API
                return {
                    'transaction_id': reference,
                    'status': 'FAIL',
                    'raw_response': {
                        'error': response.text,
                        'status_code': response.status_code
                    }
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"IkeePay request error: {str(e)}")
            return {
                'transaction_id': reference,
                'status': 'FAIL',
                'raw_response': {'error': str(e)}
            }
        except Exception as e:
            logger.error(f"IkeePay unexpected error: {str(e)}")
            return {
                'transaction_id': reference,
                'status': 'FAIL',
                'raw_response': {'error': str(e)}
            }

    def parse_callback(self, data):
        """
        Parse le webhook reçu d'iKeePay
        À adapter selon la structure exacte que iKeePay envoie
        """
        logger.debug(f"Parsing IkeePay callback: {data}")

        # Essayer différentes structures possibles
        status = data.get('status', '').upper()
        event = data.get('event', '').upper()
        state = data.get('state', '').upper()

        # Déterminer le statut final
        if status in ['SUCCESSFUL', 'SUCCESS', 'COMPLETED', 'PAID'] or \
                event in ['PAYMENT_SUCCESSFUL', 'SUCCESS'] or \
                state in ['SUCCESSFUL', 'COMPLETED']:
            final_status = 'SUCCESS'
        elif status in ['FAILED', 'FAIL', 'ERROR', 'CANCELLED'] or \
                event in ['PAYMENT_FAILED', 'FAILED'] or \
                state in ['FAILED', 'CANCELLED']:
            final_status = 'FAIL'
        else:
            final_status = 'PROCESSING'

        # Récupérer l'ID de transaction
        transaction_id = data.get('reference') or data.get('id') or data.get('transaction_id') or data.get('order_id')

        return {
            'transaction_id': transaction_id,
            'status': final_status,
            'raw_response': data
        }
