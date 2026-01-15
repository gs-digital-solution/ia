import uuid
from .base import BasePaymentProvider


class ExterneProvider(BasePaymentProvider):
    """
    Provider pour les paiements externes (Lygos, etc.)
    Pour les liens prédéfinis fournis par le système de paiement
    """

    def __init__(self, payment_method):
        super().__init__(payment_method)
        # Récupérer les infos depuis le modèle PaymentMethod
        self.lien_externe = payment_method.lien_externe
        self.instructions = payment_method.instructions_externes

    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Pour les paiements externes avec lien prédéfini.
        On retourne simplement le lien et les instructions.
        """
        # Générer un ID de transaction unique pour notre tracking
        transaction_id = f"EXT_{str(uuid.uuid4())[:12]}"

        # Retourner les informations nécessaires pour le frontend
        return {
            'type': 'EXTERNE',
            'transaction_id': transaction_id,
            'lien_paiement': self.lien_externe,  # Lien Lygos complet
            'instructions': self.instructions,
            'nom_abonnement': abonnement.nom,
            'montant': float(amount),
            'user_email': user.email or user.gmail or '',
            'user_phone': phone,
            'user_name': user.get_full_name() or user.username,
            'abonnement_id': abonnement.id,
            'payment_method_name': self.payment_method.nom_affiche
        }

    def parse_callback(self, data):
        """
        Pour les paiements externes, aucun callback automatique.
        Le statut est géré manuellement après réception de la preuve.
        """
        return {
            'transaction_id': data.get('transaction_id'),
            'status': 'PENDING_EXTERNAL'  # En attente de confirmation manuelle
        }