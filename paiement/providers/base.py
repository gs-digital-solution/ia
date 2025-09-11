from abc import ABC, abstractmethod

class BasePaymentProvider(ABC):
    """
    Classe abstraite de tous les providers de paiement.
    Chaque provider (Touchpay, Campay…) doit hériter de celle‐ci
    et implémenter initiate_payment().
    """

    def __init__(self, payment_method):
        """
        :param payment_method: instance de modèle PaymentMethod
        """
        self.payment_method = payment_method

    @abstractmethod
    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Lance la transaction auprès de l’API du prestataire.
        Doit renvoyer la réponse brute du provider (requests.Response, dict…).
        """
        pass

    def parse_callback(self, data):
        """
        (Optionnel) Normalise la charge utile reçue en callback.
        Doit renvoyer un dict avec au moins 'transaction_id' et 'status'.
        """
        return {
            'transaction_id': data.get('transaction_id'),
            'status': data.get('status', '').upper()
        }
