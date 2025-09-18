from abc import ABC, abstractmethod

class BasePaymentProvider(ABC):
    def __init__(self, payment_method):
        self.payment_method = payment_method

    @abstractmethod
    def initiate_payment(self, amount, phone, abonnement, user, callback_url):
        """
        Déclenche le cash-in (débiter le portefeuille utilisateur et verser sur votre compte).
        """
        pass

    def parse_callback(self, data):
        return {
            'transaction_id': data.get('idFromClient') or data.get('transaction_id'),
            'status': (data.get('status') or '').upper()
        }
