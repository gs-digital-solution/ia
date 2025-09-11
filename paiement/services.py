from .models import PaymentTransaction
from .providers.factory import get_provider_for_method
import uuid
import django.utils.timezone as timezone

def process_payment(user, abonnement, phone, payment_method, callback_url):
    """
    Orchestrateur de paiement :
     - Crée l’enregistrement PaymentTransaction (INITIATED)
     - Délègue l’appel au provider (ex : TouchpayProvider)
     - Met à jour la transaction d’après la réponse
    """
    # 1) Création initiale
    unique_id = "init_" + str(uuid.uuid4()).replace("-", "")[:12]
    tx = PaymentTransaction.objects.create(
        user=user,
        abonnement=abonnement,
        payment_method=payment_method,
        amount=abonnement.prix_base,
        phone=phone,
        transaction_id=unique_id,
        status="INITIATED"
    )

    # 2) Appel au provider sélectionné
    provider = get_provider_for_method(payment_method)
    response = provider.initiate_payment(
        amount=tx.amount,
        phone=phone,
        abonnement=abonnement,
        user=user,
        callback_url=callback_url
    )

    # 3) Sauvegarde du raw_response
    try:
        tx.raw_response = response.json()
    except Exception:
        tx.raw_response = {"error": "no json", "text": response.text}

    # 4) Mise à jour du statut + idFromClient si fourni
    if getattr(response, 'status_code', None) in (200, 201) and isinstance(tx.raw_response, dict):
        # Touchpay renvoie 'idFromClient'
        if tx.raw_response.get("idFromClient"):
            tx.transaction_id = tx.raw_response["idFromClient"]
        tx.status = "PROCESSING"
    else:
        tx.status = "FAIL"

    tx.save()
    return tx
