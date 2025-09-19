import uuid
import logging
from django.utils import timezone
from .models import PaymentTransaction
from .providers.factory import get_provider_for_method

logger = logging.getLogger('paiement')

def process_payment(user, abonnement, phone, payment_method, callback_url):
    """
    Crée un PaymentTransaction puis appelle le provider (Touchpay ou Campay)
    pour initier la transaction. Met à jour le statut INITIATED → PROCESSING ou FAIL.
    """
    logger.debug("→ process_payment begins")

    # 1) Création du record DB
    tx = PaymentTransaction.objects.create(
        user=user,
        abonnement=abonnement,
        payment_method=payment_method,
        amount=abonnement.prix_base,     # montant de l'offre
        phone=phone,                     # num du payeur
        transaction_id="init_" + str(uuid.uuid4())[:12],
        status="INITIATED"
    )
    logger.debug(f"Tx init: {tx.transaction_id}")

    # 2) Appel au provider
    provider = get_provider_for_method(payment_method)
    resp = provider.initiate_payment(
        amount=tx.amount,
        phone=phone,
        abonnement=abonnement,
        user=user,
        callback_url=callback_url
    )
    logger.debug(f"Provider responded HTTP {resp.status_code}")

    # 3) Stockage brut pour debug
    try:
        raw = resp.json()
    except ValueError:
        raw = {"text": resp.text}
    tx.raw_response = raw
    logger.debug(f"raw_response: {raw}")

    code = payment_method.code.upper()

    # 4a) Si c'est Touchpay
    if code.startswith("TOUCHPAY"):
        if resp.status_code in (200, 201) and 'idFromClient' in raw:
            tx.transaction_id = raw['idFromClient']
            tx.status = "PROCESSING"
        else:
            tx.status = "FAIL"

    # 4b) Si c'est Campay
    elif code.startswith("CAMPAY"):
        # Campay renvoie HTTP 200 + { "reference": "<uuid>" } en cas de succès
        if resp.status_code == 200 and raw.get('reference'):
            tx.transaction_id = raw['reference']
            tx.status = "PROCESSING"
        else:
            tx.status = "FAIL"

    else:
        # Aucun provider supporté
        tx.status = "FAIL"

    tx.save()
    logger.debug(f"Tx updated: {tx.status}")
    return tx