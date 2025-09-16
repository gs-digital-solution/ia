import uuid
import logging

from .models import PaymentTransaction
from .providers.factory import get_provider_for_method

logger = logging.getLogger('paiement')

def process_payment(user, abonnement, phone, payment_method, callback_url):
    logger.debug("→ process_payment begins")
    tx = PaymentTransaction.objects.create(
        user=user,
        abonnement=abonnement,
        payment_method=payment_method,
        amount=abonnement.prix_base,
        phone=phone,
        transaction_id="init_"+str(uuid.uuid4())[:12],
        status="INITIATED"
    )

    provider = get_provider_for_method(payment_method)
    resp = provider.initiate_payment(
        amount=tx.amount, phone=phone,
        abonnement=abonnement, user=user,
        callback_url=callback_url
    )

    try:
        tx.raw_response = resp.json()
    except:
        tx.raw_response = {'text': resp.text, 'status_code': resp.status_code}

    # ✅ CORRECTION : Logique plus flexible pour Touchpay
    if resp.status_code in (200, 201):
        if 'idFromClient' in tx.raw_response:
            tx.transaction_id = tx.raw_response['idFromClient']
        tx.status = "PROCESSING"
    else:
        tx.status = "FAIL"
        logger.error(f"Échec initiation paiement: {resp.status_code} - {tx.raw_response}")

    tx.save()
    return tx