import uuid, logging
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
    logger.debug(f"Tx init: {tx.transaction_id}")

    provider = get_provider_for_method(payment_method)
    resp = provider.initiate_payment(
        amount=tx.amount, phone=phone,
        abonnement=abonnement, user=user,
        callback_url=callback_url
    )
    logger.debug(f"Provider responded {resp.status_code}")

    try:
        tx.raw_response = resp.json()
    except:
        tx.raw_response = {'text': resp.text}
    logger.debug(f"raw_response: {tx.raw_response}")

    if resp.status_code in (200,201) and 'idFromClient' in tx.raw_response:
        tx.transaction_id = tx.raw_response['idFromClient']
        tx.status = "PROCESSING"
    else:
        tx.status = "FAIL"
    tx.save()
    logger.debug(f"Tx updated: {tx.status}")
    return tx
