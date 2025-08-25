from .providers import TouchpayProvider
from .models import PaymentTransaction
import uuid

def process_payment(user, abonnement, phone, payment_method, callback_url):
    provider = TouchpayProvider(payment_method)
    amount = abonnement.prix_base
    unique_id = "init_" + str(uuid.uuid4()).replace("-", "")[:12]
    tx = PaymentTransaction.objects.create(
        user=user,
        abonnement=abonnement,
        payment_method=payment_method,
        amount=amount,
        phone=phone,
        transaction_id=unique_id
    )
    response = provider.initiate_payment(
        amount=amount,
        phone=phone,
        abonnement=abonnement,
        user=user,
        callback_url=callback_url
    )
    try:
        tx.raw_response = response.json()
    except Exception:
        tx.raw_response = {"error": "no json", "text": response.text}
    if response.status_code in (200, 201) and "idFromClient" in tx.raw_response:
        tx.transaction_id = tx.raw_response["idFromClient"]
        tx.status = "PROCESSING"
    else:
        tx.status = "FAIL"
    tx.save()
    return tx