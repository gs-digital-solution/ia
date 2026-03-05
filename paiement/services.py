import uuid
import logging
from django.utils import timezone
from .models import PaymentTransaction
from .providers.factory import get_provider_for_method

logger = logging.getLogger('paiement')


def process_payment(user, abonnement, phone, payment_method, callback_url):
    """
    Crée un PaymentTransaction puis appelle le provider.
    """
    logger.debug("→ process_payment begins")

    # 1) Création initiale du record DB
    tx = PaymentTransaction.objects.create(
        user=user,
        abonnement=abonnement,
        payment_method=payment_method,
        amount=abonnement.prix_base,
        phone=phone,
        transaction_id="init_" + str(uuid.uuid4())[:12],
        status="INITIATED"
    )
    logger.debug(f"Tx init: {tx.transaction_id}")

    # 2) Appel au provider
    provider = get_provider_for_method(payment_method)

    # Pour les paiements externes, pas besoin de callback_url
    if payment_method.est_externe():
        callback_url = None

    result = provider.initiate_payment(
        amount=tx.amount,
        phone=phone,
        abonnement=abonnement,
        user=user,
        callback_url=callback_url
    )

    # 3) Gestion selon le type de paiement
    if payment_method.est_externe():
        # PAIEMENT EXTERNE (Lygos)
        logger.debug(f"Paiement externe initié: {result}")

        # Mettre à jour la transaction avec le résultat
        tx.transaction_id = result['transaction_id']
        tx.raw_response = result
        tx.status = "PENDING_EXTERNAL"

        tx.save()
        logger.debug(f"Tx externe créée: {tx.transaction_id} - Statut: {tx.status}")
        return tx

    # ════════════════════════════════════════════════════════════════
    #   PAIEMENT IKEEPAY (iframe)
    # ════════════════════════════════════════════════════════════════
    elif payment_method.code.startswith('IKEEPAY'):
        logger.debug(f"IkeePay provider returned: {result}")

        tx.transaction_id = result['transaction_id']
        tx.status = result['status']
        tx.raw_response = result.get('raw_response', result)
        tx.save()

        logger.debug(f"Tx iKeePay créée: {tx.transaction_id} - Statut: {tx.status}")
        return tx

    # ════════════════════════════════════════════════════════════════
    #   PAIEMENT INTERNE (Touchpay/Campay) - AVEC TRAITEMENT CALLBACK
    # ════════════════════════════════════════════════════════════════
    else:
        logger.debug(f"Provider responded HTTP {result.status_code}")

        try:
            raw = result.json()
        except (ValueError, AttributeError):
            raw = {"text": str(result) if not hasattr(result, 'text') else result.text}

        tx.raw_response = raw
        logger.debug(f"raw_response: {raw}")

        code = payment_method.code.upper()
        initial_status = tx.status

        # Touchpay
        if code.startswith("TOUCHPAY"):
            if result.status_code in (200, 201) and 'idFromClient' in raw:
                tx.transaction_id = raw['idFromClient']
                tx.status = "PROCESSING"
            else:
                tx.status = "FAIL"

        # Campay
        elif code.startswith("CAMPAY"):
            if result.status_code == 200 and raw.get('reference'):
                tx.transaction_id = raw['reference']
                tx.status = "PROCESSING"
            else:
                tx.status = "FAIL"

        else:
            tx.status = "FAIL"

        tx.save()
        logger.debug(f"Tx updated: {tx.status}")

        # ════════════════════════════════════════════════════════════════
        #   SI LE PAIEMENT EST DÉJÀ SUCCESS (via callback synchrone)
        #   NOTE: Pour les vrais callbacks asynchrones, ceci est géré
        #   dans la vue payment_callback (callback_url)
        # ════════════════════════════════════════════════════════════════
        # Certains providers peuvent retourner un succès immédiat
        if tx.status == 'SUCCESS' or (tx.status == 'PROCESSING' and raw.get('status') == 'SUCCESS'):
            from abonnement.models import UserAbonnement
            from django.utils import timezone

            now = timezone.now()

            # Chercher un abonnement actif existant
            abo_existant = UserAbonnement.objects.filter(
                utilisateur=user,
                statut='actif',
                date_fin__gt=now
            ).first()

            if abo_existant:
                # ✅ CUMUL
                logger.info(f"🔄 [process_payment] Cumul pour utilisateur {user.id}")
                abo_existant.exercice_restants += abonnement.nombre_exercices_total
                abo_existant.date_fin += timezone.timedelta(days=abonnement.duree_jours)
                abo_existant.save()
                logger.info(f"   ✅ Nouveaux crédits: {abo_existant.exercice_restants}")
            else:
                # ✅ Création
                logger.info(f"🆕 [process_payment] Création abonnement pour utilisateur {user.id}")
                UserAbonnement.objects.create(
                    utilisateur=user,
                    abonnement=abonnement,
                    code_promo_utilise=None,
                    date_debut=now,
                    date_fin=now + timezone.timedelta(days=abonnement.duree_jours),
                    exercice_restants=abonnement.nombre_exercices_total,
                    statut='actif'
                )

        return tx