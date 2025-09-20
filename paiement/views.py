import json
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from abonnement.models import SubscriptionType, UserAbonnement
from resources.models import Pays
from .models import PaymentMethod, PaymentTransaction
from .services import process_payment
from .models import PaymentTransaction
import json
import os
import jwt
from jwt import InvalidTokenError
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.utils import timezone
# facultatif : pour envoyer un mail à chaque réussite
from django.core.mail import send_mail

def start_payment(request):
    if request.method == "POST":
        abo_id = request.POST.get("abonnement_id")
        method_code = request.POST.get("method_code")
        phone = request.POST.get("phone")
        abo = SubscriptionType.objects.get(pk=abo_id)
        payment_method = PaymentMethod.objects.get(code=method_code)
        callback_url = request.build_absolute_uri('/paiement/callback/')

        tx = process_payment(
            user=request.user,
            abonnement=abo,
            phone=phone,
            payment_method=payment_method,
            callback_url=callback_url
        )
        if tx.status == "PROCESSING":
            return render(
                request,
                "paiement/waiting_for_payment.html",
                {"transaction": tx}
            )
        return render(
            request,
            "paiement/payment_error.html",
            {"error": "Erreur lors du paiement."}
        )
    else:
        offres = SubscriptionType.objects.filter(actif=True)
        pays_list = Pays.objects.all()
        methods = PaymentMethod.objects.filter(actif=True)
        return render(
            request,
            "paiement/choix_offre.html",
            {"offres": offres, "pays": pays_list, "methods": methods}
        )






@csrf_exempt
def payment_callback(request):
    """
    Callback unifié pour Touchpay (POST JSON) et Campay (GET/POST JSON).
    1) Récupère data
    2) Vérifie la signature JWT de Campay si présente
    3) Extrait la transaction et le statut
    4) Met à jour la transaction
    5) Crée un abonnement si paiement validé (idempotent)
    6) Retourne un JSON minimal
    """
    # 1) Récupération des données
    if request.method == 'GET':
        data = request.GET.dict()
    else:
        try:
            data = json.loads(request.body.decode())
        except Exception:
            return JsonResponse({'status': 'fail', 'error': 'invalid json'}, status=400)

    # 2) Validation de la signature JWT (Campay)
    sig = data.get('signature')
    if sig:
        secret = os.getenv('CAMPAY_CMR_WEBHOOK_SECRET')
        if not secret:
            return JsonResponse({'status': 'fail', 'error': 'no webhook secret'}, status=400)
        try:
            payload = jwt.decode(sig, secret, algorithms=['HS256'])
            data.update(payload)
        except InvalidTokenError:
            return JsonResponse({'status': 'fail', 'error': 'invalid signature'}, status=400)

    # 3) Extraction de l’ID et du statut
    tx_id = data.get('transaction_id') or data.get('idFromClient') or data.get('reference')
    stat  = data.get('status') or data.get('transactionStatus') or data.get('state')
    if not tx_id or not stat:
        return JsonResponse({'status': 'fail', 'error': 'missing fields'}, status=400)

    # 4) Mise à jour de la transaction
    try:
        tx = PaymentTransaction.objects.get(transaction_id=tx_id)
    except PaymentTransaction.DoesNotExist:
        return JsonResponse({'status': 'fail', 'error': 'tx not found'}, status=404)

    tx.status = stat.upper()
    tx.raw_response = data
    tx.save()

    # 5) Création idempotente de l’abonnement si paiement validé
    if tx.status in ('SUCCESS', 'SUCCESSFUL', 'PAID', 'VALIDATED'):
        exists = UserAbonnement.objects.filter(
            utilisateur=tx.user,
            abonnement=tx.abonnement,
            statut='actif',
            date_fin__gt=timezone.now()
        ).exists()
        if not exists:
            UserAbonnement.objects.create(
                utilisateur=tx.user,
                abonnement=tx.abonnement,
                exercice_restants=tx.abonnement.nombre_exercices_total,
                date_debut=timezone.now(),
                date_fin=timezone.now() + timezone.timedelta(days=tx.abonnement.duree_jours),
                statut='actif'
            )
            # Envoi d'un mail d'alerte si nécessaire
            try:
                subject = f"Paiement validé – {tx.user}"
                message = (
                    f"Utilisateur : {tx.user}\n"
                    f"Abonnement : {tx.abonnement.nom}\n"
                    f"Montant : {tx.amount}\n"
                    f"Provider : {tx.payment_method.code}"
                )
                send_mail(subject, message, None, [settings.PAYMENT_ADMIN_EMAIL])
            except Exception:
                pass

    # 6) Réponse JSON minimale
    return JsonResponse({'status': 'ok', 'app_status': tx.status})