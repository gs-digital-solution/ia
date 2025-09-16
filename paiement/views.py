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
    Callback Touchpay : appelée par l'API lors de la finalisation/validation d'un paiement.
    Version ORIGINALE qui fonctionnait.
    """
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body.decode())
        except Exception:
            return JsonResponse({"status": "fail", "error": "invalid json"}, status=400)

        transaction_id = data.get('transaction_id') or data.get('idFromClient')
        status = data.get('status') or data.get('transactionStatus') or data.get('state')

        print("Callback reçu :", transaction_id, status, data)

        if not transaction_id:
            return JsonResponse({"status": "fail", "error": "no id"}, status=400)

        # 1. On retrouve la transaction concernée
        try:
            tx = PaymentTransaction.objects.get(transaction_id=transaction_id)
        except PaymentTransaction.DoesNotExist:
            return JsonResponse({"status": "fail", "error": "tx not found"}, status=404)

        # 2. On met à jour la transaction avec le statut et le payload complet
        tx.status = (status or "UNKNOWN").upper()
        tx.raw_response = data
        tx.save()

        # 3. Si paiement validé -> crédit de l'abonnement/crédit
        if tx.status in ("SUCCESS", "PAID", "VALIDATED"):
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
                    code_promo_utilise=None,
                    exercice_restants=tx.abonnement.nombre_exercices_total,
                    date_debut=timezone.now(),
                    date_fin=timezone.now() + timezone.timedelta(days=tx.abonnement.duree_jours),
                    statut='actif'
                )
                subject = f"💰 Paiement CIS validé [{tx.user}]"
                message = f"Un paiement CIS vient d'être validé.\n\nUtilisateur : {tx.user}\nAbonnement : {tx.abonnement.nom}\nMontant : {tx.amount} FCFA"
                send_mail(
                    subject,
                    message,
                    None,
                    [settings.PAYMENT_ADMIN_EMAIL],
                    fail_silently=True
                )

        return JsonResponse({"status": "ok", "app_status": tx.status})

    else:
        return JsonResponse({"status": "method_not_allowed"}, status=405)