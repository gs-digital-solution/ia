from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from abonnement.models import SubscriptionType
from .models import PaymentMethod
from .services import process_payment
from resources.models import Pays
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import PaymentTransaction
from django.utils import timezone
from abonnement.models import UserAbonnement
from django.core.mail import send_mail
from django.conf import settings


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
        tx = process_payment(
            user=request.user,
            abonnement=abo,
            phone=phone,
            payment_method=payment_method,
            callback_url=callback_url
        )
        if tx.status == "PROCESSING":
            return render(request, "paiement/waiting_for_payment.html", {"transaction": tx})
        return render(request, "paiement/payment_error.html", {"error": "Erreur lors du paiement."})
    else:
        offres = SubscriptionType.objects.filter(actif=True)
        pays = Pays.objects.all()
        methods = PaymentMethod.objects.filter(actif=True)
        return render(request, "paiement/choix_offre.html", {"offres": offres, "pays": pays, "methods": methods})




@csrf_exempt
def payment_callback(request):
    """
    Callback Touchpay : appelée par l'API lors de la finalisation/validation d'un paiement.
    1. Met à jour la transaction en BDD.
    2. Si succès, crédite/crée l'abonnement utilisateur.
    """
    if request.method in ['POST', 'GET']:  # ACCEPTER GET ET POST
        import json
        try:
            # Gérer les données selon la méthode
            if request.method == 'POST':
                data = json.loads(request.body.decode())
            else:  # GET
                data = request.GET.dict()
        except Exception:
            return JsonResponse({"status": "fail", "error": "invalid data"}, status=400)

        transaction_id = data.get('transaction_id') or data.get('idFromClient')
        status = data.get('status') or data.get('transactionStatus') or data.get('state')

        print("Callback reçu :", transaction_id, status, data)  # Log/DEBUG

        if not transaction_id:
            return JsonResponse({"status": "fail", "error": "no id"}, status=400)

        # 1. On retrouve la transaction concernée
        try:
            tx = PaymentTransaction.objects.get(transaction_id=transaction_id)
        except PaymentTransaction.DoesNotExist:
            return JsonResponse({"status": "fail", "error": "tx not found"}, status=404)

        # 2. On met à jour la transaction avec le statut et le payload complet
        tx.status = (status or "UNKNOWN").upper()
        tx.raw_response = data  # Trace complète = utile support/remontée
        tx.save()

        # 3. Si paiement validé -> crédit de l'abonnement/crédit
        if tx.status in ("SUCCESS", "PAID", "VALIDATED"):
            # Vérifie s'il a déjà un abonnement actif équivalent (par sécurité)
            exists = UserAbonnement.objects.filter(
                utilisateur=tx.user,
                abonnement=tx.abonnement,
                statut='actif',
                date_fin__gt=timezone.now()
            ).exists()
            if not exists:
                # Crée/crédite un abonnement utilisateur, pile up si besoin !
                UserAbonnement.objects.create(
                    utilisateur=tx.user,
                    abonnement=tx.abonnement,
                    code_promo_utilise=None,
                    exercice_restants=tx.abonnement.nombre_exercices_total,
                    date_debut=timezone.now(),
                    date_fin=timezone.now() + timezone.timedelta(days=tx.abonnement.duree_jours),
                    statut='actif'
                )
                # Email d'alerte ADMIN
                subject = f"💰 Paiement CIS validé [{tx.user}]"
                message = (
                    f"Un paiement CIS vient d'être validé.\n\n"
                    f"Utilisateur : {tx.user} (id {tx.user.id})\n"
                    f"Abonnement : {tx.abonnement.nom}\n"
                    f"Montant : {tx.amount} FCFA\n"
                    f"Méthode de paiement : {tx.payment_method.nom_affiche}\n"
                    f"Téléphone : {tx.phone}\n"
                    f"Date : {tx.updated.strftime('%d/%m/%Y %H:%M')}\n"
                    f"Transaction ID : {tx.transaction_id}\n"
                    f"Statut Provider : {tx.status}\n"
                )
                send_mail(
                    subject,
                    message,
                    None,  # DEFAULT_FROM_EMAIL
                    [settings.PAYMENT_ADMIN_EMAIL],  # liste, donc tu peux faire [email1, email2, ...]
                    fail_silently=True
                    # Mets à True en prod pour éviter un crash qui bloque le callback si le mail plante
                )

        return JsonResponse({"status": "ok", "app_status": tx.status})

    else:
        return JsonResponse({"status": "method_not_allowed"}, status=405)