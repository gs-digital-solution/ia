import json
import traceback
import logging
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.generics import ListAPIView

from .serializers import (
    PaymentMethodSerializer,
    PaymentStartSerializer,
    PaymentTransactionSerializer
)
from .models import PaymentMethod, PaymentTransaction
from .services import process_payment
from abonnement.models import SubscriptionType


class PaymentMethodListAPI(generics.ListAPIView):
    serializer_class = PaymentMethodSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = PaymentMethod.objects.filter(actif=True)
        pid = self.request.query_params.get('pays')
        if pid:
            qs = qs.filter(pays_id=pid)
        return qs


class StartPaymentAPI(APIView):
    """
    POST /api/paiement/start/
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = PaymentStartSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=400)

        try:
            abo = SubscriptionType.objects.get(
                pk=ser.validated_data['abonnement_id'], actif=True
            )
            pm = PaymentMethod.objects.get(
                code=ser.validated_data['method_code'], actif=True
            )
        except Exception:
            return Response(
                {"detail": "Offre ou méthode invalide"},
                status=400
            )

        # Vérifier si c'est un paiement externe
        if pm.est_externe():
            # Pour les paiements externes, pas besoin de callback_url
            callback_url = None
        else:
            callback_url = request.build_absolute_uri('/api/paiement/callback/')

        phone_input = ser.validated_data['phone']

        try:
            tx = process_payment(
                user=request.user,
                abonnement=abo,
                phone=phone_input,
                payment_method=pm,
                callback_url=callback_url
            )

            # Si c'est un paiement externe, retourner le lien
            response_data = PaymentTransactionSerializer(tx).data
            if pm.est_externe() and tx.raw_response:
                response_data.update({
                    'type_paiement': 'EXTERNE',
                    'lien_paiement': tx.raw_response.get('lien_paiement'),
                    'instructions': tx.raw_response.get('instructions'),
                    'montant': tx.raw_response.get('montant'),
                    'nom_abonnement': tx.raw_response.get('nom_abonnement'),
                })

            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception:
            tb = traceback.format_exc()
            logging.getLogger(__name__).error("Erreur process_payment:\n%s", tb)
            return Response(
                {"detail": "Erreur interne du serveur. Veuillez réessayer plus tard."},
                status=500
            )

class PaymentStatusAPI(generics.RetrieveAPIView):
    queryset = PaymentTransaction.objects.all()
    serializer_class = PaymentTransactionSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'transaction_id'




class PaymentTransactionListAPI(ListAPIView):
    """Liste des paiements de l’utilisateur connecté"""
    serializer_class = PaymentTransactionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return PaymentTransaction.objects.filter(user=self.request.user).order_by('-created')