import json, os
import traceback
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from tqdm.contrib import logging

from .serializers import (
    PaymentMethodSerializer,
    PaymentStartSerializer,
    PaymentTransactionSerializer
)
from .models import PaymentMethod, PaymentTransaction
from .services import process_payment
from abonnement.models import SubscriptionType


def normalize_phone(phone: str) -> str:
    """
    Nettoie et formate le numéro en MSISDN commençant par 237.
    Exemples :
      "691234567"    → "237691234567"
      "0691234567"   → "237691234567"
      "+237691234567"→ "237691234567"
      "1234567"      → "2371234567"
    """
    # Garde que les chiffres
    num = ''.join(c for c in phone if c.isdigit())
    if num.startswith('0'):
        num = '237' + num[1:]
    elif not num.startswith('237'):
        num = '237' + num
    return num


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

        # Vérifier l'offre et la méthode
        try:
            abo = SubscriptionType.objects.get(
                pk=ser.validated_data['abonnement_id'], actif=True
            )
            pm  = PaymentMethod.objects.get(
                code=ser.validated_data['method_code'], actif=True
            )
        except Exception:
            return Response(
                {"detail": "Offre ou méthode invalide"},
                status=400
            )

        # Construire l'URL de callback (publique)
        callback_url = request.build_absolute_uri('/api/paiement/callback/')
        print(">>> StartPaymentAPI callback_url:", callback_url)

        # 1) Normaliser le numéro avant l'appel provider
        phone_raw   = ser.validated_data['phone']
        phone_clean = normalize_phone(phone_raw)
        print(">>> StartPaymentAPI phone_raw:", phone_raw, "→ phone_clean:", phone_clean)

        # 2) Lancer le paiement
        try:
            tx = process_payment(
                user=request.user,
                abonnement=abo,
                phone=phone_clean,
                payment_method=pm,
                callback_url=callback_url
            )
            return Response(
                PaymentTransactionSerializer(tx).data,
                status=status.HTTP_201_CREATED
            )
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