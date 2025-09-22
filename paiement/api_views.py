import json
import traceback
import logging
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

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
            pm  = PaymentMethod.objects.get(
                code=ser.validated_data['method_code'], actif=True
            )
        except Exception:
            return Response(
                {"detail": "Offre ou méthode invalide"},
                status=400
            )

        callback_url = request.build_absolute_uri('/api/paiement/callback/')
        print(">>> StartPaymentAPI callback_url:", callback_url)

        # On passe le numéro tel quel ; chaque provider le normalisera à sa façon
        phone_input = ser.validated_data['phone']
        print(">>> StartPaymentAPI raw phone:", phone_input)

        try:
            tx = process_payment(
                user=request.user,
                abonnement=abo,
                phone=phone_input,
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