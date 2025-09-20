from rest_framework import generics, status
from rest_framework.response import Response
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

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PaymentStartSerializer
from .models import PaymentMethod
from .services import process_payment
import traceback
import logging

logger = logging.getLogger(__name__)

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
                pk=ser.validated_data['abonnement_id'],
                actif=True
            )
            pm = PaymentMethod.objects.get(
                code=ser.validated_data['method_code'],
                actif=True
            )
        except Exception:
            return Response(
                {"detail": "Offre ou méthode invalide"},
                status=400
            )

        callback_url = request.build_absolute_uri('/api/paiement/callback/')

        try:
            tx = process_payment(
                user=request.user,
                abonnement=abo,
                phone=ser.validated_data['phone'],
                payment_method=pm,
                callback_url=callback_url
            )

            # renvoyer la transaction
            from .serializers import PaymentTransactionSerializer
            return Response(
                PaymentTransactionSerializer(tx).data,
                status=status.HTTP_201_CREATED
            )

        except Exception as e:
            # Log complet de la stack
            tb = traceback.format_exc()
            logger.error("Erreur process_payment:\n%s", tb)

            # Renvoyer l'erreur en JSON pour debug
            return Response(
                {"error": str(e), "trace": tb},
                status=500
            )
class PaymentStatusAPI(generics.RetrieveAPIView):
    queryset = PaymentTransaction.objects.all()
    serializer_class = PaymentTransactionSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'transaction_id'
