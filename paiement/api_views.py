from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .serializers import (
    PaymentMethodSerializer,
    PaymentStartSerializer,
    PaymentTransactionSerializer
)
from .models import PaymentMethod, PaymentTransaction
from .services import process_payment
from abonnement.models import SubscriptionType

class PaymentMethodListAPI(generics.ListAPIView):
    """
    GET /api/paiement/methods/ → liste des méthodes de paiement actives
    """
    queryset = PaymentMethod.objects.filter(actif=True)
    serializer_class = PaymentMethodSerializer
    permission_classes = [IsAuthenticated]

class StartPaymentAPI(generics.GenericAPIView):
    """
    POST /api/paiement/start/ { abonnement_id, method_code, phone }
    → lance la transaction Touchpay, renvoie transaction_id et status
    """
    serializer_class = PaymentStartSerializer
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = self.get_serializer(data=request.data)
        ser.is_valid(raise_exception=True)

        # récupération des objets
        try:
            abo = SubscriptionType.objects.get(
                pk=ser.validated_data['abonnement_id'], actif=True
            )
            pm  = PaymentMethod.objects.get(
                code=ser.validated_data['method_code'], actif=True
            )
        except (SubscriptionType.DoesNotExist, PaymentMethod.DoesNotExist):
            return Response(
                {"detail": "Offre ou méthode invalide"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # lance le paiement
        callback_url = request.build_absolute_uri('/api/paiement/callback/')
        tx = process_payment(
            user=request.user,
            abonnement=abo,
            phone=ser.validated_data['phone'],
            payment_method=pm,
            callback_url=callback_url
        )

        data = PaymentTransactionSerializer(tx).data
        http_status = status.HTTP_201_CREATED if tx.status == "PROCESSING" else status.HTTP_400_BAD_REQUEST
        return Response(data, status=http_status)

class PaymentStatusAPI(generics.RetrieveAPIView):
    """
    GET /api/paiement/status/<transaction_id>/ → interroger le statut en cours
    """
    queryset = PaymentTransaction.objects.all()
    serializer_class = PaymentTransactionSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'transaction_id'