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
    serializer_class = PaymentMethodSerializer
    permission_classes = [IsAuthenticated]
    def get_queryset(self):
        qs = PaymentMethod.objects.filter(actif=True)
        pid = self.request.query_params.get('pays')
        if pid:
            qs = qs.filter(pays_id=pid)
        return qs

class StartPaymentAPI(generics.GenericAPIView):
    serializer_class = PaymentStartSerializer
    permission_classes = [IsAuthenticated]
    def post(self, request):
        ser = self.get_serializer(data=request.data)
        ser.is_valid(raise_exception=True)
        try:
            abo = SubscriptionType.objects.get(pk=ser.validated_data['abonnement_id'], actif=True)
            pm  = PaymentMethod.objects.get(code=ser.validated_data['method_code'], actif=True)
        except:
            return Response({"detail":"Offre ou méthode invalide"}, status=400)
        callback_url = request.build_absolute_uri('/api/paiement/callback/')
        print(f"📞 CALLBACK URL: {callback_url}")
        tx = process_payment(request.user, abo, ser.validated_data['phone'], pm, callback_url)
        data = PaymentTransactionSerializer(tx).data
        return Response(data, status=201 if tx.status=="PROCESSING" else 400)

class PaymentStatusAPI(generics.RetrieveAPIView):
    queryset = PaymentTransaction.objects.all()
    serializer_class = PaymentTransactionSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'transaction_id'