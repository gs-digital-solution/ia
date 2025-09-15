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
        # AJOUT: Log des données reçues
        print(f"🎯 DONNÉES REÇUES: {request.data}")
        print(f"🎯 UTILISATEUR: {request.user}")
        print(f"🎯 HEADERS: {dict(request.headers)}")

        ser = self.get_serializer(data=request.data)

        # AJOUT: Log des erreurs de validation
        if not ser.is_valid():
            print(f"❌ ERREURS VALIDATION: {ser.errors}")
            return Response({"detail": "Données invalides", "errors": ser.errors}, status=400)

        print(f"✅ DONNÉES VALIDÉES: {ser.validated_data}")

        try:
            abo = SubscriptionType.objects.get(pk=ser.validated_data['abonnement_id'], actif=True)
            pm = PaymentMethod.objects.get(code=ser.validated_data['method_code'], actif=True)
            print(f"✅ ABONNEMENT TROUVÉ: {abo}")
            print(f"✅ MÉTHODE PAIEMENT TROUVÉE: {pm}")
        except SubscriptionType.DoesNotExist:
            print(f"❌ ABONNEMENT NON TROUVÉ: ID {ser.validated_data['abonnement_id']}")
            return Response({"detail": "Offre invalide"}, status=400)
        except PaymentMethod.DoesNotExist:
            print(f"❌ MÉTHODE PAIEMENT NON TROUVÉE: CODE {ser.validated_data['method_code']}")
            return Response({"detail": "Méthode de paiement invalide"}, status=400)
        except Exception as e:
            print(f"❌ ERREUR INATTENDUE: {e}")
            return Response({"detail": "Offre ou méthode invalide"}, status=400)

        callback_url = request.build_absolute_uri('/api/paiement/callback/')
        print(f"📞 CALLBACK URL: {callback_url}")

        try:
            tx = process_payment(request.user, abo, ser.validated_data['phone'], pm, callback_url)
            print(f"✅ TRANSACTION CRÉÉE: {tx.transaction_id} - STATUT: {tx.status}")

            data = PaymentTransactionSerializer(tx).data
            return Response(data, status=201 if tx.status == "PROCESSING" else 400)

        except Exception as e:
            print(f"❌ ERREUR PROCESS_PAYMENT: {e}")
            return Response({"detail": "Erreur lors du traitement du paiement"}, status=500)

class PaymentStatusAPI(generics.RetrieveAPIView):
    queryset = PaymentTransaction.objects.all()
    serializer_class = PaymentTransactionSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'transaction_id'