from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SubscriptionType
from .serializers import (
    SubscriptionTypeSerializer,
    PromoActivateSerializer,
    UserAbonnementSerializer
)
from .services import activer_abonnement_via_promo, user_abonnement_actif

class SubscriptionTypeListAPI(generics.ListAPIView):
    """
    GET /api/abonnement/types/ → lister les offres d'abonnement actives
    """
    queryset = SubscriptionType.objects.filter(actif=True)
    serializer_class = SubscriptionTypeSerializer
    permission_classes = [IsAuthenticated]

class ActivatePromoAPI(generics.GenericAPIView):
    """
    POST /api/abonnement/activate-promo/ { code_promo } → activation via code promo
    """
    serializer_class = PromoActivateSerializer
    permission_classes = [IsAuthenticated]

    def post(self, request):
        code = request.data.get('code_promo', '').strip().upper()
        abo, msg = activer_abonnement_via_promo(request.user, code)
        if abo:
            data = UserAbonnementSerializer(abo).data
            return Response({
                "detail": msg,
                "abonnement": data
            }, status=status.HTTP_200_OK)
        return Response({"detail": msg}, status=status.HTTP_400_BAD_REQUEST)

class CurrentSubscriptionAPI(generics.RetrieveAPIView):
    """
    GET /api/abonnement/current/ → récupérer l'abonnement le plus récent
    """
    serializer_class = UserAbonnementSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user.userabonnement_set.order_by('-date_debut').first()
