from rest_framework import generics
from rest_framework.permissions import AllowAny
from .models import Pays, SousSysteme
from .serializers import PaysSerializer, SousSystemeSerializer

class PaysListAPIView(generics.ListAPIView):
    queryset = Pays.objects.all()
    serializer_class = PaysSerializer
    permission_classes = [AllowAny]  # Rendre public
    authentication_classes = []  # Désactiver l'authentification

class SousSystemeListAPIView(generics.ListAPIView):
    serializer_class = SousSystemeSerializer
    permission_classes = [AllowAny]  # Rendre public
    authentication_classes = []  # Désactiver l'authentification

    def get_queryset(self):
        queryset = SousSysteme.objects.all()
        pays_id = self.request.query_params.get('pays')
        if pays_id:
            queryset = queryset.filter(pays_id=pays_id)
        return queryset