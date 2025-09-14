from rest_framework.generics import ListAPIView
from rest_framework.permissions import AllowAny
from .models import Pays, SousSysteme
from .serializers import PaysSerializer, SousSystemeSerializer

class PaysListAPIView(ListAPIView):
    """
    GET /api/pays/ → Liste publique des pays
    """
    permission_classes = [AllowAny]
    queryset = Pays.objects.all().order_by('nom')
    serializer_class = PaysSerializer

class SousSystemeListAPIView(ListAPIView):
    """
    GET /api/sous-systeme/?pays=<id> → Liste publique
    des sous-systèmes filtrés par pays
    """
    permission_classes = [AllowAny]
    serializer_class = SousSystemeSerializer

    def get_queryset(self):
        pays_id = self.request.query_params.get('pays')
        qs = SousSysteme.objects.all().order_by('nom')
        if pays_id:
            qs = qs.filter(pays_id=pays_id)
        return qs
