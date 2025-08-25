from rest_framework.generics import ListAPIView
from .models import Pays, SousSysteme
from .serializers import PaysSerializer, SousSystemeSerializer

class PaysListAPIView(ListAPIView):
    queryset = Pays.objects.all().order_by('nom')
    serializer_class = PaysSerializer

class SousSystemeListAPIView(ListAPIView):
    serializer_class = SousSystemeSerializer

    def get_queryset(self):
        pays_id = self.request.query_params.get('pays')
        qs = SousSysteme.objects.all().order_by('nom')
        if pays_id:
            qs = qs.filter(pays__id=pays_id)
        return qs