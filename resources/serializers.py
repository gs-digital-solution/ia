from rest_framework import serializers
from .models import Pays, SousSysteme

class PaysSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pays
        fields = ['id', 'nom', 'code', 'indicatif']

class SousSystemeSerializer(serializers.ModelSerializer):
    pays = PaysSerializer(read_only=True)
    pays_id = serializers.PrimaryKeyRelatedField(queryset=Pays.objects.all(), source='pays', write_only=True)
    class Meta:
        model = SousSysteme
        fields = ['id', 'nom', 'pays', 'pays_id']