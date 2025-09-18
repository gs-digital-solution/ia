from rest_framework import serializers
from .models import SubscriptionType, PromoCode, UserAbonnement

class SubscriptionTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubscriptionType
        fields = [
            'id', 'code', 'nom', 'description',
            'prix_base', 'taux_reduction',
            'nombre_exercices_total', 'duree_jours', 'actif'
        ]

class PromoActivateSerializer(serializers.Serializer):
    code_promo = serializers.CharField(max_length=12)

class UserAbonnementSerializer(serializers.ModelSerializer):
    abonnement = SubscriptionTypeSerializer()
    class Meta:
        model = UserAbonnement
        fields = [
            'id', 'abonnement', 'date_debut',
            'date_fin', 'exercice_restants', 'statut'
        ]
