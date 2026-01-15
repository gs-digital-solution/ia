from rest_framework import serializers
from .models import PaymentMethod, PaymentTransaction
from resources.models import Pays

class PaysSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pays
        fields = ['id', 'nom']


class PaymentMethodSerializer(serializers.ModelSerializer):
    est_externe = serializers.SerializerMethodField()

    class Meta:
        model = PaymentMethod
        fields = [
            'id', 'code', 'nom_affiche', 'operateur', 'type_paiement',
            'pays', 'ussd', 'service_code', 'lien_externe',
            'instructions_externes', 'logo_url', 'description',
            'actif', 'priorite', 'est_externe'
        ]

    def get_est_externe(self, obj):
        return obj.est_externe()

class PaymentStartSerializer(serializers.Serializer):
    abonnement_id = serializers.IntegerField()
    method_code   = serializers.CharField()
    phone         = serializers.CharField()


class PaymentTransactionSerializer(serializers.ModelSerializer):
    payment_method = PaymentMethodSerializer(read_only=True)
    est_externe = serializers.SerializerMethodField()

    class Meta:
        model = PaymentTransaction
        fields = [
            'id', 'user', 'abonnement', 'payment_method',
            'amount', 'phone', 'transaction_id', 'status',
            'created', 'updated', 'raw_response', 'est_externe'
        ]

    def get_est_externe(self, obj):
        return obj.payment_method.est_externe()