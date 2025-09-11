from rest_framework import serializers
from .models import PaymentMethod, PaymentTransaction

class PaymentMethodSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaymentMethod
        fields = ['code', 'nom_affiche', 'operateur', 'pays', 'ussd', 'logo_url']

class PaymentStartSerializer(serializers.Serializer):
    abonnement_id = serializers.IntegerField()
    method_code   = serializers.CharField()
    phone         = serializers.CharField()

class PaymentTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaymentTransaction
        fields = ['transaction_id', 'status', 'amount', 'payment_method', 'created']