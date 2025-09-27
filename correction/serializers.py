from rest_framework import serializers
from .models import CustomUser, Pays, SousSysteme

class UserRegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    pays = serializers.PrimaryKeyRelatedField(queryset=Pays.objects.all(), required=True)
    sous_systeme = serializers.PrimaryKeyRelatedField(queryset=SousSysteme.objects.all(), required=True)

    class Meta:
        model = CustomUser
        fields = [
            'first_name',         # Prénom
            #'gmail',               Email utilisateur
            'whatsapp_number',    # Numéro WhatsApp
            'pays',               # FK objet (id)
            'sous_systeme',       # FK objet (id)
            'secret_question',    # Question secrète (contenu)
            'secret_answer',      # Réponse
            'password',           # Mot de passe
            # ON NE DEMANDE PAS code_promo ni role à la création !
        ]

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = CustomUser(**validated_data)
        user.set_password(password)
        user.username = user.whatsapp_number  # Peut-être adapter selon ta logique
        # Code promo généré plus tard (par signal post_save ou dans la vue)
        user.save()
        return user