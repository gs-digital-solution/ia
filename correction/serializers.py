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


from .models import DemandeCorrection, SoumissionIA, CorrigePartiel

class CorrigePartielSerializer(serializers.ModelSerializer):
    fichier_pdf_url = serializers.SerializerMethodField()

    class Meta:
        model = CorrigePartiel
        fields = [
            'id',
            'titre_exercice',
            'date_creation',
            'fichier_pdf_url',
        ]

    def get_fichier_pdf_url(self, obj):
        request = self.context.get('request')
        if obj.fichier_pdf and hasattr(obj.fichier_pdf, 'url'):
            return request.build_absolute_uri(obj.fichier_pdf.url)
        return None


class SoumissionIASerializer(serializers.ModelSerializer):
    corriges         = CorrigePartielSerializer(many=True, read_only=True)
    global_pdf_url   = serializers.SerializerMethodField()
    date_creation    = serializers.DateTimeField(format="%d/%m/%Y %H:%M")

    class Meta:
        model = SoumissionIA
        fields = [
            'id',
            'exercice_index',
            'statut',
            'date_creation',
            'corriges',         # liste des CorrigePartiel
            'global_pdf_url',   # ancien PDF global
        ]

    def get_global_pdf_url(self, obj):
        request = self.context.get('request')
        # On récupère l’URL stockée dans resultat_json pour l’ancien pipeline
        pdf_url = obj.resultat_json.get('pdf_url') if obj.resultat_json else None
        if pdf_url:
            # build_absolute_uri gère les chemins relatifs
            return request.build_absolute_uri(pdf_url)
        return None


class HistoriqueSerializer(serializers.ModelSerializer):
    matiere     = serializers.CharField(source='matiere.nom')
    date        = serializers.DateTimeField(source='date_soumission', format="%d/%m/%Y %H:%M")
    soumissions = SoumissionIASerializer(source='soumissionia_set', many=True)

    class Meta:
        model  = DemandeCorrection
        fields = [
            'id',           # id de la DemandeCorrection
            'matiere',      # nom de la matière
            'date',         # date de soumission
            'soumissions',  # liste de vos SoumissionIA et leurs corrigés
        ]