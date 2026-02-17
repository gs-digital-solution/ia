from rest_framework import serializers
from .models import CustomUser, Pays, SousSysteme
from abonnement.models import UserAbonnement, SubscriptionType
from django.db import transaction
from abonnement.models import UserAbonnement, SubscriptionType
import logging
logger = logging.getLogger(__name__)


class UserRegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    pays = serializers.PrimaryKeyRelatedField(queryset=Pays.objects.all(), required=True)
    sous_systeme = serializers.PrimaryKeyRelatedField(queryset=SousSysteme.objects.all(), required=True)

    class Meta:
        model = CustomUser
        fields = [
            'first_name',
            'whatsapp_number',
            'pays',
            'sous_systeme',
            'secret_question',
            'secret_answer',
            'password',
        ]
        extra_kwargs = {
            'password': {'write_only': True},
            'whatsapp_number': {'validators': []},  # Optionnel : évite les validateurs par défaut
        }

    def create(self, validated_data):
        logger.debug("=" * 50)
        logger.debug("DÉBUT CRÉATION UTILISATEUR")
        logger.debug(f"Données reçues: {validated_data}")

        password = validated_data.pop('password')
        user = CustomUser(**validated_data)
        user.set_password(password)
        user.username = user.whatsapp_number
        user.save()

        logger.debug(f"Utilisateur créé: {user.id} - {user.whatsapp_number}")

        try:
            from abonnement.models import UserAbonnement, SubscriptionType

            logger.debug("Recherche du type d'abonnement...")
            sub_type = SubscriptionType.objects.filter(code='gratuit_promo', actif=True).first()
            logger.debug(f"Type trouvé: {sub_type}")

            if sub_type:
                abonnement = UserAbonnement.objects.create(
                    utilisateur=user,
                    abonnement=sub_type,
                    exercice_restants=1,
                )
                logger.debug(f"✅ Abonnement créé: ID={abonnement.id}")
            else:
                logger.error("❌ Aucun type d'abonnement trouvé!")

        except Exception as e:
            logger.error(f"❌ Erreur création abonnement: {e}", exc_info=True)

        logger.debug("FIN CRÉATION UTILISATEUR")
        logger.debug("=" * 50)

        return user

from .models import DemandeCorrection, SoumissionIA, CorrigePartiel

class CorrigePartielSerializer(serializers.ModelSerializer):
    url_pdf = serializers.SerializerMethodField()  # ← CHANGÉ : 'url_pdf' pour matcher Flutter
    titre = serializers.CharField(source='titre_exercice')  # ← CHANGÉ : 'titre' pour matcher Flutter

    class Meta:
        model = CorrigePartiel
        fields = [
            'id',
            'titre',          # ← CHANGÉ : était 'titre_exercice'
            'date_creation',
            'url_pdf',        # ← CHANGÉ : était 'fichier_pdf_url'
        ]

    def get_url_pdf(self, obj):
        if not obj.fichier_pdf:
            return None
        request = self.context.get('request')
        try:
            if request and hasattr(obj.fichier_pdf, 'url'):
                return request.build_absolute_uri(obj.fichier_pdf.url)
        except:
            pass
        return None

class SoumissionIASerializer(serializers.ModelSerializer):
    corriges = CorrigePartielSerializer(many=True, read_only=True)
    global_pdf_url = serializers.SerializerMethodField()
    date_soumission = serializers.DateTimeField(  # ← CHANGÉ : 'date_soumission' pour matcher Flutter
        source='date_creation', format="%d/%m/%Y %H:%M"
    )

    class Meta:
        model = SoumissionIA
        fields = [
            'id',
            'exercice_index',
            'statut',
            'date_soumission',  # ← CHANGÉ : était 'date_creation'
            'corriges',
            'global_pdf_url',
        ]

    def get_global_pdf_url(self, obj):
        request = self.context.get('request')
        if not obj.resultat_json:
            return None
        pdf_url = obj.resultat_json.get('pdf_url')
        if pdf_url and request:
            return request.build_absolute_uri(pdf_url)
        return pdf_url  # ou None


class HistoriqueSerializer(serializers.ModelSerializer):
    matiere = serializers.SerializerMethodField()
    nom_fichier = serializers.SerializerMethodField()  # ← CHANGÉ en SerializerMethodField
    date = serializers.DateTimeField(source='date_soumission', format="%d/%m/%Y %H:%M")
    soumissions = SoumissionIASerializer(source='soumissionia_set', many=True, required=False)

    class Meta:
        model = DemandeCorrection
        fields = ['id', 'matiere', 'nom_fichier', 'date', 'soumissions']

    def get_matiere(self, obj):
        return obj.matiere.nom if obj.matiere else "Matière inconnue"

    def get_nom_fichier(self, obj):
        """
        Récupère le nom du fichier de manière robuste.
        Gère les anciens enregistrements sans champ nom_fichier.
        """
        try:
            # Essayer d'abord le champ nom_fichier
            if hasattr(obj, 'nom_fichier') and obj.nom_fichier:
                return obj.nom_fichier

            # Fallback: extraire du fichier
            if obj.fichier and hasattr(obj.fichier, 'name'):
                import os
                return os.path.basename(obj.fichier.name)

            # Fallback final
            return f"Sujet #{obj.id}"

        except Exception as e:
            # En cas d'erreur, retourner un fallback
            print(f"⚠️ Erreur dans get_nom_fichier pour obj {obj.id}: {e}")
            return f"Sujet #{obj.id}"