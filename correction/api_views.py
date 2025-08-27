from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserRegisterSerializer
from .models import CustomUser, DeviceConnectionHistory
from .models import DeviceMigrationRequest
from rest_framework.permissions import IsAuthenticated
from .models import  Pays, SousSysteme

from rest_framework.views import APIView
from abonnement.services import user_abonnement_actif, debiter_credit_abonnement
from .models import DemandeCorrection, SoumissionIA
from .ia_utils import generer_corrige_ia_et_graphique_async
import json


class UserRegisterAPIView(APIView):
    # permission_classes = [AllowAny] # à n’activer que si tu as activé la protection dans settings/auth

    def post(self, request):
        serializer = UserRegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response(
                {"success": True, "message": "Inscription réussie."},
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# *API de connexion — code complet et expliqué*
class UserLoginAPIView(APIView):
    authentication_classes = []
    permission_classes = []  # Pour le login, pas besoin d'être authentifié d'avance

    def post(self, request):
        whatsapp_number = request.data.get('whatsapp_number')
        password = request.data.get('password')
        device_id = request.data.get('device_id')
        user = CustomUser.objects.filter(whatsapp_number=whatsapp_number).first()
        if not user or not user.check_password(password):
            return Response({"success": False, "error": "Identifiants invalides."}, status=401)
        # Blocage device multi-login :
        if user.device_id and user.device_id != device_id:
            DeviceConnectionHistory.objects.create(user=user, device_id=device_id, successful=False)
            return Response({
                "success": False,
                "error": "Ce compte est déjà utilisé sur un autre appareil. Demandez une migration auprès du support via l’interface d’assistance."
            }, status=403)
        # Première connexion ou device ok :
        if not user.device_id:
            user.device_id = device_id
            user.save()
        DeviceConnectionHistory.objects.create(user=user, device_id=device_id, successful=True)
        # Tu peux renvoyer plus d'infos (token, prénom…) ici :
        return Response({"success": True, "message": "Connexion réussie."})

#vue pour la demande de migration
class CreateDeviceMigrationRequestAPIView(APIView):
    permission_classes = []  # public (pas d'authent)

    def post(self, request):
        whatsapp_number = request.data.get("whatsapp_number")
        new_device_id = request.data.get("new_device_id")
        justification = request.data.get("justification") or ""
        # Récupération user par whatsapp_number
        user = CustomUser.objects.filter(whatsapp_number=whatsapp_number).first()
        if not user:
            return Response({"success": False, "error": "Compte non trouvé."}, status=404)
        if DeviceMigrationRequest.objects.filter(user=user, status='pending').exists():
            return Response({"success": False, "error": "Déjà une demande en attente."}, status=400)
        DeviceMigrationRequest.objects.create(
            user=user,
            previous_device_id=user.device_id,
            new_device_id=new_device_id,
            justification=justification,
            status="pending"
        )
        return Response({"success": True, "message": "Demande de migration envoyée."})


 # vue de réinitialisation de mot de pass sur flutter
class PasswordResetAPI(APIView):
    permission_classes = []  # ou [AllowAny]
    authentication_classes = []

    def post(self, request):
        whatsapp = request.data.get('whatsapp_number')
        answer = request.data.get('secret_answer')
        new_pwd = request.data.get('new_password')

        user = CustomUser.objects.filter(whatsapp_number=whatsapp).first()
        if not user:
            return Response({'success': False, 'error': "Numéro WhatsApp inconnu."}, status=404)

        if request.data.get('check_only'):
            # On affiche juste la question secrète (pour le premier POST)
            return Response({'success': True, 'question': user.secret_question})
        else:
            # Ici, on vérifie la réponse et réinitialise si bon
            if user.check_secret(answer):
                user.set_password(new_pwd)
                user.save()
                return Response({'success': True, 'message': "Mot de passe réinitialisé avec succès."})
            return Response({'success': False, 'error': "Réponse à la question secrète incorrecte."}, status=400)



# vue pour le profil coté flutter
class ProfileAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        data = {
            "first_name": user.first_name,
            "pays": user.pays.id if user.pays else None,
            "sous_systeme": user.sous_systeme.id if user.sous_systeme else None,
            "gmail": user.gmail,
            "whatsapp_number": user.whatsapp_number,
        }
        return Response(data)

    def put(self, request):
        user = request.user
        first_name = request.data.get("first_name")
        pays_id = request.data.get("pays")
        sous_systeme_id = request.data.get("sous_systeme")
        if first_name:
            user.first_name = first_name.strip()
        if pays_id:
            user.pays = Pays.objects.filter(pk=pays_id).first()
        if sous_systeme_id:
            user.sous_systeme = SousSysteme.objects.filter(pk=sous_systeme_id).first()
        user.save()
        return Response({"success": True, "message": "Profil mis à jour."})

# vue pour la soumission coté flutter
class SoumissionExerciceAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Vérifier l'abonnement actif
        if not user_abonnement_actif(request.user):
            return Response(
                {"error": "Abonnement requis pour soumettre un exercice"},
                status=status.HTTP_402_PAYMENT_REQUIRED
            )

        # Récupérer les données
        pays_id = request.data.get('pays')
        sous_systeme_id = request.data.get('sous_systeme')
        classe_id = request.data.get('classe')
        matiere_id = request.data.get('matiere')
        type_exercice_id = request.data.get('type_exercice')
        enonce_texte = request.data.get('enonce_texte')
        fichier = request.FILES.get('fichier')

        # Créer la demande
        demande = DemandeCorrection.objects.create(
            user=request.user,
            pays_id=pays_id,
            sous_systeme_id=sous_systeme_id,
            classe_id=classe_id,
            matiere_id=matiere_id,
            type_exercice_id=type_exercice_id,
            fichier=fichier
        )

        # Créer le suivi IA
        soumission = SoumissionIA.objects.create(
            user=request.user,
            demande=demande,
            statut='en_attente'
        )

        # Lancer le traitement async
        generer_corrige_ia_et_graphique_async.delay(
            enonce_texte,
            f"Exercice de {demande.matiere.nom} - {demande.classe.nom}",
            matiere_id=matiere_id,
            demande_id=demande.id
        )

        # Débiter le crédit
        debiter_credit_abonnement(request.user)

        return Response({
            "success": True,
            "soumission_id": soumission.id,
            "message": "Exercice soumis avec succès. Traitement en cours..."
        })


class StatutSoumissionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, soumission_id):
        try:
            soumission = SoumissionIA.objects.get(id=soumission_id, user=request.user)
            return Response({
                "statut": soumission.statut,
                "resultat": soumission.resultat_json,
                "date_creation": soumission.date_creation
            })
        except SoumissionIA.DoesNotExist:
            return Response({"error": "Soumission non trouvée"}, status=404)