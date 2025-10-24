# 1) VUE PAYS et SOUS SYSTEME DEPUIS RESOURCES/VIEWQ.PY
#from resources.api_views import PaysListAPIView, SousSystemeListAPIView
#LA SUITE DES IMPORTS
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserRegisterSerializer
from .models import CustomUser, DeviceConnectionHistory, FeedbackCorrection
from .models import DeviceMigrationRequest
from rest_framework.permissions import IsAuthenticated

from rest_framework.views import APIView
from abonnement.services import user_abonnement_actif, debiter_credit_abonnement
from .models import DemandeCorrection, SoumissionIA
from .ia_utils import generer_corrige_ia_et_graphique_async
from resources.models import Pays, SousSysteme, Classe, Matiere, TypeExercice,Lecon,Departement
import json
from rest_framework.parsers import MultiPartParser, JSONParser
from django.shortcuts import get_object_or_404
import markdown
import re
from .ia_utils import detect_and_format_math_expressions, generate_corrige_html
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from correction.models import SoumissionIA
from django.http import HttpResponse
from weasyprint import HTML as WeasyHTML
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from .models import SoumissionIA
from .pdf_utils import generer_pdf_corrige
from abonnement.services import user_abonnement_actif, debiter_credit_abonnement
from rest_framework.permissions import AllowAny
from resources.api_views import PaysListAPIView, SousSystemeListAPIView
from correction.models import AppConfig
from rest_framework.response import Response




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
        # Blocage device multi-login :
        if user.device_id and user.device_id != device_id:
            DeviceConnectionHistory.objects.create(user=user, device_id=device_id, successful=False)
            return Response({
                "success": False,
                "error": "Ce compte est déjà utilisé sur un autre appareil. Demandez une migration auprès du support via l’interface d’assistance."
            }, status=403)
        # Première connexion ou device ok :
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
            "code_promo": user.code_promo or ""
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
    parser_classes = [MultiPartParser, JSONParser]

    def post(self, request):
        try:
            # 1) Vérifier l'abonnement actif / crédits restants
            if not user_abonnement_actif(request.user):
                return Response(
                    {"error": "Crédits épuisés ou abonnement expiré. Veuillez recharger votre abonnement."},
                    status=status.HTTP_402_PAYMENT_REQUIRED
                )

            # 2) Débite 1 crédit
            debited = debiter_credit_abonnement(request.user)
            if not debited:
                return Response(
                    {"error": "Impossible de débiter un crédit. Réessayez plus tard."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            # Récupérer les données
            pays_id = request.data.get('pays')
            sous_systeme_id = request.data.get('sous_systeme')
            classe_id = request.data.get('classe')
            matiere_id = request.data.get('matiere')
            type_exercice_id = request.data.get('type_exercice')
            departement_id = request.data.get('departement')
            enonce_texte = request.data.get('enonce_texte', '')
            fichier = request.FILES.get('fichier')

            # Récupérer les leçons sélectionnées
            lecons_ids = request.data.get('lecons_ids', [])
            if isinstance(lecons_ids, str):
                try:
                    lecons_ids = json.loads(lecons_ids)
                except json.JSONDecodeError:
                    lecons_ids = []

            # Validation des données requises
            if not matiere_id or not fichier:
                return Response(
                    {"error": "Matière et fichier sont obligatoires"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Créer la demande
            demande = DemandeCorrection.objects.create(
                user=request.user,
                pays_id=pays_id,
                sous_systeme_id=sous_systeme_id,
                classe_id=classe_id,
                matiere_id=matiere_id,
                departement_id=departement_id,
                type_exercice_id=type_exercice_id,
                fichier=fichier,
                enonce_texte=enonce_texte
            )

            # Ajouter les leçons sélectionnées
            if lecons_ids:
                lecons = Lecon.objects.filter(id__in=lecons_ids)
                demande.lecons.set(lecons)

            # Créer le suivi IA
            soumission = SoumissionIA.objects.create(
                user=request.user,
                demande=demande,
                statut='en_attente'
            )

            # Lancer le traitement async
            from .ia_utils import generer_corrige_ia_et_graphique_async
            generer_corrige_ia_et_graphique_async.delay(demande.id, matiere_id)

            # Débiter le crédit (commenté temporairement)
            debiter_credit_abonnement(request.user)

            return Response({
                "success": True,
                "soumission_id": soumission.id,
                "message": "Exercice soumis avec succès. Traitement en cours..."
            })

        except Exception as e:
            print(f"Erreur lors de la soumission: {e}")
            return Response(
                {"error": f"Erreur serveur: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class StatutSoumissionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, soumission_id):
        try:
            from .ia_utils import detect_and_format_math_expressions, generate_corrige_html

            soumission = SoumissionIA.objects.get(id=soumission_id, user=request.user)
            resultat = soumission.resultat_json or {}

            # Corrigé existant = string, sinon chaîne vide
            corrige_raw = resultat.get('corrige_text') or ""
            print("==== DEBUG: corrige_text avant sanitation ====")
            print(repr(corrige_raw))

            # 🏆 Sanitation ONLY if string is not empty
            if corrige_raw.strip() != "":
                latex_clean = detect_and_format_math_expressions(corrige_raw)
                html_corrige = generate_corrige_html(latex_clean)
                resultat['corrige_text'] = html_corrige
            else:
                print("==== DEBUG: AUCUN corrigé renvoyé ou string vide ====")
                resultat['corrige_text'] = ""  # Renvoyer une chaîne vide

            return Response({
                "statut": soumission.statut,
                "resultat": resultat,
                "date_creation": soumission.date_creation
            })

        except SoumissionIA.DoesNotExist:
            return Response({"error": "Soumission non trouvée"}, status=404)


class DepartementsListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        pays_id = request.query_params.get('pays')
        departements = Departement.objects.all()
        if pays_id:
            departements = departements.filter(pays_id=pays_id)
        data = [{"id": d.id, "nom": d.nom} for d in departements]
        return Response(data)


class ClassesListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        sous_systeme_id = request.query_params.get('sous_systeme')
        classes = Classe.objects.all()
        if sous_systeme_id:
            classes = classes.filter(sous_systeme_id=sous_systeme_id)
        data = [{"id": c.id, "nom": c.nom} for c in classes]
        return Response(data)


class MatieresListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        classe_id = request.query_params.get('classe')
        matieres = Matiere.objects.all()
        if classe_id:
            matieres = matieres.filter(classe_id=classe_id)
        data = [{"id": m.id, "nom": m.nom} for m in matieres]
        return Response(data)


class TypesExerciceListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        departement_id = request.query_params.get('departement')
        types = TypeExercice.objects.all()
        if departement_id:
            types = types.filter(departement_id=departement_id)
        data = [{"id": t.id, "nom": t.nom} for t in types]
        return Response(data)


class LeconsListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        matiere_id = request.query_params.get('matiere')
        lecons = Lecon.objects.all()
        if matiere_id:
            lecons = lecons.filter(matiere_id=matiere_id)
        data = [{"id": l.id, "titre": l.titre} for l in lecons]
        return Response(data)

#vue corrige
class DownloadCorrigeAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, soumission_id):
        try:
            soumission = SoumissionIA.objects.get(id=soumission_id, user=request.user)
            if soumission.statut != 'termine':
                return Response({"error": "Corrigé non prêt"}, status=400)

            pdf_url = soumission.resultat_json.get('pdf_url')
            return Response({"pdf_url": pdf_url})

        except SoumissionIA.DoesNotExist:
            return Response({"error": "Non trouvé"}, status=404)




class HistoriqueCorrectionsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        demandes = DemandeCorrection.objects.filter(user=request.user).order_by('-date_soumission')
        data = []
        for demande in demandes:
            soumission = SoumissionIA.objects.filter(demande=demande).first()
            data.append({
                "id": demande.id,
                "matiere": demande.matiere.nom if demande.matiere else "",
                "date": demande.date_soumission.strftime("%d/%m/%Y %H:%M"),
                "statut": soumission.statut if soumission else "inconnu"
            })
        return Response(data)


class FeedbackAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, correction_id):
        demande = get_object_or_404(DemandeCorrection, id=correction_id, user=request.user)
        note = request.data.get('note')
        comment = request.data.get('comment', '')

        feedback, created = FeedbackCorrection.objects.get_or_create(
            user=request.user,
            correction=demande,
            defaults={'note': note, 'comment': comment}
        )

        if not created:
            feedback.note = note
            feedback.comment = comment
            feedback.save()

        return Response({"success": True, "message": "Feedback enregistré"})



class PartagerCorrigeAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, soumission_id):
        # À implémenter avec un service de partage
        return Response({"success": True, "message": "Fonctionnalité de partage à venir"})


class DebugExtractionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Endpoint pour tester l'extraction"""
        fichier = request.FILES.get('fichier')

        if not fichier:
            return Response({"error": "Aucun fichier"}, status=400)

        from .ia_utils import extraire_texte_fichier
        texte_extraite = extraire_texte_fichier(fichier)

        return Response({
            "success": True,
            "texte_extraite": texte_extraite,
            "type_fichier": fichier.content_type,
            "taille_fichier": fichier.size
        })

#vue pour afficher le corrigé sur une page web dans le mobile ( nouvelle approche)
#mais son url (route) se trouve dans correction/url.py et non api_urls.py car c'est pour le web


class CorrigeHTMLView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, soumission_id):
        soum = get_object_or_404(SoumissionIA, id=soumission_id, user=request.user)

        raw = soum.resultat_json.get("corrige_text") or ""
        latex = detect_and_format_math_expressions(raw)
        html_body = generate_corrige_html(latex)

        return render(request, "correction/corrige_view.html", {
            "titre_corrige": f"Corrigé CIS – Exercice {soum.demande.id}",
            "corrige_html": html_body,
            "soumission_id": soum.id
        })


class CorrigePDFView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, soumission_id):
        soum = get_object_or_404(SoumissionIA, id=soumission_id, user=request.user)

        raw = soum.resultat_json.get("corrige_text") or ""
        latex = detect_and_format_math_expressions(raw)
        html_body = generate_corrige_html(latex)

        context = {
            "titre_corrige": f"Corrigé CIS – Exercice {soum.demande.id}",
            "corrige_html": html_body,
            "soumission_id": soum.id
        }

        pdf_url = generer_pdf_corrige(context, soum.id)

        # Renvoie l'URL du PDF (texte brut)
        return HttpResponse(pdf_url, content_type="text/plain")


 # vue de BLOCAGE PDF ou SOUMISSION coté backend

class AppConfigAPIView(APIView):
    permission_classes = [AllowAny]
    def get(self, request):
        # Un seul objet AppConfig, donc pk=1
        app_config = AppConfig.objects.first()
        if not app_config:
            return Response({"pdf_enabled": True, "correction_enabled": True, "message_bloquant": ""})
        return Response({
            "pdf_enabled": app_config.pdf_enabled,
            "correction_enabled": app_config.correction_enabled,
            "message_bloquant": app_config.message_bloquant,
        })

