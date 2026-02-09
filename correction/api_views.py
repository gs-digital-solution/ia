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
from .ia_utils import generer_corrige_ia_et_graphique_async, is_departement_scientifique
from resources.models import Pays, SousSysteme, Classe, Matiere, TypeExercice,Lecon,Departement
import json
from rest_framework.parsers import MultiPartParser, JSONParser
from django.shortcuts import get_object_or_404
import markdown
import re
from .ia_utils import detect_and_format_math_expressions, generate_corrige_html,format_corrige_pdf_structure
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
import time
from .ia_utils import (
    flatten_multiline_latex_blocks,
    extract_and_process_graphs,
    format_corrige_pdf_structure,
    generer_corrige_exercice_async,

)
from rest_framework.permissions import IsAuthenticated
from .ia_utils import separer_exercices, extraire_texte_fichier
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
import traceback
import tempfile, os
from django.conf import settings
import mimetypes
from rest_framework import generics
from .models import SoumissionIA, CorrigePartiel
from .serializers import CorrigePartielSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from correction.ia_utils import  separer_exercices_avec_titres
from .models import ContactWhatsApp


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
            # → Le débit est différé : il interviendra plus tard, une fois le PDF généré.

            # Récupérer les données
            pays_id = request.data.get('pays')
            sous_systeme_id = request.data.get('sous_systeme')
            classe_id = request.data.get('classe')
            matiere_id = request.data.get('matiere')
            type_exercice_id = request.data.get('type_exercice')
            departement_id = request.data.get('departement')
            fichier = request.FILES.get('fichier')

            # Récupérer les leçons sélectionnées
            lecons_ids = request.data.get('lecons_ids', [])
            if isinstance(lecons_ids, str):
                try:
                    lecons_ids = json.loads(lecons_ids)
                except json.JSONDecodeError:
                    lecons_ids = []

            # Validation des données requises
            if not matiere_id:
                return Response(
                    {"error": "Matière obligatoire"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Vérifier qu'on a un fichier
            if not fichier :
                return Response(
                    {"error": "Fichier ou énoncé texte requis"},
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

            # Lancer le traitement async AVEC DÉCOUPAGE
            from .ia_utils import generer_corrige_ia_et_graphique_async
            generer_corrige_ia_et_graphique_async.delay(demande.id, matiere_id)

            return Response({
                "success": True,
                "soumission_id": soumission.id,
                "message": "Exercice soumis avec succès. Traitement en cours...",
                "info": "Le système détectera automatiquement si un découpage est nécessaire"
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
            soumission = SoumissionIA.objects.get(id=soumission_id, user=request.user)
            resultat   = soumission.resultat_json or {}

            corrige_raw = resultat.get('corrige_text') or ""
            print("==== DEBUG: corrige_text avant sanitation ====")
            print(repr(corrige_raw))

            if corrige_raw.strip():
                # 1) Structuration
                corrige_struct = format_corrige_pdf_structure(corrige_raw)
                # 2) Fusion LaTeX multi-lignes
                corrige_flat   = flatten_multiline_latex_blocks(corrige_struct)
                # 3) Passage MathJax (inline & display)
                latex_clean    = detect_and_format_math_expressions(corrige_flat)
                # 4) Extraction + insertion des graphiques
                corrige_graphs, graphs_data = extract_and_process_graphs(latex_clean)
                # 5) Génération du HTML final
                html_corrige   = generate_corrige_html(corrige_graphs)

                resultat['corrige_text'] = html_corrige
                # (optionnel) renvoyer aussi la liste des graphiques JSON
                resultat['graphiques']   = graphs_data
            else:
                print("==== DEBUG: AUCUN corrigé renvoyé ou string vide ====")
                resultat['corrige_text'] = ""

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




from rest_framework import generics
from .serializers import HistoriqueSerializer
from .models import DemandeCorrection

class HistoriqueCorrectionsAPIView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class   = HistoriqueSerializer

    def get_queryset(self):
        # On précharge les SoumissionIA + CorrigePartiel pour optimiser les requêtes
        return DemandeCorrection.objects.filter(
            user=self.request.user
        ).order_by('-date_soumission').prefetch_related(
            'soumissionia_set__corriges'
        )



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
    permission_classes     = [IsAuthenticated]

    def get(self, request, soumission_id):
        soum = get_object_or_404(SoumissionIA, id=soumission_id, user=request.user)
        raw  = soum.resultat_json.get("corrige_text") or ""

        # Même pipeline que pour StatutSoumissionAPIView
        corrige_struct = format_corrige_pdf_structure(raw)
        corrige_flat   = flatten_multiline_latex_blocks(corrige_struct)
        latex_clean    = detect_and_format_math_expressions(corrige_flat)
        corrige_graphs, _  = extract_and_process_graphs(latex_clean)
        html_body      = generate_corrige_html(corrige_graphs)

        return render(request, "correction/corrige_view.html", {
            "titre_corrige": f"Corrigé CIS – Exercice {soum.demande.id}",
            "corrige_html": html_body,
            "soumission_id": soum.id
        })



class CorrigePDFView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes     = [IsAuthenticated]

    def get(self, request, soumission_id):
        soum = get_object_or_404(SoumissionIA, id=soumission_id, user=request.user)
        raw  = soum.resultat_json.get("corrige_text") or ""

        # Pipeline identique avant génération PDF
        corrige_struct = format_corrige_pdf_structure(raw)
        corrige_flat   = flatten_multiline_latex_blocks(corrige_struct)
        latex_clean    = detect_and_format_math_expressions(corrige_flat)
        corrige_graphs, _  = extract_and_process_graphs(latex_clean)
        html_body      = generate_corrige_html(corrige_graphs)

        context = {
            "titre_corrige": f"Corrigé CIS – SUJET {soum.demande.id} corrigé par CIS IA sur playstore",
            "corrige_html": html_body,
            "soumission_id": soum.id
        }

        pdf_url = generer_pdf_corrige(context, soum.id)
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



#VUE POUR LISTE D'EXERCICES A CORRIGER
class SplitExercisesAPIView(APIView):
    """
    POST /api/split/
    - Crée une DemandeCorrection
    - Stocke les exercices avec leurs titres complets dans exercices_data
    - Retourne { demande_id: ..., exercices: [...] } avec vrais titres
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, JSONParser]

    def post(self, request):
        user = request.user

        # 1) Récupérer les IDs passés
        pays_id = request.data.get('pays')
        sous_id = request.data.get('sous_systeme')
        departement_id = request.data.get('departement')
        classe_id = request.data.get('classe')
        matiere_id = request.data.get('matiere')
        type_exo_id = request.data.get('type_exercice')
        lecons_ids = request.data.get('lecons_ids')

        # 2) Récupérer le fichier d'énoncé
        fichier = request.FILES.get('fichier')
        if not fichier:
            return Response(
                {"error": "Le fichier d'énoncé est requis."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2b) VÉRIFICATION TAILLE FICHIER (1 Mo max)
        if fichier.size > 1048576:
            return Response(
                {"error": "Le fichier ne doit pas dépasser 1 Mo."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2c) VÉRIFICATION FORMAT
        ext = os.path.splitext(fichier.name)[1].lower()
        allowed_ext = ['.pdf', '.png', '.jpg', '.jpeg']
        if ext not in allowed_ext:
            return Response(
                {"error": f"Format {ext} non supporté. Utilisez PDF, PNG, JPG ou JPEG."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 3) Créer la demande
        demande = DemandeCorrection.objects.create(
            user=user,
            pays_id=pays_id,
            sous_systeme_id=sous_id,
            departement_id=departement_id,
            classe_id=classe_id,
            matiere_id=matiere_id,
            type_exercice_id=type_exo_id,
            fichier=fichier
        )

        # 3b) Lier les leçons si présentes
        if lecons_ids:
            try:
                ids = json.loads(lecons_ids) if isinstance(lecons_ids, str) else lecons_ids
                demande.lecons.set(ids)
            except Exception:
                pass

        # 4) RÉCUPÉRER LE DÉPARTEMENT POUR L'EXTRACTION
        departement = None
        if departement_id:
            departement = Departement.objects.filter(pk=departement_id).first()

        if departement:
            print(f"📁 Département pour extraction: {departement.nom}")
            print(f"   Scientifique? {is_departement_scientifique(departement)}")

        # 5) Extraire le texte avec la méthode adaptée
        texte = extraire_texte_fichier(fichier, departement)  # ← DEPARTEMENT PASSÉ ICI !

        if not texte:
            return Response(
                {"error": "Impossible d'extraire le texte de la demande."},
                status=status.HTTP_400_BAD_REQUEST
            )

        print(f"✅ [SplitExercises] Texte extrait: {len(texte)} caractères")
        print(
            f"   Méthode: {'Mathpix' if departement and is_departement_scientifique(departement) else 'OCR standard'}")

        # 6) Séparation + validation index
        exercices_detaillees = separer_exercices_avec_titres(texte)

        # 7) Construire la liste JSON complète pour stockage AVEC CONTENU COMPLET
        exercices_complets = []
        for idx, ex in enumerate(exercices_detaillees):
            titre_complet = ex.get('titre_complet', ex.get('titre', f"Exercice {idx + 1}"))
            contenu_complet = ex.get('contenu', '')  # ← CONTENU COMPLET

            # Nettoyer le titre pour l'affichage
            titre_affichage = titre_complet
            if len(titre_affichage) > 80:
                titre_affichage = titre_affichage[:77] + "..."

            # Extraire un extrait (premières lignes) pour l'affichage rapide
            lignes = contenu_complet.strip().split('\n')
            extrait_lignes = []
            for line in lignes[:3]:  # Prendre jusqu'à 3 premières lignes non vides
                line_stripped = line.strip()
                if line_stripped and len(line_stripped) < 100:
                    extrait_lignes.append(line_stripped)

            extrait = ' / '.join(extrait_lignes) if extrait_lignes else contenu_complet.strip()[:150]
            if len(extrait) > 150:
                extrait = extrait[:147] + "..."

            # ✅ STOCKER LE CONTENU COMPLET CETTE FOIS
            exercices_complets.append({
                "index": idx,
                "titre": titre_affichage,
                "titre_complet": titre_complet,
                "extrait": extrait,
                "contenu_complet": contenu_complet,  # ← NOUVEAU : CONTENU COMPLET
                "longueur_contenu": len(contenu_complet)
            })

        # 8) Stocker les exercices COMPLETS dans la demande
        demande.exercices_data = json.dumps(exercices_complets, ensure_ascii=False)
        demande.save()

        print(f"✅ [SplitExercises] {len(exercices_complets)} exercices stockés avec contenu complet")

        # 9) Construire la réponse pour le frontend (extraits seulement)
        exercices_reponse = []
        for ex in exercices_complets:
            exercices_reponse.append({
                "index": ex["index"],
                "titre": ex["titre"],
                "extrait": ex["extrait"]
            })

        # 10) Répondre
        return Response({
            "demande_id": demande.id,
            "exercices": exercices_reponse,
            "nom_fichier": demande.nom_fichier or os.path.basename(fichier.name),
            "matiere": demande.matiere.nom if demande.matiere else "Non spécifiée",
            "info": f"{len(exercices_complets)} exercices détectés, contenu complet stocké"
        })

#VUE PARTIELLE DES EXERCICES
class PartialCorrectionAPIView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        try:
            user = request.user
            # ===== AJOUT: VÉRIFICATION CRÉDITS AVANT DE COMMENCER =====
            if not user_abonnement_actif(user):
                return Response(
                    {"error": "Crédits épuisés ou abonnement expiré. Veuillez recharger votre abonnement."},
                    status=status.HTTP_402_PAYMENT_REQUIRED
                )
            # ===========================================================
            demande_id = request.data.get("demande_id")
            idx = request.data.get("index")

            # 1) Validation minimale
            if demande_id is None or idx is None:
                return Response(
                    {"error": "demande_id et index sont requis"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            # Conversion et revalidation
            try:
                idx = int(idx)
            except:
                return Response(
                    {"error": "index doit être un entier"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 2) Vérifier la demande
            demande = get_object_or_404(DemandeCorrection, id=demande_id, user=user)

            # 3) OPTIMISATION : Vérifier si le contenu est déjà dans exercices_data
            fragment_trouve = False

            if demande.exercices_data:
                try:
                    exercices_list = json.loads(demande.exercices_data)
                    for ex in exercices_list:
                        if ex.get('index') == idx:
                            # Vérifier qu'on a du contenu complet
                            if ex.get('contenu_complet') and len(ex['contenu_complet']) > 50:
                                fragment_trouve = True
                                print(f"✅ [PartialCorrection] Contenu trouvé dans exercices_data pour index {idx}")
                                break
                except json.JSONDecodeError:
                    print(f"⚠️ [PartialCorrection] JSON invalide dans exercices_data")

            # 4) Si pas de contenu stocké, vérifier qu'on a un fichier
            if not fragment_trouve and not demande.fichier:
                return Response(
                    {"error": "Aucun contenu disponible pour cet exercice."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 5) Création de la soumission
            soumission = SoumissionIA.objects.create(
                user=user,
                demande=demande,
                statut='en_attente',
                progression=0,
                exercice_index=idx
            )

            # 6) Information de debug
            print(f"✅ [PartialCorrection] Soumission {soumission.id} créée pour exercice {idx}")
            print(f"   - Contenu pré-stocké: {'OUI' if fragment_trouve else 'NON (nécessitera extraction)'}")
            print(f"   - Fichier disponible: {'OUI' if demande.fichier else 'NON'}")

            # 7) Lancement asynchrone
            generer_corrige_exercice_async.delay(soumission.id)

            # 8) Réponse
            return Response({
                "success": True,
                "soumission_exercice_id": soumission.id,
                "message": "Exercice envoyé au traitement.",
                "optimisation": "contenu_pré_stocké" if fragment_trouve else "nécessite_extraction"
            }, status=status.HTTP_202_ACCEPTED)

        except Exception as e:
            # Affiche la stack complète dans les logs
            traceback.print_exc()
            # Renvoie un message minimal au front
            return Response(
                {"error": f"Erreur interne: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

#Lister les corrigés partiels d’une soumission
class CorrigesListAPIView(generics.ListAPIView):
    serializer_class = CorrigePartielSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        soumission_id = self.kwargs['soumission_id']
        # On s’assure que l’utilisateur y a bien droit
        soumission = get_object_or_404(SoumissionIA, id=soumission_id, user=self.request.user)
        return soumission.corriges.all()


#  VUE POUR TELECHARGER LES PDF PARTIELS
class DownloadPartialCorrigeAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, corrige_id):
        corrige = get_object_or_404(
            CorrigePartiel,
            id=corrige_id,
            soumission__user=request.user
        )
        if not corrige.fichier_pdf:
            return Response({"error": "Pas de fichier PDF disponible."}, status=404)
        url = request.build_absolute_uri(corrige.fichier_pdf.url)
        return Response({"pdf_url": url})


# Vue pour récupérer le lien WhatsApp
class ContactWhatsAppAPIView(APIView):
    """
    API pour récupérer le lien WhatsApp de contact
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            # Récupérer le premier contact WhatsApp actif
            contact = ContactWhatsApp.objects.filter(actif=True).first()

            if not contact:
                return Response({
                    "success": False,
                    "error": "Aucun contact WhatsApp configuré"
                }, status=status.HTTP_404_NOT_FOUND)

            # Construire le lien avec le message d'accueil
            lien_complet = contact.lien_whatsapp
            # S'assurer que le lien contient le paramètre text
            if "?text=" not in lien_complet:
                if "?" in lien_complet:
                    lien_complet += f"&text={contact.message_accueil}"
                else:
                    lien_complet += f"?text={contact.message_accueil}"

            return Response({
                "success": True,
                "lien_whatsapp": lien_complet,
                "message_accueil": contact.message_accueil,
                "actif": contact.actif
            })

        except Exception as e:
            print(f"❌ Erreur récupération contact WhatsApp: {e}")
            return Response({
                "success": False,
                "error": "Erreur lors de la récupération du contact"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

