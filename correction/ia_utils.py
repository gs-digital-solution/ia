import requests
import os
import tempfile
import json
import re
import numpy as np
import cv2
from pdf2image import convert_from_path
import matplotlib
import openai
from .pdf_utils import generer_pdf_corrige
from datetime import datetime
import logging
import camelot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from django.conf import settings
from django.utils.safestring import mark_safe
from celery import shared_task
from PIL import Image
import base64
from resources.models import PromptIA,Matiere
from .pdf_utils import generer_pdf_corrige
from .models import SoumissionIA
from resources.models import Matiere
from abonnement.services import debiter_credit_abonnement
from .models import CorrigePartiel
from django.core.files import File
import time
from datetime import datetime
#from .tasks import generer_un_exercice
#from celery import group
import logging
# Logger dédié
logger = logging.getLogger(__name__)

mathpix_logger = logging.getLogger('mathpix')
def log_extraction_method(demande, method, success=True):
    """Journaliser la méthode d'extraction utilisée"""
    if demande:
        dept = demande.departement.nom if demande.departement else "inconnu"
        mathpix_logger.info(
            f"📊 Extraction - Département: {dept}, "
            f"Méthode: {method}, Succès: {success}"
        )


def extraire_avec_mathpix(fichier_path: str) -> dict:
    """
    Extraction avec Mathpix – gère les images et les PDF multi-pages.
    Pour les PDF, convertit et traite TOUTES les pages, puis concatène les résultats.
    Retourne le texte avec les formules formatées pour MathJax.
    """
    headers = {
        "app_id": os.getenv("MATHPIX_APP_ID"),
        "app_key": os.getenv("MATHPIX_APP_KEY"),
        "Content-type": "application/json"
    }

    ext = os.path.splitext(fichier_path)[1].lower()
    logger.info(f"📁 Fichier reçu par Mathpix: {ext}")

    temp_files = []
    all_text_parts = []
    all_latex_blocks = []

    try:
        # === 1. GESTION DES PDF (conversion de TOUTES les pages) ===
        if ext == '.pdf':
            logger.info("📄 PDF détecté, conversion de toutes les pages en images...")
            from pdf2image import convert_from_path

            # Convertir TOUTES les pages du PDF
            images = convert_from_path(
                fichier_path,
                dpi=300  # Bonne résolution pour l'OCR
            )

            logger.info(f"   {len(images)} page(s) trouvée(s)")

            # Traiter chaque page une par une
            for page_num, image in enumerate(images, 1):
                logger.info(f"   🔄 Traitement page {page_num}/{len(images)}...")

                # Sauvegarder temporairement la page
                temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_img.close()
                image.save(temp_img.name, 'PNG')
                temp_files.append(temp_img.name)

                # Lire l'image
                with open(temp_img.name, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()

                # Appel à Mathpix pour cette page
                data = {
                    "src": f"data:image/jpeg;base64,{image_data}",
                    "formats": ["text", "latex_styled"],
                    "ocr": ["math", "text"],
                    "skip_recrop": False,
                    "math_inline_delimiters": ["$", "$"],
                    "rm_spaces": True,
                    "format": "text"
                }

                try:
                    response = requests.post(
                        os.getenv("MATHPIX_API_URL", "https://api.mathpix.com/v3/text"),
                        headers=headers,
                        data=json.dumps(data),
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        page_texte = result.get("text", "")
                        page_latex = result.get("latex_styled", [])

                        # Ajouter un séparateur de page pour la lisibilité
                        if page_texte:
                            all_text_parts.append(f"[Page {page_num}]\n{page_texte}")
                        else:
                            all_text_parts.append(f"[Page {page_num} - vide]")

                        all_latex_blocks.extend(page_latex)

                        logger.info(f"   ✅ Page {page_num}: {len(page_texte)} caractères")
                    else:
                        logger.warning(f"   ⚠️ Page {page_num}: erreur {response.status_code}")
                        all_text_parts.append(f"[Page {page_num} - erreur]")

                except Exception as e:
                    logger.warning(f"   ⚠️ Page {page_num}: exception {e}")
                    all_text_parts.append(f"[Page {page_num} - exception]")

                # Petite pause pour éviter de surcharger l'API
                time.sleep(0.5)

        # === 2. GESTION DES IMAGES (une seule page) ===
        else:
            logger.info("🖼️ Image détectée, traitement direct...")
            with open(fichier_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()

            data = {
                "src": f"data:image/jpeg;base64,{image_data}",
                "formats": ["text", "latex_styled"],
                "ocr": ["math", "text"],
                "skip_recrop": False,
                "math_inline_delimiters": ["$", "$"],
                "rm_spaces": True,
                "format": "text"
            }

            response = requests.post(
                os.getenv("MATHPIX_API_URL", "https://api.mathpix.com/v3/text"),
                headers=headers,
                data=json.dumps(data),
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                all_text_parts = [result.get("text", "")]
                all_latex_blocks = result.get("latex_styled", [])
                logger.info(f"✅ Image traitée: {len(all_text_parts[0])} caractères")
            else:
                logger.error(f"❌ Mathpix error {response.status_code}")
                return {"text": "", "latex_blocks": [], "source": "error"}

        # === 3. CONCATÉNATION ET FORMATAGE FINAL ===
        texte_complet = "\n\n".join(all_text_parts)

        # Formatage global pour MathJax
        texte_complet = re.sub(
            r'\$\$(.*?)\$\$',
            lambda m: '\\[' + m.group(1).strip() + '\\]',
            texte_complet,
            flags=re.DOTALL
        )

        texte_complet = re.sub(
            r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)',
            lambda m: '\\(' + m.group(1).strip() + '\\)',
            texte_complet,
            flags=re.DOTALL
        )

        logger.info(
            f"✅ Extraction terminée: {len(texte_complet)} caractères au total, {len(all_latex_blocks)} blocs LaTeX")

        return {
            "text": texte_complet,
            "latex_blocks": all_latex_blocks,
            "source": "mathpix",
            "pages_traitees": len(images) if ext == '.pdf' else 1
        }

    except Exception as e:
        logger.error(f"❌ Mathpix exception: {e}")
        import traceback
        traceback.print_exc()
        return {"text": "", "latex_blocks": [], "source": "error"}

    finally:
        # === 4. NETTOYAGE ===
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"🧹 Fichier temporaire supprimé: {temp_file}")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de supprimer {temp_file}: {e}")

def preprocess_image_for_ocr(pil_image):
    """
    Convertit une PIL.Image en image binaire nettoyée pour Tesseract.
    """
    # PIL → CV2 BGR
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # seuillage adaptatif
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    # ouverture pour nettoyer le petit bruit
    kernel = np.ones((1,1), np.uint8)
    clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    return clean

# Cache en mémoire des PromptIA pour éviter les hits répétés en BDD
_PROMPTIA_CACHE = {}

# ========== BLIP LAZY-LOADER ==========
_blip_model = None
_blip_processor = None
def get_blip_model():
    """
    Charge le modèle BLIP au premier appel (lazy load).
    """
    global _blip_model, _blip_processor
    if _blip_model is None:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device).eval()
        logger.info("🖼️ BLIP chargé sur %s", device)
    return _blip_processor, _blip_model


DEPARTEMENTS_SCIENTIFIQUES = [
    'mathématiques', 'physique', 'chimie', 'biologie',
    'svt', 'sciences', 'informatique', 'anglais'
    # Ajouter des variantes
    'maths', 'mathematiques', 'math', 'physique-chimie',
    'science', 'scientifique', 'biologie-géologie'
]


def is_departement_scientifique(departement):
    """
    Renvoie True si le département fait partie des filières scientifiques.
    Version robuste avec plusieurs variantes.
    """
    if not departement or not departement.nom:
        return False

    dep_name = departement.nom.strip().lower()

    # Liste étendue de termes scientifiques
    scientific_terms = [
        'math', 'physique', 'chimie', 'biologie', 'svt',
        'science', 'informatique', 'technologie', 'géologie',
        'astronomie', 'écologie', 'génétique', 'électricité',
        'mécanique', 'optique', 'thermodynamique', 'statistique',
        'algèbre', 'géométrie', 'analyse', 'calcul', 'numérique'
    ]

    # Vérification simple
    for term in scientific_terms:
        if term in dep_name:
            print(f"✅ Département '{dep_name}' reconnu comme scientifique (contient '{term}')")
            return True

    # Vérification spécifique pour les débuts de mots
    for term in DEPARTEMENTS_SCIENTIFIQUES:
        if dep_name.startswith(term):
            print(f"✅ Département '{dep_name}' commence par '{term}' → scientifique")
            return True

    print(f"❌ Département '{dep_name}' non reconnu comme scientifique")
    return False


# ── CODE D'EXTRACTION DU PROMPT LE PLUS SPECIFIQUE POSSIBLE ────────────────────
def get_best_promptia(demande):
    """
    Retourne le PromptIA le plus spécifique pour la demande, ou None.
    Ne fait jamais filter({}) qui retomberait sur le 1er prompt anglais.
    Fallback progressif, puis prompt par défaut si rien trouvé.
    """
    # 1) Construire le filtre initial
    filtra = {
        'pays': demande.pays,
        'sous_systeme': demande.sous_systeme,
        'classe': demande.classe,
        'matiere': demande.matiere,
        'departement': demande.departement,
        'type_exercice': demande.type_exercice,
    }
    # Ne garder que les clés non-nulles
    filtra = {k: v for k, v in filtra.items() if v is not None}

    # 2) Si on a au moins un critère, tenter la recherche exacte
    if filtra:
        qs = PromptIA.objects.filter(**filtra)
        if qs.exists():
            return qs.first()

        # 2b) Fallback progressif en retirant un champ à la fois
        for champ in ['type_exercice', 'departement', 'classe', 'sous_systeme', 'pays']:
            if champ in filtra:
                filtra2 = dict(filtra)
                filtra2.pop(champ)
                if filtra2:
                    qs2 = PromptIA.objects.filter(**filtra2)
                    if qs2.exists():
                        return qs2.first()

        # 2c) Fallback par matière seule si matière faisait partie du filtre
        if 'matiere' in filtra:
            qs3 = PromptIA.objects.filter(matiere=demande.matiere)
            if qs3.exists():
                return qs3.first()

    # 3) Aucune correspondance : retomber sur DEFAULT_SYSTEM_PROMPT
    return None


# ── CONFIGURATION DEEPSEEK AVEC VISION ────────────────────
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.api_base = "https://api.deepseek.com"

# ── MODÈLE POUR LA VISION ────────────────────────────────
# deepseek-chat a les capacités vision quand on envoie des images
DEEPSEEK_VISION_MODEL = "deepseek-reasoner"


# ─── NEW ─── appel multimodal à DeepSeek-V3 pour PDF / images ────
# ── CORRIGÉ : Appel multimodal à DeepSeek pour PDF/images ────
def call_deepseek_vision(path_fichier: str) -> dict:
    """
    Envoie un PDF ou une image à DeepSeek - Version corrigée pour l'API DeepSeek.
    """
    system_prompt = r"""
    Tu es un expert en schémas et documents scientifiques.
    Prends cette image ou ce PDF (base64) et renvoie **SEULEMENT** un JSON structuré :
    {
      "text": "<le texte complet>",
      "latex_blocks": ["…","…"],
      "captions": ["légende du schéma", …],
      "graphs": [ { … données graphiques … } ],
      "angles": [ {"valeur":30,"unité":"°","coord":[x,y]}, … ],
      "numbers": [ {"valeur":9.81,"unité":"m/s²","coord":[x,y]}, … ]
    }
    Ne renvoie aucun texte hors de ce JSON.
    """
    try:
        # Encoder le fichier en base64
        with open(path_fichier, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")

        # ✅ CORRECTION : Format DeepSeek compatible
        message_content = f"""
        [image]{data_b64}[/image]

        Extrait le texte, les formules LaTeX et les légendes de ce document.
        """

        response = openai.ChatCompletion.create(
            model=DEEPSEEK_VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=8000
        )

        content = response.choices[0].message.content
        return content if isinstance(content, dict) else json.loads(content)

    except Exception as e:
        print(f"❌ Erreur call_deepseek_vision: {e}")
        return {"text": "", "latex_blocks": [], "captions": [], "graphs": []}


# ============== NOUVELLE FONCTION: DeepSeek Vision Améliorée avec extraction structurée ==============
def call_deepseek_vision_ameliore(path_fichier: str, demande=None) -> dict:
    """
    Appel DeepSeek amélioré avec timeout long (120s) et redimensionnement automatique des images.
    Version optimisée pour gérer les images volumineuses et les timeouts.
    """
    logger.info(f"🔄 Appel DeepSeek Vision Amélioré pour {path_fichier}")

    # Vérification clé API
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("❌ DEEPSEEK_API_KEY non configurée")
        return {"exercices": [], "texte_complet": "", "elements_visuels": []}

    # Vérification fichier
    if not os.path.exists(path_fichier):
        logger.error(f"❌ Fichier non trouvé: {path_fichier}")
        return {"exercices": [], "texte_complet": "", "elements_visuels": []}

    # Taille du fichier
    file_size = os.path.getsize(path_fichier)
    logger.info(f"📁 Taille fichier originale: {file_size} octets ({file_size/1024:.1f} Ko)")

    system_prompt = """
    Tu es un expert en reconnaissance de textes et schémas dans des documents scolaires.

    INSTRUCTIONS ABSOLUES:
    1. Ce document est un SUJET D'EXAMEN. Il contient du texte et des schémas.
    2. Tu dois EXTRAIRE le texte EXACTEMENT comme il apparaît, sans modification, sans réécriture.
    3. Tu dois IDENTIFIER la structure du document (exercices, parties).
    4. Tu dois DÉCRIRE les schémas en détail.

    RÈGLE D'OR: Ne réécris PAS l'énoncé. Recopie-le mot pour mot, avec les mêmes phrases, les mêmes mots.

    EXEMPLE:
    Si le texte dit "Dans la cour de récréation, l'élève Mbonto se vante", tu dois écrire exactement cela.
    Tu ne dois PAS écrire "Un élève se vante dans la cour".

    POUR LES SCHÉMAS:
    - Décris leur type (circuit, figure, graphique, plan incliné, etc.)
    - Décris les éléments visibles et leur position
    - Décris les relations entre éléments
    - Extrais toutes les valeurs numériques (angles, longueurs, tensions)

    RENVOIE UNIQUEMENT CE JSON:
    {
      "exercices": [
        {
          "titre": "titre exact de l'exercice",
          "texte": "texte exact recopié sans modification",
          "schemas": [
            {
              "type_schema": "type de schéma",
              "description": "description détaillée",
              "elements": ["élément1", "élément2"],
              "relations": "relations entre éléments",
              "valeurs": {"angle": "30°", "longueur": "70cm"}
            }
          ],
          "formules": ["$formule1$", "$formule2$"]
        }
      ]
    }
    """

    try:
        # ========== REDIMENSIONNEMENT INTELLIGENT DE L'IMAGE ==========
        from PIL import Image
        import io

        logger.info("📖 Lecture et optimisation de l'image...")

        # Ouvrir l'image avec PIL
        img = Image.open(path_fichier)

        # Log des dimensions originales
        original_width, original_height = img.size
        logger.info(f"📐 Dimensions originales: {original_width}x{original_height}")

        # Redimensionner si trop grande (max 1200px de côté)
        max_dimension = 1200
        if original_width > max_dimension or original_height > max_dimension:
            # Calculer le ratio de redimensionnement
            ratio = min(max_dimension / original_width, max_dimension / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)

            # Redimensionner avec conservation de la qualité
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logger.info(f"📐 Image redimensionnée: {new_width}x{new_height} (ratio: {ratio:.2f})")

        # Convertir en RGB si nécessaire (pour les PNG avec transparence)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Sauvegarder en JPEG avec compression optimale
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        compressed_size = len(buffer.getvalue())
        logger.info(f"📦 Taille après compression: {compressed_size} octets ({compressed_size/1024:.1f} Ko)")

        # Encodage en base64
        data_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.info(f"🔐 Base64: {len(data_b64)} caractères ({len(data_b64)/1024:.1f} Ko)")

        # Vérification taille base64 (limite DeepSeek ~500Ko)
        if len(data_b64) > 600000:  # ~450 Ko après décodage
            logger.warning(f"⚠️ Image encore trop grande ({len(data_b64)/1024:.1f} Ko), compression plus forte...")

            # Recompression avec qualité plus faible
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=60, optimize=True)
            data_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.info(f"📦 Après compression renforcée: {len(data_b64)/1024:.1f} Ko")

        # Construction du message
        message_content = f"[image]{data_b64}[/image]\n\nExtrais le texte et les exercices exactement comme dans l'image."

        # Appel API avec timeout long
        logger.info("📡 Envoi requête à DeepSeek (timeout 120s)...")

        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 8000  # Augmenté pour les descriptions détaillées
        }

        logger.info(f"📤 Taille payload: {len(str(payload))/1024:.1f} Ko")

        # Timeout long (120 secondes)
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,  # 2 minutes
            stream=False
        )

        logger.info(f"📡 Réponse reçue: status {response.status_code}")

        if response.status_code != 200:
            logger.error(f"❌ Erreur HTTP {response.status_code}: {response.text[:500]}")
            return {"exercices": [], "texte_complet": "", "elements_visuels": []}

        result = response.json()

        # Vérification de la structure de la réponse
        if 'choices' not in result or not result['choices']:
            logger.error(f"❌ Structure réponse invalide: {result}")
            return {"exercices": [], "texte_complet": "", "elements_visuels": []}

        content = result['choices'][0]['message']['content']
        logger.info(f"📦 Réponse brute ({len(content)} caractères): {content[:300]}...")

        # Nettoyage de la réponse (enlever les markdown json éventuels)
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'\s*```', '', content)
        content = content.strip()

        # Parser le JSON
        try:
            resultat = json.loads(content)

            # Validation de la structure
            if "exercices" not in resultat:
                logger.warning("⚠️ Structure JSON incorrecte (pas de clé 'exercices')")
                # Tentative de correction
                if isinstance(resultat, list):
                    resultat = {"exercices": resultat}
                elif isinstance(resultat, dict) and len(resultat) == 1:
                    # Prendre la première clé comme exercices
                    first_key = list(resultat.keys())[0]
                    resultat = {"exercices": resultat[first_key]}

            nb_exercices = len(resultat.get('exercices', []))
            logger.info(f"✅ Parsing réussi: {nb_exercices} exercices")

            # Log des schémas détectés
            if nb_exercices > 0:
                for i, ex in enumerate(resultat['exercices']):
                    nb_schemas = len(ex.get('schemas', []))
                    logger.info(f"   Exercice {i+1}: {nb_schemas} schéma(s)")

            return resultat

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON invalide: {e}")
            logger.error(f"Contenu: {content[:500]}")

            # Tentative de récupération avec regex
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    resultat = json.loads(json_match.group())
                    logger.info(f"✅ JSON récupéré par regex: {len(resultat.get('exercices', []))} exercices")
                    return resultat
                except:
                    pass

            return {"exercices": [], "texte_complet": "", "elements_visuels": []}

    except requests.exceptions.Timeout:
        logger.error("❌ Timeout DeepSeek (120s dépassé)")
        return {"exercices": [], "texte_complet": "", "elements_visuels": []}

    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Erreur connexion DeepSeek: {e}")
        return {"exercices": [], "texte_complet": "", "elements_visuels": []}

    except Exception as e:
        logger.error(f"❌ Erreur DeepSeek: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"exercices": [], "texte_complet": "", "elements_visuels": []}

# ── NOUVELLE FONCTION : Analyse scientifique avancée ────

def analyser_document_scientifique(fichier_path: str, demande=None) -> dict:
    """
    Analyse scientifique avancée - Version DeepSeek First
    Pour les départements scientifiques : DeepSeek Vision (texte + schémas)
    Fallback : Mathpix (si DeepSeek échoue) ou OCR standard
    """
    logger.info(f"🔍 Début analyse scientifique pour {fichier_path}")

    # 1) DÉTECTION DU DÉPARTEMENT POUR CHOIX DE LA MÉTHODE
    use_deepseek = False
    dept_nom = "inconnu"

    if demande and demande.departement:
        dept_nom = demande.departement.nom
        use_deepseek = is_departement_scientifique(demande.departement)
        logger.info(f"📊 Département '{dept_nom}' → DeepSeek = {use_deepseek}")

    # 2) PRIORITÉ: DEEPSEEK POUR DÉPARTEMENTS SCIENTIFIQUES
    if use_deepseek:
        logger.info("🧠 Extraction avec DeepSeek Vision (département scientifique)")

        try:
            resultat_deepseek = call_deepseek_vision_ameliore(fichier_path, demande)

            # Vérifier que le résultat est utilisable
            texte = resultat_deepseek.get("texte_complet", "")
            if texte and len(texte) > 100:
                logger.info(f"✅ DeepSeek réussi: {len(texte)} caractères, "
                            f"{len(resultat_deepseek.get('exercices', []))} exercices, "
                            f"{len(resultat_deepseek.get('elements_visuels', []))} schémas")

                return {
                    "texte_complet": texte,
                    "elements_visuels": resultat_deepseek.get("elements_visuels", []),
                    "formules_latex": resultat_deepseek.get("latex_blocks", []),
                    "graphs": [],  # Sera extrait des exercices si besoin
                    "angles": [],
                    "numbers": [],
                    "structure_exercices": resultat_deepseek.get("exercices", []),
                    "source_extraction": "deepseek",
                    "departement": dept_nom,
                    "exercices_struct": resultat_deepseek.get("exercices", [])  # NOUVEAU: structure complète
                }
            else:
                logger.warning("⚠️ DeepSeek échec ou résultat trop court (<100 chars), fallback Mathpix")
                use_deepseek = False

        except Exception as e:
            logger.error(f"❌ DeepSeek exception: {e}")
            import traceback
            traceback.print_exc()
            use_deepseek = False

    # 3) FALLBACK 1: MATHPIX (si DeepSeek a échoué mais que Mathpix est configuré)
    if not use_deepseek and os.getenv("MATHPIX_APP_ID") and os.getenv("MATHPIX_APP_KEY"):
        logger.info("🧮 Fallback avec Mathpix")

        resultat_mathpix = extraire_avec_mathpix(fichier_path)

        if resultat_mathpix.get("text") and len(resultat_mathpix["text"]) > 100:
            logger.info(f"✅ Mathpix réussi: {len(resultat_mathpix['text'])} caractères")

            return {
                "texte_complet": resultat_mathpix["text"],
                "elements_visuels": [],
                "formules_latex": resultat_mathpix.get("latex_blocks", []),
                "graphs": [],
                "angles": [],
                "numbers": [],
                "structure_exercices": [],
                "source_extraction": "mathpix",
                "departement": dept_nom,
                "exercices_struct": []  # Pas de structure d'exercices
            }
        else:
            logger.warning("⚠️ Mathpix échec, fallback standard")

    # 4) FALLBACK 2: ANALYSE STANDARD (OCR uniquement)
    logger.info("🔤 Fallback final: OCR standard")

    # Code OCR standard existant (à garder tel quel)
    config_tesseract = r'--oem 3 --psm 6 -l fra+eng+digits'
    texte_ocr = ""

    try:
        if fichier_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(fichier_path)
            clean = preprocess_image_for_ocr(img)
            texte_ocr = pytesseract.image_to_string(clean, config=config_tesseract)
            logger.info(f"    ✓ OCR image brut extrait {len(texte_ocr)} caractères")

        elif fichier_path.lower().endswith('.pdf'):
            texte_ocr = extraire_texte_pdf(fichier_path)
            logger.info(f"    ✓ PDFMiner extrait {len(texte_ocr)} caractères")

            if len(texte_ocr) < 50:
                logger.warning("    ⚠️ OCR PDFMiner trop court, fallback page à page")
                pages = convert_from_path(fichier_path, dpi=300)
                txts = []
                for page in pages:
                    clean = preprocess_image_for_ocr(page)
                    txts.append(pytesseract.image_to_string(clean, config=config_tesseract))
                texte_ocr = "\n".join(txts)
                logger.info(f"    ✓ fallback OCR pages donne {len(texte_ocr)} caractères")
    except Exception as e:
        logger.error(f"❌ Erreur pendant OCR/PDF: {e}")

    return {
        "texte_complet": texte_ocr,
        "elements_visuels": [],
        "formules_latex": [],
        "graphs": [],
        "angles": [],
        "numbers": [],
        "structure_exercices": [],
        "source_extraction": "fallback_ocr",
        "departement": dept_nom,
        "exercices_struct": []
    }


def extraire_texte_robuste(fichier_path: str) -> str:
    """
    Extraction simple : OCR direct → Analyse IA
    """
    print("🔄 Extraction simple...")

    # Juste utiliser l'analyse scientifique directe
    try:
        analyse = analyser_document_scientifique(fichier_path)
        texte = analyse.get("texte_complet", "")
        if texte and len(texte) > 50:
            print("✅ Extraction réussie")
            return texte
        else:
            print("❌ Texte trop court, utilisation fallback OCR")
            return texte
    except Exception as e:
        print(f"❌ Extraction échouée: {e}")
        return ""


def debug_ocr(fichier_path: str):
    """
    Debug simple de l'OCR
    """
    try:
        if fichier_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(fichier_path)
            custom_config = r'--oem 3 --psm 6 -l fra+eng'
            texte = pytesseract.image_to_string(image, config=custom_config)
            print("🔍 DEBUG OCR - Texte brut:")
            print(texte[:500])
            print(f"Longueur: {len(texte)} caractères")
            return texte
    except Exception as e:
        print(f"❌ DEBUG OCR échoué: {e}")
    return ""
# ========== EXTRAIRE LES BLOCS JSON POUR LES GRAPHIQUES ==========
def extract_json_blocks(text: str):
    """Extrait les blocs JSON pour les graphiques"""
    decoder = json.JSONDecoder()
    idx = 0
    blocks = []

    while True:
        # Cherche le début d'un bloc JSON (après ```json ou {)
        start = text.find('{', idx)
        if start == -1:
            break

        try:
            # Vérifie si c'est un bloc graphique
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and 'graphique' in obj:
                blocks.append((obj, start, start + end))
            idx = start + end
        except ValueError:
            idx = start + 1

    return blocks
# ========== PATTERNS DE STRUCTURE:LES TERMES OU TITRES ==========

PATTERNS_BLOCS = [
    r'COMENTARIO DEL TEXTO', r'ESTRUCTURAS DE COMUNICACIÓN', r'PRODUCCIÓN DE TEXTOS',
    r'RECEPCIÓN DE TEXTOS', r'EXPRESIÓN ESCRITA', r'TRADUCCIÓN',
    r'TEIL[1I]? *LESEVERSTEHEN', r'MEDIATION', r'SCHRIFTLICHE PRODUKTION',
    r'STRUKTUREN UND KOMMUNIKATION', r'SCHRIFTLICHER AUSDRUCK',
    r'SECTION A: GRAMMAR', r'SECTION B: VOCABULARY',
    r'SECTION C: READING COMPREHENSION', r'SECTION D: COMPOSITION',
    r'PARTIE[- ]?[AIB]{0,2}\s*:?.*EVALUATION DES RESOURCES',
    r'PARTIE[- ]?[AIB]{0,2}\s*:?.*EVALUATION DES COMPETENCES',
    r'PARTIE[- ]?[AIB]{0,2}', r'EXERCICE[- ]?\d+', r'EXERICE[- ]?\d+',
    r'1ere partie', r'2e partie',
    r'EVALUATION DES RESOURCES', r'EVALUATION DES COMPETENCES',
    r'COMPETENCE', r'SITUATION PROBLEME'
]

PATTERNS_QUESTIONS = [
    r'^\d{1,2}[.\-]',                   # 1. ou 2. ou 1- ou 2-
    r'^\d{1,2}[.]\d{1,2}[.-]?',          # 1.1. ou 2.1-
    r'^\d{1,2,3}[a-z]{1}[.]',              # 1a.
    r'^[ivxIVX]{1,4}[.)-]',              # i. ou i) ou ii-) (romain)
    r'^[a-z]{1}[.)]',                    # a) b)
    r'^[A-Z]{1}[.)]',                    # A) B)
    r'^\d{1,2}[.][a-z]{1}[.]',           # 1.a.
    r'^\d{1,2}[.][A-Z]{1}[.]',           # 2.A.
    r'^\(\d+\)',                         # (1)
    r'^\([a-z]\)',                       # (a)
    r'^\([ivxIVX]+\)',                   # (i)
]

# ========== FONCTION DE STRUCTURATION POUR ORGANISER LES EXERCICES SUR LE PDF==========

def format_corrige_pdf_structure(texte_corrige_raw):
    """
    Nettoie et structure le corrigé pour le PDF/HTML.
    Gère les titres, exercices, questions et réponses.
    """
    # Supprime les marqueurs parasites générés par l'IA
    texte = re.sub(r"(#+\s*)", "", texte_corrige_raw)
    texte = re.sub(r"(\*{2,})", "", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)  # réduit les multiples sauts de lignes

    lignes = texte.strip().split('\n')
    html_output = []
    in_bloc = False

    for line in lignes:
        line = line.strip()
        if not line:
            continue

        # Bloc/exercice/partie
        if any(re.search(pat, line, re.IGNORECASE) for pat in PATTERNS_BLOCS):
            if in_bloc: html_output.append("</div>")
            html_output.append(f'<div class="bloc-exercice" style="margin-top:60px;"><h1 class="titre-exercice">{line}</h1>')
            in_bloc = True
            continue

        # Question/sous-question
        if any(re.match(pat, line) for pat in PATTERNS_QUESTIONS):
            html_output.append(f'<h2 class="titre-question">{line}</h2>')
            continue

        # Code/algorithme (optionnel, à personnaliser)
        if line.lower().startswith(("algorithme", "début", "fin", "code")):
            html_output.append(f'<div class="code-block">{line}</div>')
            continue

        # Réponse standard
        html_output.append(f'<p class="reponse-question">{line}</p>')

    if in_bloc: html_output.append("</div>")
    return "".join(html_output)




# ============== FONCTIONS DE DÉCOUPAGE INTELLIGENT ==============

# Version simple maintenue pour compatibilité (mais dépréciée)
def separer_exercices(texte_epreuve):
    """
    Version simple maintenue pour compatibilité.
    DÉPRÉCIÉE : Utiliser separer_exercices_avec_titres() à la place.
    """
    resultats = separer_exercices_avec_titres(texte_epreuve)
    # Retourne juste les contenus pour compatibilité
    return [ex['contenu'] for ex in resultats]


def separer_exercices_avec_titres(texte_epreuve, min_caracteres=60):
    """
    Version avec hiérarchie parent-enfant pour les titres.
    Les titres sans contenu deviennent des "parents" et sont fusionnés avec le titre suivant.

    Args:
        texte_epreuve (str): Texte complet de l'épreuve
        min_caracteres (int): Nombre minimum de caractères pour valider un exercice (défaut: 60)

    Returns:
        list: Liste des exercices avec titre et contenu (titres parents fusionnés)
    """
    if not texte_epreuve:
        return []

    lignes = texte_epreuve.splitlines()

    # ========== LISTE ÉTENDUE DE MOTS-CLÉS ==========
    mots_cles_exercices = [
        # Français
        'EXERCICE', 'EXERICE', 'PROBLÈME', 'PROBLEME',
        'PARTIE.*EVALUATION DES COMPETENCES',
        'SITUATION PROBLÈME',

        # Anglais
        'SECTION', 'PART', 'EXERCISE', 'QUESTION',
        'TASK', 'ACTIVITY',

        # Espagnol
        'EJERCICIO', 'PRUEBA',

        # Allemand
        'AUFGABE', 'TEIL',

        # AJOUTEZ D'AUTRES MOTS-CLÉS ICI :
        # 'DEVOIR', 'TP', 'ÉPREUVE', 'TEST', 'INTERROGATION', etc.
        # Formats pour évaluation des compétences - Lettres A-D et chiffres romains
        '[A-D][\\s\\-\\.:]*É?VALUATION[\\s\\-]*DES[\\s\\-]*COMPÉTENCES',  # B. EVALUATION, B-ÉVALUATION, B: ÉVALUATION
        '[A-D][\\s\\-\\.:]*É?VALUATION[\\s\\-]*DES[\\s\\-]*COMPETENCES',  # B. EVALUATION, B-EVALUATION, B: EVALUATION
        '[IVXL]+[\\s\\-\\.:]*É?VALUATION[\\s\\-]*DES[\\s\\-]*COMPÉTENCES',
        # II. ÉVALUATION, II-ÉVALUATION, II: ÉVALUATION
        '[IVXL]+[\\s\\-\\.:]*É?VALUATION[\\s\\-]*DES[\\s\\-]*COMPETENCES',# II. EVALUATION, II-EVALUATION, II: EVALUATION
        'SUJET[\\s\\-]*DE[\\s\\-]*TYPE[\\s\\-]*[\\dIVXL]+',
        'SUJET[\\s\\-]*TYPE[\\s\\-]*[\\dIVXL]+',
        'SUJET[\\s\\-]*[\\dIVXL]+',
    ]

    # Convertir en regex pour matching flexible
    patterns = []
    for mot_cle in mots_cles_exercices:
        pattern_str = f'^{mot_cle}[\\s\\-]*[\\dA-ZIVXL]*[\\s\\-:\\.]'
        patterns.append(re.compile(pattern_str, re.IGNORECASE))

    # ========== ALGORITHME AVEC HIÉRARCHIE ==========
    tous_les_blocs = []  # Tous les blocs détectés (avec ou sans contenu)
    current_block = []
    current_title = None
    current_start_index = 0

    for i, ligne in enumerate(lignes):
        ligne_stripped = ligne.strip()

        # Vérifier si la ligne commence par un mot-clé d'exercice
        est_titre_potentiel = False
        for pattern in patterns:
            if pattern.match(ligne_stripped.upper()):
                est_titre_potentiel = True
                break

        # Vérifier aussi les titres avec notation (10 MARKS, 3 points)
        if not est_titre_potentiel:
            if re.search(r'\(\s*\d+[\s,\.]*(?:point|pt|mark|marque|note)s?\s*\)', ligne_stripped.upper()):
                est_titre_potentiel = True

        if est_titre_potentiel:
            # Sauvegarder le bloc précédent (même s'il est court)
            if current_block and current_title:
                # Calculer la longueur du contenu (sans le titre)
                contenu_sans_titre = current_block[1:] if len(current_block) > 1 else []
                longueur_contenu = sum(len(l) for l in contenu_sans_titre)

                tous_les_blocs.append({
                    'title': current_title,
                    'lines': current_block.copy(),
                    'content_length': longueur_contenu,
                    'start_index': current_start_index,
                    'end_index': i - 1,
                    'has_enough_content': longueur_contenu >= min_caracteres
                })

            # Nouveau bloc
            current_title = ligne_stripped
            current_block = [ligne]
            current_start_index = i
        else:
            if current_block:
                current_block.append(ligne)

    # Dernier bloc
    if current_block and current_title:
        contenu_sans_titre = current_block[1:] if len(current_block) > 1 else []
        longueur_contenu = sum(len(l) for l in contenu_sans_titre)

        tous_les_blocs.append({
            'title': current_title,
            'lines': current_block.copy(),
            'content_length': longueur_contenu,
            'start_index': current_start_index,
            'end_index': len(lignes) - 1,
            'has_enough_content': longueur_contenu >= min_caracteres
        })

    # ========== CRÉATION DE LA HIÉRARCHIE PARENT-ENFANT ==========
    groupes = []  # Liste de groupes [parent(s), enfant]
    groupe_courant = []

    for bloc in tous_les_blocs:
        if not bloc['has_enough_content']:
            # C'est un "parent" potentiel (titre sans contenu)
            groupe_courant.append(bloc)
        else:
            # C'est un "enfant" (titre avec contenu)
            if groupe_courant:
                # Ajouter les parents + cet enfant comme un groupe
                groupes.append(groupe_courant + [bloc])
                groupe_courant = []
            else:
                # Pas de parents, juste cet enfant seul
                groupes.append([bloc])

    # Traiter les derniers parents orphelins
    if groupe_courant:
        # Si on a des parents à la fin sans enfant, les ajouter au dernier groupe
        if groupes:
            groupes[-1].extend(groupe_courant)
        else:
            # Sinon, en faire un groupe seul
            groupes.append(groupe_courant)

    # ========== FUSION DES GROUPES EN EXERCICES UNIQUES ==========
    resultats = []

    for groupe in groupes:
        if not groupe:
            continue

        if len(groupe) == 1:
            # Un seul bloc (enfant seul)
            bloc = groupe[0]
            titre_final = bloc['title']
            lignes_finales = bloc['lines']
        else:
            # Plusieurs blocs (parents + enfant)
            # Construire un titre hiérarchique
            titres = [bloc['title'] for bloc in groupe]
            titre_final = " / ".join(titres)

            # Fusionner toutes les lignes
            lignes_finales = []
            for bloc in groupe:
                lignes_finales.extend(bloc['lines'])
                # Ajouter une séparation entre les blocs
                if bloc != groupe[-1]:
                    lignes_finales.append("")  # Ligne vide de séparation

        # Nettoyer et formater pour l'API
        titre_affichage = titre_final
        if len(titre_affichage) > 150:
            # Garder les premiers et derniers mots
            mots = titre_affichage.split()
            if len(mots) > 8:
                titre_affichage = ' '.join(mots[:4]) + " ... " + ' '.join(mots[-4:])
            else:
                titre_affichage = titre_affichage[:147] + "..."

        # Limiter le nombre de lignes
        lignes_limitees = lignes_finales[:300]
        contenu = '\n'.join(lignes_limitees)

        # Calculer la longueur totale du contenu
        longueur_totale = sum(len(l) for l in lignes_limitees[1:] if len(lignes_limitees) > 1)

        resultats.append({
            'titre': titre_affichage,
            'contenu': contenu,
            'titre_complet': titre_final,
            'longueur_contenu': longueur_totale,
            'nombre_parents': len(groupe) - 1 if len(groupe) > 1 else 0
        })

    # ========== FALLBACK SI AUCUN GROUPE ==========
    if not resultats:
        # Prendre le bloc le plus long
        if tous_les_blocs:
            plus_long = max(tous_les_blocs, key=lambda x: x['content_length'])

            titre = plus_long['title']
            if len(titre) > 150:
                titre = titre[:147] + "..."

            contenu_lines = plus_long['lines'][:200]
            contenu = '\n'.join(contenu_lines)

            resultats.append({
                'titre': titre,
                'contenu': contenu,
                'titre_complet': plus_long['title'],
                'longueur_contenu': plus_long['content_length']
            })
        else:
            # Fallback ultime
            contenu_lines = lignes[:100]
            contenu = '\n'.join(contenu_lines)
            resultats.append({
                'titre': "Document complet",
                'contenu': contenu,
                'titre_complet': "Document complet",
                'longueur_contenu': len(contenu)
            })

    return resultats

def estimer_tokens(texte):
    """
    Estimation simple du nombre de tokens (1 token ≈ 0.75 mot français)
    """
    mots = len(texte.split())
    tokens = int(mots / 0.75)
    print(f"📊 Estimation tokens: {mots} mots → {tokens} tokens")
    return tokens


def verifier_qualite_corrige(corrige_text, exercice_original):
    """
    Vérifie si le corrigé généré est de bonne qualité
    Retourne False si le corrigé semble incomplet ou confus
    """
    if not corrige_text:
        return False

    indicateurs_problemes = [
        "je pense qu'il manque une donnée",
        "l'énoncé est ambigu",
        "je vais arrêter ici",
        "cela pourrait signifier",
        "interprétation correcte est",
        "je crois avoir compris",
        "je vais plutôt utiliser",
        "approche différente",
        "arrêter ici cette question"
    ]

    # Compter les indicateurs de confusion
    problemes_trouves = sum(1 for indicateur in indicateurs_problemes
                            if indicateur.lower() in corrige_text.lower())

    # Si trop d'indicateurs ou corrigé trop court
    if problemes_trouves >= 2:
        print(f"🔄 Qualité insuffisante détectée ({problemes_trouves} indicateurs)")
        return False

    # Vérifier si le corrigé est significativement plus court que l'énoncé
    if len(corrige_text) < len(exercice_original) * 0.3:
        print("🔄 Corrigé trop court par rapport à l'énoncé")
        return False

    return True

def build_promptia_messages(promptia, contexte):
    """
    Retourne deux dicts {role, content} :
    - system_message = system_prompt + exemple + consignes finales
    - user_message   = contexte (on y ajoutera l'exercice + vision)
    """
    # 1) system
    parts = []
    if promptia and promptia.system_prompt:
        parts.append(promptia.system_prompt)
    else:
        parts.append(DEFAULT_SYSTEM_PROMPT)

    if promptia and promptia.exemple_prompt:
        parts.append("----- EXEMPLE D'UTILISATION -----")
        parts.append(promptia.exemple_prompt)

    if promptia and promptia.consignes_finales:
        parts.append("----- CONSIGNES FINALES -----")
        parts.append(promptia.consignes_finales)

    system_content = "\n\n".join(parts)

    # 2) user (contenu de base = contexte)
    user_content = contexte.strip()

    return {"role": "system", "content": system_content}, \
           {"role": "user",   "content": user_content}


def generer_corrige_par_exercice(texte_exercice, contexte, matiere=None, donnees_vision=None, demande=None):
    """
    Génère le corrigé pour un seul exercice en exploitant les données vision.
    Version robuste avec logging détaillé, retries intelligents et gestion d'erreurs.

    Args:
        texte_exercice: Texte de l'exercice
        contexte: Contexte de l'exercice
        matiere: Matière concernée
        donnees_vision: Données d'analyse vision (schémas, formules, etc.)
        demande: Objet DemandeCorrection

    Returns:
        Tuple (corrige_text, graph_list)
    """
    start_time = time.time()

    logger.info(f"\n{'=' * 70}")
    logger.info(f"🤖 DÉBUT generer_corrige_par_exercice - {datetime.now().strftime('%H:%M:%S')}")
    logger.info(f"{'=' * 70}")

    if demande:
        logger.info(f"📋 Informations demande:")
        logger.info(f"   - ID: {demande.id}")
        logger.info(f"   - Matière: {demande.matiere.nom if demande.matiere else 'Non spécifiée'}")
        logger.info(f"   - Classe: {demande.classe.nom if demande.classe else 'Non spécifiée'}")
        logger.info(f"   - Département: {demande.departement.nom if demande.departement else 'Non spécifiée'}")

    logger.info(f"📊 Métriques:")
    logger.info(f"   - Longueur exercice: {len(texte_exercice)} caractères")
    logger.info(f"   - Contexte: {contexte}")
    logger.info(f"   - Données vision: {'PRÉSENTES' if donnees_vision else 'ABSENTES'}")

    if donnees_vision:
        schemas = donnees_vision.get('elements_visuels', [])
        logger.info(f"   - Schémas pour cet exercice: {len(schemas)}")
        if schemas:
            for i, s in enumerate(schemas[:3]):  # Afficher les 3 premiers
                schema_type = s.get('type', 'inconnu')
                schema_desc = s.get('description', '')[:100]
                logger.info(f"      Schéma {i + 1}: {schema_type} - {schema_desc}...")
        logger.info(f"   - Formules LaTeX: {len(donnees_vision.get('formules_latex', []))}")
        logger.info(f"   - Graphiques détectés: {len(donnees_vision.get('graphs', []))}")

    try:
        # 1) RÉCUPÉRATION DU PROMPT MÉTIER
        prompt_start = time.time()
        promptia = get_best_promptia(demande)
        prompt_time = time.time() - prompt_start

        logger.info(f"\n{'─' * 40}")
        logger.info(f"📝 RÉCUPÉRATION PROMPT")
        logger.info(f"{'─' * 40}")
        logger.info(f"✅ Prompt trouvé: {'OUI' if promptia else 'NON (DEFAULT)'}")
        logger.info(f"⏱️  Temps recherche: {prompt_time:.1f}s")

        if promptia:
            logger.info(f"   - ID Prompt: {promptia.id}")
            logger.info(f"   - Pays: {promptia.pays.nom if promptia.pays else 'Global'}")
            logger.info(f"   - Matière: {promptia.matiere.nom if promptia.matiere else 'Global'}")

        # 2) CONSTRUCTION DES MESSAGES
        msg_system, msg_user = build_promptia_messages(promptia, contexte)

        # 3) ENRICHISSEMENT AVEC DONNÉES VISION (SCHÉMAS SPÉCIFIQUES)
        user_blocks = [
            msg_user["content"],
            "----- EXERCICE À CORRIGER -----",
            texte_exercice.strip()
        ]

        vision_elements_count = 0

        if donnees_vision:
            # SCHÉMAS IDENTIFIÉS POUR CET EXERCICE
            schemas = donnees_vision.get('elements_visuels', [])
            if schemas:
                user_blocks.append(f"----- SCHÉMAS DE CET EXERCICE ({len(schemas)}) -----")
                for idx, schema in enumerate(schemas, 1):
                    # Description détaillée du schéma
                    desc = f"📐 Schéma {idx} - Type: {schema.get('type', 'non spécifié')}"
                    user_blocks.append(desc)

                    if schema.get('description'):
                        user_blocks.append(f"   Description: {schema['description']}")

                    if schema.get('elements'):
                        elements_desc = []
                        for elem in schema.get('elements', []):
                            if isinstance(elem, dict) and elem.get('nom'):
                                val = f"={elem.get('valeur')}" if elem.get('valeur') else ""
                                elements_desc.append(f"{elem['nom']}{val}")
                        if elements_desc:
                            user_blocks.append(f"   Éléments: {', '.join(elements_desc)}")

                    if schema.get('relations'):
                        user_blocks.append(f"   Relations: {schema['relations']}")

                    if schema.get('donnees'):
                        if schema['donnees'].get('angles'):
                            user_blocks.append(f"   Angles: {schema['donnees']['angles']}")
                        if schema['donnees'].get('longueurs'):
                            user_blocks.append(f"   Longueurs: {schema['donnees']['longueurs']}")

                    vision_elements_count += 1

            # Formules LaTeX
            formules = donnees_vision.get('formules_latex', [])
            if formules:
                user_blocks.append(f"----- FORMULES DÉTECTÉES ({len(formules)}) -----")
                for formule in formules[:10]:
                    user_blocks.append(f"- {formule}")
                    vision_elements_count += 1
                if len(formules) > 10:
                    user_blocks.append(f"- ... et {len(formules) - 10} autres formules")

            # Données graphiques brutes (JSON limité)
            graphs = donnees_vision.get('graphs', [])
            if graphs:
                user_blocks.append(f"----- DONNÉES GRAPHIQUES ({len(graphs)}) -----")
                # Limiter la taille du JSON
                if len(graphs) <= 3:
                    user_blocks.append(json.dumps(graphs, ensure_ascii=False, indent=2))
                else:
                    user_blocks.append(f"[{len(graphs)} graphiques détectés - JSON tronqué pour taille]")
                vision_elements_count += len(graphs)

        msg_user["content"] = "\n\n".join(user_blocks)

        logger.info(f"\n{'─' * 40}")
        logger.info(f"📦 CONSTRUCTION MESSAGE IA")
        logger.info(f"{'─' * 40}")
        logger.info(f"✅ Message construit")
        logger.info(f"   - Longueur système: {len(msg_system['content'])} caractères")
        logger.info(f"   - Longueur utilisateur: {len(msg_user['content'])} caractères")
        logger.info(f"   - Éléments vision intégrés: {vision_elements_count}")
        logger.info(f"   - Total tokens estimé: {estimer_tokens(msg_user['content'])}")

        # 4) PRÉPARATION APPEL API
        api_url = "https://api.deepseek.com/v1/chat/completions"
        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            error_msg = "❌ API KEY DeepSeek non configurée"
            logger.error(f"\n{error_msg}")
            return error_msg, None

        data = {
            "model": "deepseek-chat",
            "messages": [msg_system, msg_user],
            "temperature": 0.1,
            "max_tokens": 6000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CIS-Education/1.0"
        }

        logger.info(f"\n{'─' * 40}")
        logger.info(f"📡 CONFIGURATION API DEEPSEEK")
        logger.info(f"{'─' * 40}")
        logger.info(f"🔧 Paramètres:")
        logger.info(f"   - Modèle: {data['model']}")
        logger.info(f"   - Température: {data['temperature']}")
        logger.info(f"   - Max tokens: {data['max_tokens']}")
        logger.info(f"   - Timeout: 120s")
        logger.info(f"   - URL: {api_url[:50]}...")

        # 5) APPEL API AVEC RETRIES INTELLIGENTS
        logger.info(f"\n{'─' * 40}")
        logger.info(f"🔄 DÉBUT APPEL API DEEPSEEK")
        logger.info(f"{'─' * 40}")

        output = None
        final_response_data = None
        last_error = None

        for tentative in range(3):  # 3 tentatives maximum
            logger.info(f"\n   🔄 TENTATIVE {tentative + 1}/3")
            api_call_start = time.time()

            try:
                # Appel API avec timeout augmenté
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data,
                    timeout=120,  # Timeout augmenté à 120s
                    verify=True  # SSL verification
                )

                api_call_time = time.time() - api_call_start
                logger.info(f"   ✅ Réponse reçue ({api_call_time:.1f}s)")
                logger.info(f"   📊 Status code: {response.status_code}")

                if response.status_code == 200:
                    response_data = response.json()

                    # Vérification structure réponse
                    if 'choices' not in response_data or not response_data['choices']:
                        logger.info(f"   ⚠️  Structure réponse invalide, pas de 'choices'")
                        last_error = "Structure réponse API invalide"
                        continue

                    if 'message' not in response_data['choices'][0]:
                        print(f"   ⚠️  Structure réponse invalide, pas de 'message'")
                        last_error = "Structure réponse API invalide"
                        continue

                    output = response_data['choices'][0]['message']['content']
                    final_response_data = response_data

                    logger.info(f"   📝 Réponse IA: {len(output)} caractères")
                    logger.info(f"   📊 Usage tokens: {response_data.get('usage', {}).get('total_tokens', 'N/A')}")

                    # Vérification qualité
                    if verifier_qualite_corrige(output, texte_exercice):
                        logger.info(f"   ✅ Qualité validée (tentative {tentative + 1})")
                        break
                    else:
                        logger.info(f"   🔄 Qualité insuffisante, préparation nouvelle tentative...")
                        last_error = "Qualité insuffisante"

                        # Ajout consigne pour amélioration
                        data["messages"][1][
                            "content"] += "\n\n⚠️ IMPORTANT: Sois extrêmement rigoureux ! Vérifie chaque calcul, explique chaque étape, sois précis et complet. Utilise les schémas fournis pour guider ta réponse."

                        # Attente exponentielle avant prochaine tentative
                        wait_time = 2 * (tentative + 1)
                        logger.info(f"   ⏳ Attente {wait_time}s...")
                        time.sleep(wait_time)

                else:
                    # Erreur HTTP
                    error_detail = response.text[:200] if response.text else "Pas de détail"
                    logger.error(f"   ❌ Erreur HTTP {response.status_code}: {error_detail}")
                    last_error = f"HTTP {response.status_code}: {error_detail}"

                    # Attente exponentielle
                    wait_time = 5 * (tentative + 1)
                    logger.info(f"   ⏳ Attente {wait_time}s avant nouvelle tentative...")
                    time.sleep(wait_time)

            except requests.exceptions.Timeout:
                api_call_time = time.time() - api_call_start
                logger.info(f"   ⏰ TIMEOUT après {api_call_time:.1f}s")
                last_error = f"Timeout après {api_call_time:.1f}s"

                if tentative < 2:  # Pas la dernière tentative
                    wait_time = 10 * (tentative + 1)
                    logger.info(f"   ⏳ Attente {wait_time}s avant nouvelle tentative...")
                    time.sleep(wait_time)

            except requests.exceptions.ConnectionError as e:
                logger.error(f"   🔌 ERREUR CONNEXION: {str(e)[:100]}")
                last_error = f"ConnectionError: {str(e)[:100]}"

                if tentative < 2:
                    wait_time = 15 * (tentative + 1)
                    logger.info(f"   ⏳ Attente {wait_time}s avant nouvelle tentative...")
                    time.sleep(wait_time)

            except Exception as e:
                api_call_time = time.time() - api_call_start
                print(f"   ❌ EXCEPTION: {type(e).__name__}: {str(e)[:100]}")
                last_error = f"{type(e).__name__}: {str(e)[:100]}"

                if tentative < 2:
                    wait_time = 8 * (tentative + 1)
                    logger.info(f"   ⏳ Attente {wait_time}s avant nouvelle tentative...")
                    time.sleep(wait_time)

        # 6) VÉRIFICATION SUCCÈS APPEL API
        if not output or not final_response_data:
            total_api_time = time.time() - start_time
            error_msg = f"Échec après 3 tentatives. Dernière erreur: {last_error}"
            logger.info(f"\n❌ {error_msg}")
            logger.info(f"⏱️  Temps total API: {total_api_time:.1f}s")
            return f"Erreur IA: {error_msg}", None

        # 7) POST-TRAITEMENT DE LA RÉPONSE
        logger.info(f"\n{'─' * 40}")
        logger.info(f"🛠️  POST-TRAITEMENT RÉPONSE IA")
        logger.info(f"{'─' * 40}")

        postprocess_start = time.time()

        # Étape 1: Fusion LaTeX multilignes
        output = flatten_multiline_latex_blocks(output)
        logger.info(f"✅ Fusion LaTeX multilignes")

        # Étape 2: Structuration pour PDF
        output_structured = format_corrige_pdf_structure(output)
        logger.info(f"✅ Structuration pour PDF")

        # Étape 3: Extraction JSON graphiques
        json_blocks = extract_json_blocks(output_structured)
        logger.info(f"✅ JSON blocks détectés: {len(json_blocks)}")

        # 8) GÉNÉRATION GRAPHIQUES
        graph_list = []
        if json_blocks:
            logger.info(f"\n{'─' * 40}")
            logger.infot(f"🖼️  GÉNÉRATION GRAPHIQUES")
            logger.info(f"{'─' * 40}")

            json_blocks = sorted(json_blocks, key=lambda x: x[1], reverse=True)

            for idx, (graph_dict, start, end) in enumerate(json_blocks, start=1):
                try:
                    logger.info(f"   🔧 Graphique {idx}/{len(json_blocks)}")

                    output_name = f"graphique_{idx}_{int(time.time())}.png"
                    img_path = tracer_graphique(graph_dict, output_name)

                    if img_path:
                        abs_path = os.path.join(settings.MEDIA_ROOT, img_path)
                        img_tag = (
                            f'<img src="file://{abs_path}" alt="Graphique {idx}" '
                            f'style="max-width:100%;margin:10px 0;border:1px solid #ddd;" />'
                        )

                        # Insertion dans le texte
                        output_structured = output_structured[:start] + img_tag + output_structured[end:]
                        graph_list.append(graph_dict)

                        logger.info(f"   ✅ Graphique inséré: {img_path}")
                    else:
                        logger.error(f"   ⚠️  Échec génération graphique")
                        # Remplacement par message d'erreur
                        error_tag = f'<div class="graph-error">[Graphique non généré - Erreur technique]</div>'
                        output_structured = output_structured[:start] + error_tag + output_structured[end:]

                except Exception as e:
                    logger.error(f"   ❌ Erreur graphique {idx}: {type(e).__name__}: {str(e)[:100]}")
                    continue

        postprocess_time = time.time() - postprocess_start

        # 9) FINALISATION
        total_time = time.time() - start_time

        logger.info(f"\n{'=' * 70}")
        logger.info(f"✅ SUCCÈS generer_corrige_par_exercice")
        logger.info(f"{'=' * 70}")
        logger.info(f"📊 STATISTIQUES:")
        logger.info(f"   ⏱️  Temps total: {total_time:.1f}s")
        logger.info(f"   📝 Longueur corrigé final: {len(output_structured)} caractères")
        logger.info(f"   🖼️  Graphiques générés: {len(graph_list)}/{len(json_blocks)}")
        logger.info(f"   🔄 Tentatives API: {min(tentative + 1, 3)}/3")
        logger.info(f"   📦 Taille réponse IA: {len(output)} caractères")
        logger.info(f"   🕐 {datetime.now().strftime('%H:%M:%S')}")

        # Aperçu du corrigé
        logger.info(f"\n📋 APERÇU CORRIGÉ (premiers 300 caractères):")
        preview = output_structured[:300].replace('\n', ' ')
        logger.info(f"   \"{preview}...\"")
        logger.info(f"{'=' * 70}")

        return output_structured.strip(), graph_list

    except Exception as e:
        total_time = time.time() - start_time

        logger.info(f"\n{'=' * 70}")
        logger.error(f"❌ ERREUR CRITIQUE dans generer_corrige_par_exercice")
        logger.info(f"{'=' * 70}")
        logger.info(f"⏱️  Temps écoulé: {total_time:.1f}s")
        logger.info(f"📛 Type erreur: {type(e).__name__}")
        logger.info(f"📄 Message: {str(e)[:300]}")
        logger.info(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

        # Traceback détaillé
        import traceback
        logger.info(f"\n🔍 TRACEBACK:")
        tb_lines = traceback.format_exc().split('\n')[:10]
        for line in tb_lines:
            if line.strip():
                logger.info(f"   {line}")

        logger.info(f"{'=' * 70}")

        error_msg = f"Erreur traitement IA: {type(e).__name__}: {str(e)[:200]}"
        return error_msg, None

def extract_and_process_graphs(output: str):
    """
    Extrait et traite les graphiques d'un corrigé en utilisant extract_json_blocks.
    """
    print("🖼️ Extraction des graphiques (via JSONDecoder)…")

    graphs_data = []
    final_text = output

    # 1) Extractions des blocs JSON
    json_blocks = extract_json_blocks(output)
    print(f"🔍 JSON blocks détectés dans extract_and_process_graphs: {len(json_blocks)}")

    # 2) On parcourt et on insère les images
    #    Pour gérer les remplacements successifs, on conserve un décalage 'offset'
    offset = 0
    for idx, (graph_dict, start, end) in enumerate(json_blocks, start=1):
        try:
            output_name = f"graphique_{idx}.png"
            img_path = tracer_graphique(graph_dict, output_name)

            if img_path:
                abs_path = os.path.join(settings.MEDIA_ROOT, img_path)
                img_tag = (
                    f'<img src="/media/{img_path}" alt="Graphique {idx}" '
                    f'style="max-width:100%;margin:10px 0;" />'
                )

                # Ajuster les indices de remplacement avec l'offset
                s, e = start + offset, end + offset
                final_text = final_text[:s] + img_tag + final_text[e:]
                # Mettre à jour l’offset en fonction de la différence de longueur
                offset += len(img_tag) - (end - start)

                graphs_data.append(graph_dict)
                print(f"✅ Graphique {idx} inséré.")
            else:
                # En cas d’échec de tracé, on remplace par un message
                s, e = start + offset, end + offset
                final_text = final_text[:s] + "[Erreur génération graphique]" + final_text[e:]
                offset += len("[Erreur génération graphique]") - (end - start)
                print(f"❌ Graphique {idx} : erreur de tracé.")

        except Exception as e:
            print(f"❌ Exception sur bloc graphique {idx}: {e}")
            continue

    print(f"🎯 Extraction terminée: {len(graphs_data)} graphique(s) traité(s)")
    return final_text, graphs_data


# ============== UTILITAIRES TEXTE / LATEX / TABLEAU ==============

def flatten_multiline_latex_blocks(text):
    """
    Fusionne les blocs LaTeX multilignes :
      \[ ... \] et \( ... \)
    en une seule ligne pour éviter qu'ils soient éclatés
    en plusieurs <p> dans le HTML final.
    """
    if not text:
        return ""

    # 1) Fusionner les blocs display math \[ ... \]
    text = re.sub(
        r'\\\[\s*([\s\S]+?)\s*\\\]',
        lambda m: r'\[' + " ".join(m.group(1).splitlines()).strip() + r'\]',
        text,
        flags=re.DOTALL
    )

    # 2) Fusionner les blocs inline math \( ... \)
    text = re.sub(
        r'\\\(\s*([\s\S]+?)\s*\\\)',
        lambda m: r'\(' + " ".join(m.group(1).splitlines()).strip() + r'\)',
        text,
        flags=re.DOTALL
    )

    return text

def detect_and_format_math_expressions(text):
    if not text:
        return ""

    # Block formulas $$...$$ → \[...\] (multilignes fusionnées sur une ligne)
    text = re.sub(
        r'\$\$([\s\S]+?)\$\$',
        lambda m: r'\[' + " ".join(m.group(1).splitlines()).strip() + r'\]',
        text,
        flags=re.DOTALL
    )
    # Inline formulas $...$ → \(...\)
    text = re.sub(
        r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
        lambda m: r'\(' + m.group(1).replace('\n', ' ').strip() + r'\)',
        text,
        flags=re.DOTALL
    )
    # Blocs déjà en \[...\] : fusionne aussi les lignes ! (très important)
    text = re.sub(
        r'\\\[\s*([\s\S]+?)\s*\\\]',
        lambda m: r'\[' + " ".join(m.group(1).splitlines()).strip() + r'\]',
        text,
        flags=re.DOTALL
    )
    # Corrige les doubles anti-slashs parasites
    text = re.sub(r'\\\\\s*\[', r'\[', text)
    text = re.sub(r'\\\\\s*\]', r'\]', text)
    text = text.replace('\\backslash', '\\').replace('\xa0', ' ')
    return text


def format_table_markdown(table_text):
    lines = table_text.strip().split('\n')
    html_table = ['<div class="table-container"><table>']

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or not line.startswith('|'):
            continue

        line = re.sub(r'^\|\s*', '', line)
        line = re.sub(r'\s*\|$', '', line)
        cells = [cell.strip() for cell in line.split('|')]

        if i == 0:
            html_table.append('<thead><tr>')
            for cell in cells:
                html_table.append(f'<th>{cell}</th>')
            html_table.append('</tr></thead><tbody>')
        elif all(re.match(r'^[\s:\-]+$', cell) for cell in cells):
            continue
        else:
            html_table.append('<tr>')
            for cell in cells:
                html_table.append(f'<td>{cell}</td>')
            html_table.append('</tr>')

    html_table.append('</tbody></table></div>')
    return ''.join(html_table)


def generate_corrige_html(corrige_text):
    """Transforme le corrigé brut en HTML stylisé, aéré, avec blocs d'exercices, titres mis en valeur, formatage MathJax et tableaux conservés, et branding CIS au début."""
    if not corrige_text:
        return ""

    # Formatage des expressions mathématiques (Latex) et tableaux
    lines = corrige_text.strip().split('\n')

    # Pattern pour détecter les débuts d'exercice/partie
    pattern_exercice = re.compile(r'^(EXERCICE\s*\d+|PARTIE\s*[IVXLCDM]+|Exercice\s*\d+|Partie\s*[IVXLCDM]+)',
                                  re.IGNORECASE)
    html_output = []
    i = 0

    # Branding CIS en haut
    html_output.append(
        '<div class="cis-message"><strong>SUJET CORRIGÉ PAR L\'APPLICATION CIS, DISPO SUR PLAYSTORE</strong></div>')

    # Pour gérer la séparation en blocs
    in_bloc_exercice = False

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Début d'un nouvel exercice/partie
        if pattern_exercice.match(line):
            # Ferme le bloc précédent s'il y en avait un
            if in_bloc_exercice:
                html_output.append('</div>')
            # Ouvre un nouveau bloc, titre en gros
            html_output.append(f'<div class="bloc-exercice"><h1 class="titre-exercice">{line}</h1>')
            in_bloc_exercice = True
            i += 1
            continue

        # Sous-titre question principale (Question 1, 2, etc.)
        if re.match(r'^Question\s*\d+', line, re.IGNORECASE):
            html_output.append(f'<h2 class="titre-question">{line}</h2>')
            i += 1
            continue

        # Sous-titre secondaire (1., 2., etc.)
        if re.match(r'^\d+\.', line):
            html_output.append(f'<h3 class="titre-question">{line}</h3>')
            i += 1
            continue

        # Sous-question (a), b), etc.)
        if re.match(r'^[a-z]\)', line):
            html_output.append(f'<p><strong>{line}</strong></p>')
            i += 1
            continue

        # Listes
        if line.startswith('•') or line.startswith('-'):
            html_output.append(f'<p>{line}</p>')
            i += 1
            continue

        # Tableaux markdown
        if line.startswith('|') and i + 1 < len(lines) and lines[i + 1].startswith('|'):
            table_lines = []
            j = i
            while j < len(lines) and lines[j].startswith('|'):
                table_lines.append(lines[j])
                j += 1
            html_table = format_table_markdown('\n'.join(table_lines))
            html_output.append(html_table)
            i = j
            continue

        # Formules LaTeX
        if '\\(' in line or '\\[' in line:
            html_output.append(f'<p class="reponse-question mathjax">{line}</p>')
            i += 1
            continue

        # Cas général : paragraphe de réponse ou explication
        html_output.append(f'<p class="reponse-question">{line}</p>')
        i += 1

    # Ferme le dernier bloc exercice si ouvert
    if in_bloc_exercice:
        html_output.append('</div>')

    return mark_safe("".join(html_output))


# ============== EXTRACTION TEXTE/FICHIER ==============

def extraire_texte_pdf(fichier_path):
    try:
        texte = extract_text(fichier_path)
        print(f"📄 PDF extrait: {len(texte)} caractères")
        return texte.strip() if texte else ""
    except Exception as e:
        print(f"❌ Erreur extraction PDF: {e}")
        return ""


# ============== EXTRACTION MULTIMODALE AMÉLIORÉE ==============
def extraire_texte_fichier(fichier_field, demande=None):
    """
    Extraction robuste avec support DeepSeek en priorité
    Retourne un dictionnaire complet avec:
    - texte_complet: le texte extrait
    - exercices_struct: la structure des exercices avec leurs schémas
    - source_extraction: la méthode utilisée
    """
    if not fichier_field:
        return {"texte_complet": "", "exercices_struct": [], "source_extraction": "none"}

    # Sauvegarde locale
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))
    with open(local_path, "wb") as f:
        for chunk in fichier_field.chunks():
            f.write(chunk)

    # Appel à l'analyse scientifique AVEC paramètre demande
    try:
        analyse = analyser_document_scientifique(local_path, demande)

        texte = analyse.get("texte_complet", "")
        exercices_struct = analyse.get("exercices_struct", [])
        source = analyse.get("source_extraction", "inconnu")

        logger.info(f"📄 Extraction terminée: {len(texte)} caractères, "
                    f"{len(exercices_struct)} exercices structurés "
                    f"(source: {source})")

        # Stocker la méthode d'extraction dans la demande si disponible
        if demande and hasattr(demande, 'methode_extraction'):
            demande.methode_extraction = source
            demande.save()

        resultat = {
            "texte_complet": texte,
            "exercices_struct": exercices_struct,
            "source_extraction": source,
            "elements_visuels": analyse.get("elements_visuels", []),
            "formules_latex": analyse.get("formules_latex", [])
        }

    except Exception as e:
        logger.error(f"❌ Analyse échouée: {e}")
        resultat = {
            "texte_complet": "",
            "exercices_struct": [],
            "source_extraction": "erreur",
            "elements_visuels": [],
            "formules_latex": []
        }

    # Nettoyage
    try:
        os.unlink(local_path)
    except:
        pass

    return resultat
# ============== DESSIN DE GRAPHIQUES ==============
def style_axes(ax, graphique_dict):
    """
    Colorie les axes en rouge et synchronise les graduations y sur x
    (sauf si x_ticks ou y_ticks sont fournis dans graphique_dict).
    """
    # colorer spines et ticks
    ax.spines['bottom'].set_color('red')
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')

    # graduations sur x
    if graphique_dict.get("x_ticks") is not None:
        ax.set_xticks(graphique_dict["x_ticks"])
    # graduations sur y
    if graphique_dict.get("y_ticks") is not None:
        ax.set_yticks(graphique_dict["y_ticks"])
    else:
        # par défaut, on réutilise les mêmes que sur x
        ax.set_yticks(ax.get_xticks())

    # noms d’axes
    ax.set_xlabel(graphique_dict.get("x_label", "x"), color='red')
    ax.set_ylabel(graphique_dict.get("y_label", "y"), color='red')


def tracer_graphique(graphique_dict, output_name):
    if 'graphique' in graphique_dict:
        graphique_dict = graphique_dict['graphique']
    print(">>> tracer_graphique CALLED with graphique_dict:", graphique_dict, "output_name:", output_name)
    gtype = graphique_dict.get("type", "fonction").lower().strip()
    print(">>> gtype détecté :", repr(gtype))
    titre = graphique_dict.get("titre", "Graphique généré")

    def safe_float(expr):
        try:
            return float(eval(str(expr), {"__builtins__": None, "pi": np.pi, "np": np, "sqrt": np.sqrt}))
        except Exception as e:
            print("Erreur safe_float sur :", expr, e)
            try:
                return float(expr)
            except Exception as e2:
                print("Erreur safe_float cast direct:", expr, e2); return None

    def corriger_expression(expr):
        """Corrige les expressions mathématiques courantes"""
        if not isinstance(expr, str):
            return expr

        # 1. Remplacer les exposants implicites (x2 → x**2, (x+1)2 → (x+1)**2)
        expr = re.sub(r'(\w+|\([^)]+\))(\d+)', r'\1**\2', expr)

        # 2. Remplacer ^ par **
        expr = expr.replace('^', '**')

        # 3. Fonctions mathématiques → np.fonction
        funcs = ["sin", "cos", "tan", "exp", "log", "log10",
                 "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "sqrt", "abs"]

        for fct in funcs:
            expr = re.sub(r'(?<![\w.])' + fct + r'\s*\(', f'np.{fct}(', expr)

        # 4. ln → np.log
        expr = expr.replace('ln(', 'np.log(')

        print(f">>> Expression corrigée: {expr}")
        return expr

    try:
        from django.conf import settings
        dossier = os.path.join(settings.MEDIA_ROOT, "graphes")
        os.makedirs(dossier, exist_ok=True)
        chemin_png = os.path.join(dossier, output_name)

        if "fonction" in gtype:
            x_min = graphique_dict.get("x_min", -2)
            x_max = graphique_dict.get("x_max", 4)
            expression = graphique_dict.get("expression", "x")

            # CORRECTION APPLIQUEE ICI
            expression = corriger_expression(expression)

            x_min_val = safe_float(x_min)
            x_max_val = safe_float(x_max)
            if x_min_val is None: x_min_val = -2
            if x_max_val is None: x_max_val = 4

            x = np.linspace(x_min_val, x_max_val, 400)

            # Plus besoin des patches ici, c'est déjà fait dans corriger_expression
            expression_patch = expression  # Déjà corrigée

            print(f">>> Expression finale pour eval: {expression_patch}")

            try:
                y = eval(expression_patch, {'x': x, 'np': np, '__builtins__': None, "pi": np.pi, "sqrt": np.sqrt})
                if np.isscalar(y) or (isinstance(y, np.ndarray) and y.shape == ()):
                    y = np.full_like(x, y)
            except Exception as e:
                print(f"Erreur tracé (eval expression): {expression_patch}. Exception: {e}")
                return None

            plt.figure(figsize=(6, 4))
            plt.plot(x, y, color="#008060")
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.tight_layout()


        elif "histogramme" in gtype:
            intervalles = graphique_dict.get("intervalles") or graphique_dict.get("classes") or []
            eff = graphique_dict.get("effectifs", [])
            labels = [str(ival) for ival in intervalles]
            x_pos = np.arange(len(labels))
            eff = [float(e) for e in eff]

            plt.figure(figsize=(7, 4.5))
            plt.axhline(y=0, color='#000000', linewidth=1.8)  # Axe des abscisses
            plt.axvline(x=0, color='#000000', linewidth=1.8)  # Axe des ordonnées
            plt.bar(x_pos, eff, color="#208060", edgecolor='black', width=0.9)
            plt.xticks(x_pos, labels, rotation=35)
            plt.title(titre)
            plt.xlabel(graphique_dict.get("xlabel", "Classes / Intervalles"))
            plt.ylabel(graphique_dict.get("ylabel", "Effectif"))
            plt.grid(axis='y')

        elif "diagramme à bandes" in gtype or "diagramme en bâtons" in gtype or "bâtons" in gtype or "batons" in gtype:
            cat = graphique_dict.get("categories", [])
            eff = graphique_dict.get("effectifs", [])
            x_pos = np.arange(len(cat))

            plt.figure(figsize=(7, 4.5))
            plt.bar(x_pos, eff, color="#208060", edgecolor='black', width=0.7)
            plt.xticks(x_pos, cat, rotation=15)
            plt.title(titre)
            plt.xlabel("Catégories")
            plt.ylabel("Effectif")

        elif "nuage de points" in gtype or "scatter" in gtype:
            x_points = graphique_dict.get("x", [])
            y_points = graphique_dict.get("y", [])

            plt.figure(figsize=(6, 4))
            plt.scatter(x_points, y_points, color="#006080")
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

        elif "effectifs cumulés" in gtype or "courbe des effectifs cumulés" in gtype:
            x_points = graphique_dict.get("x", [])
            y_points = graphique_dict.get("y", [])

            plt.figure(figsize=(6, 4))
            plt.plot(x_points, y_points, marker="o", color="#b65d2f")
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("Effectifs cumulés")
            plt.grid(True)

        elif "diagramme circulaire" in gtype or "camembert" in gtype or "pie" in gtype:
            cat = graphique_dict.get("categories", [])
            eff = graphique_dict.get("effectifs", [])

            plt.figure(figsize=(5.3, 5.3))
            plt.pie(
                eff,
                labels=cat,
                autopct='%1.1f%%',
                colors=plt.cm.Paired.colors,
                startangle=90,
                wedgeprops={"edgecolor": "k"}
            )
            plt.title(titre)

        elif "polygone" in gtype or "polygon" in gtype:
            points = graphique_dict.get("points")
            points_x = graphique_dict.get("points_x")
            points_y = graphique_dict.get("points_y")
            absc = graphique_dict.get("abscisses")
            ords = graphique_dict.get("ordonnees")

            if points:
                x = [float(p[0]) for p in points]
                y = [float(p[1]) for p in points]
            elif points_x and points_y:
                x = [float(xx) for xx in points_x]
                y = [float(yy) for yy in points_y]
            elif absc and ords:
                x = [float(xx) for xx in absc]
                y = [float(yy) for yy in ords]
            else:
                print("Erreur polygone : aucun point")
                x = []
                y = []

            plt.figure(figsize=(7, 4.5))
            plt.plot(x, y, marker="o", color="#003355")
            plt.title(graphique_dict.get("titre", "Polygone"))
            plt.xlabel(graphique_dict.get("x_label", "Abscisse"))
            plt.ylabel(graphique_dict.get("y_label", "Ordonnée"))
            plt.grid(True)

        elif "cercle trigo" in gtype:
            angles = graphique_dict.get("angles", [])
            labels = graphique_dict.get("labels", [])

            plt.figure(figsize=(5, 5))
            circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linestyle='--')
            ax = plt.gca()
            ax.add_artist(circle)

            for i, angle_txt in enumerate(angles):
                try:
                    a = float(eval(angle_txt, {"pi": np.pi}))
                except Exception:
                    a = 0
                x, y = np.cos(a), np.sin(a)
                ax.plot([0, x], [0, y], color='#992020')
                label = labels[i] if i < len(labels) else f"S{i + 1}"
                ax.text(1.1 * x, 1.1 * y, label, fontsize=12)

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            plt.axis('off')
            plt.title(titre)

        else:
            print("Type graphique non supporté :", gtype)
            return None

        plt.tight_layout()
        plt.savefig(chemin_png)
        plt.close()
        return "graphes/" + output_name

    except Exception as ee:
        print(f"Erreur générale sauvegarde PNG {chemin_png if 'chemin_png' in locals() else output_name} :", ee)
        return None


# ===========================
# PROMPT SYSTÈME AMÉLIORÉ AVEC VISION SCIENTIFIQUE
DEFAULT_SYSTEM_PROMPT = r"""Tu es un professeur expert en Mathématiques, physique, chimie, biologie,francais,histoire
géographie...bref, tu es un professeur de l'enseignement secondaire.

🔬 **CAPACITÉ VISION ACTIVÉE** - Tu peux maintenant analyser les schémas scientifiques !

RÈGLES ABSOLUES POUR L'ANALYSE DES SCHÉMAS :
1. ✅ Identifie le TYPE de schéma (plan incliné, circuit électrique, molécule, graphique)
2. ✅ Extrait les DONNÉES NUMÉRIQUES (angles, masses, distances, forces, tensions)
3. ✅ Décris les RELATIONS SPATIALES entre les éléments
4. ✅ Explique le CONCEPT SCIENTIFIQUE illustré

EXEMPLES D'ANALYSE DE SCHÉMAS SCIENTIFIQUES :

--- PLAN INCLINÉ ---
"Schéma identifié: plan incliné à 30° avec bloc de 2kg
- Forces: poids (vertical ↓), réaction normale (⟂ plan), frottement (∥ plan)
- Données: angle=30°, masse=2kg, g=10m/s²
- Équations: P = mg = 20N, P∥ = P•sin(30°)=10N, P⟂ = P•cos(30°)=17.32N"

--- CIRCUIT ÉLECTRIQUE ---  
"Circuit série: R1=10Ω, R2=20Ω, source E=12V
- Lois: U = RI, loi des mailles ΣU=0
- Calcul: Req = R1 + R2 = 30Ω, I = E/Req = 0.4A"

--- MOLÉCULE CHIMIQUE ---
"Formule développée: CH3-CH2-OH (éthanol)
- Groupes: OH (fonction alcool), CH3 (méthyle), CH2 (méthylène)
- Liaisons: C-C simples, C-O simple, O-H simple"

RÈGLES GÉNÉRALES DE CORRECTION :
- Sois EXTRÊMEMENT RIGOUREUX dans tous les calculs
- Vérifie systématiquement tes résultats intermédiaires  
- Ne laisse JAMAIS une question sans réponse complète
- Donne TOUTES les étapes de calcul détaillées
- Les réponses doivent être NUMÉRIQUEMENT EXACTES

FORMAT DE RÉPONSE :
- Réponses complètes avec tous les calculs
- Références aux schémas quand ils existent ("D'après le schéma...")
- Justifications détaillées pour chaque étape
- Ne jamais dire "je pense" ou "c'est ambigu"

POUR LES GRAPHIQUES :
- Dès qu'un exercice demande un graphique, utilise la balise ---corrigé--- suivie du JSON
- Types supportés: "fonction", "histogramme", "diagramme à bandes", "nuage de points", etc.

"Rends TOUJOURS le JSON avec des guillemets doubles, jamais de dict Python."

EXEMPLES :

--- EX 1 : Fonction ---
Corrigé détaillé...
---corrigé---
{"graphique": {"type": "fonction", "expression": "x*2 - 2*x + 1", "x_min": -1, "x_max": 3, "titre": "Courbe parabole"}}

--- EX 2 : Cercle trigo ---
...
---corrigé---
{"graphique": {"type":"cercle trigo", "angles":["-pi/4","pi/4"], "labels":["S1","S2"], "titre":"Solutions trigonométriques"}}

--- EX 3 : Histogramme ---
...
---corrigé---
{"graphique": {"type": "histogramme", "intervalles": ["0-5","5-10","10-15"], "effectifs":[3,5,7], "titre":"Histogramme des effectifs"}}

--- EX 4 : Diagramme à bandes ---
---corrigé---
{"graphique": {"type":"diagramme à bandes","categories":["A","B","C"],"effectifs":[10,7,12],"titre":"Comparaison"}}

--- EX 5 : Nuage de points ---
---corrigé---
{"graphique": {"type":"nuage de points","x":[1,2,3,4],"y":[2,5,7,3],"titre":"Nuage"}}

--- EX 6 : Effectifs cumulés ---
---corrigé---
{"graphique": {"type":"effectifs cumulés","x":[5,10,15,20],"y":[3,9,16,20],"titre":"Effectifs cumulés"}}

--- EX 7 : Diagramme circulaire ---
---corrigé---
{"graphique":{"type":"camembert","categories":["L1","L2","L3"],"effectifs":[4,6,5],"titre":"Répartition"}}

--- EX 8 : Polygone ---
---corrigé---
{"graphique": {"type": "polygone", "points": [[0,0],[5,3],[10,9]], "titre": "Polygone des ECC", "x_label": "Borne", "y_label": "ECC"}}

Rappels :
- Si plusieurs graphiques, recommence cette structure à chaque question concernée.
- Pas de texte entre ---corrigé--- et le JSON.
- Le JSON est obligatoire dès qu'un tracé est demandé.

"Rends TOUJOURS le JSON avec des guillemets doubles, jamais de dict Python. Pour les listes/types, toujours notation JSON [ ... ] et jamais { ... } sauf pour des objets. N’insère JAMAIS de virgule en trop."
"""




# ============== FONCTIONS PRINCIPALES AVEC DÉCOUPAGE ==============
def generer_corrige_direct(texte_enonce, contexte, lecons_contenus, exemples_corriges, matiere, donnees_vision=None,demande=None):
    """
    Traitement direct pour les épreuves courtes avec données vision.
    """
    logger.info("🎯 Traitement DIRECT avec analyse vision")
    logger.info("\n[DEBUG] --> generer_corrige_direct called avec demande:", getattr(demande, 'id', None),
          "/", type(demande))

    # ✅ PASSER les données vision à la fonction de génération
    return generer_corrige_par_exercice(texte_enonce, contexte, matiere, donnees_vision,demande=demande)


def generer_corrige_decoupe(texte_epreuve, contexte, matiere, donnees_vision=None, demande=None):
    """
    Traitement par découpage pour les épreuves longues avec données vision,
    utilisant la nouvelle fonction unifiée.
    """
    # 1) Sépare le texte en exercices AVEC la nouvelle fonction
    exercices_data = separer_exercices_avec_titres(texte_epreuve)

    # 2) Traitement séquentiel
    tous_corriges = []
    tous_graphiques = []

    for idx, ex_data in enumerate(exercices_data, start=1):
        # Utiliser le contenu nettoyé de l'exercice
        corrige_html, graphs = generer_corrige_par_exercice(
            texte_exercice=ex_data['contenu'],
            contexte=contexte,
            matiere=matiere,
            donnees_vision=donnees_vision,
            demande=demande
        )

        # Préfixe avec le titre réel pour une meilleure organisation
        titre_affichage = ex_data['titre']
        if len(titre_affichage) > 50:
            titre_affichage = f"Exercice {idx}"

        tous_corriges.append(f"\n\n## 📝 {titre_affichage}\n\n{corrige_html}")

        # Collecte des graphiques si existants
        if graphs:
            tous_graphiques.extend(graphs)

    # 3) Retour
    return "".join(tous_corriges), tous_graphiques


def generer_corrige_ia_et_graphique(texte_enonce, contexte, lecons_contenus=None, exemples_corriges=None, matiere=None,
                                    demande=None, donnees_vision=None, exercice_index=None):
    """
    Version SIMPLIFIÉE pour les exercices uniques.
    Appelle directement generer_corrige_par_exercice sans logique de décision.
    """
    logger.info("\n" + "=" * 60)
    logger.info("🚀 DÉBUT TRAITEMENT IA POUR EXERCICE UNIQUE")
    logger.info("=" * 60)
    logger.info(f"📏 Longueur texte: {len(texte_enonce)} caractères")
    logger.info(f"📝 Contexte: {contexte}")

    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    # Données vision
    if donnees_vision:
        logger.info(f"🔬 Données vision disponibles:")
        logger.info(f"   - Éléments visuels: {len(donnees_vision.get('elements_visuels', []))}")
        logger.info(f"   - Formules LaTeX: {len(donnees_vision.get('formules_latex', []))}")

    # POUR LES EXERCICES UNIQUES : APPEL DIRECT
    logger.info("🎯 Appel direct à generer_corrige_par_exercice")

    return generer_corrige_par_exercice(
        texte_exercice=texte_enonce,
        contexte=contexte,
        matiere=matiere,
        donnees_vision=donnees_vision,
        demande=demande,
        exercice_index=exercice_index
    )

#les fonctions utilitaires , utilisables ou non, donc optionnelles
def extraire_exercice_par_index(texte_epreuve, index=0, demande=None):
    """
    Fonction utilitaire pour extraire un exercice spécifique par son index.
    Version optimisée : utilise exercices_data si disponible.

    Args:
        texte_epreuve: Texte complet (fallback si pas de demande)
        index: Index de l'exercice
        demande: DemandeCorrection optionnelle (pour utiliser exercices_data)

    Returns:
        dict avec titre et contenu, ou None
    """
    # PRIORITÉ : Utiliser exercices_data si disponible
    if demande and demande.exercices_data:
        try:
            exercices_list = json.loads(demande.exercices_data)
            for ex in exercices_list:
                if ex.get('index') == index:
                    # Retourner le contenu complet si disponible
                    contenu = ex.get('contenu_complet') or ex.get('contenu', '')
                    return {
                        'index': index,
                        'titre': ex.get('titre_complet', ex.get('titre', f"Exercice {index + 1}")),
                        'contenu': contenu,
                        'extrait': ex.get('extrait', ''),
                        'source': 'exercices_data'  # Pour le debug
                    }
        except json.JSONDecodeError as e:
            logger.info(f"❌ [extraire_exercice_par_index] Erreur JSON: {e}")

    # FALLBACK : Extraction traditionnelle
    exercices_data = separer_exercices_avec_titres(texte_epreuve)

    if index < 0 or index >= len(exercices_data):
        return None

    ex_data = exercices_data[index]

    return {
        'index': index,
        'titre': ex_data.get('titre', f"Exercice {index + 1}"),
        'titre_complet': ex_data.get('titre_complet', ''),
        'contenu': ex_data.get('contenu', ''),
        'source': 'extraction_fraiche'  # Pour le debug
    }

def obtenir_liste_exercices(texte_epreuve, avec_preview=False):
    """
    Retourne la liste de tous les exercices détectés.
    Optionnellement avec un aperçu du contenu.
    """
    exercices_data = separer_exercices_avec_titres(texte_epreuve)

    result = []
    for i, ex in enumerate(exercices_data):
        item = {
            'index': i,
            'titre': ex['titre'],
            'titre_complet': ex['titre_complet'],
            'longueur_contenu': len(ex['contenu'])
        }

        if avec_preview:
            # Ajouter un aperçu des premières lignes
            lignes = ex['contenu'].split('\n')[:3]
            preview_text = ' '.join([l[:100] for l in lignes if l.strip()])
            item['preview'] = (preview_text[:200] + '...') if len(preview_text) > 200 else preview_text

        result.append(item)

    return result


# ========== ANALYSE DES SCHÉMAS AVEC DEEPSEEK-CHAT ==========
# À AJOUTER vers la fin de ia_utils.py, avant les @shared_task

def analyser_schema_avec_deepseek_vl(image_path: str, question: str = None) -> dict:
    """
    Analyse un schéma/image avec deepseek-chat et retourne une description structurée.
    Version améliorée avec prompt plus détaillé pour des descriptions riches.
    """
    logger.info(f"🖼️ Analyse schéma avec deepseek-chat: {image_path}")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("❌ DEEPSEEK_API_KEY non configurée")
        return {"description": "", "error": "api_key_missing"}

    try:
        # Encodage de l'image en base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # Vérifier la taille (limite ~5Mo)
        img_size = len(img_b64) * 3 / 4
        if img_size > 5 * 1024 * 1024:  # 5Mo
            logger.warning(f"⚠️ Image trop grande ({img_size / 1024 / 1024:.1f}Mo), redimensionnement")
            from PIL import Image
            import io
            img = Image.open(image_path)
            img.thumbnail((1200, 1200))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", quality=85, optimize=True)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            logger.info(f"✅ Image redimensionnée: {len(img_b64) * 3 / 4 / 1024:.1f}Ko")

        # Construction du prompt amélioré
        if not question:
            question = """
            Analyse ce schéma scientifique en détail et retourne UNIQUEMENT un JSON structuré avec :

            {
                "type_schema": "type précis (plan incliné, circuit électrique, montage optique, graphique, etc.)",
                "description": "description détaillée de ce que représente le schéma",
                "elements_principaux": ["liste", "des", "éléments", "clés"],

                "angles": [
                    {
                        "valeur": 30,
                        "unite": "°",
                        "description": "angle entre quels éléments"
                    }
                ],

                "dimensions": [
                    {
                        "valeur": 5,
                        "unite": "cm",
                        "description": "quelle dimension"
                    }
                ],

                "textes": ["tous", "les", "textes", "lus", "dans", "le", "schéma"],

                "objets": ["cercle", "triangle", "ligne", "fleche", "resistance", "bobine", ...],

                "interpretation": "interprétation scientifique complète (lois, théorèmes, concepts illustrés)"
            }

            RÈGLES IMPORTANTES:
            - Sois extrêmement précis sur les angles et dimensions si visibles
            - Si une valeur exacte n'est pas claire, mets "≈" devant (ex: "≈45°")
            - Décris TOUS les éléments visibles et leurs relations
            - Ne retourne que du JSON valide, pas de texte avant/après
            - Utilise des guillemets doubles, pas simples
            """

        prompt_texte = f"[image]{img_b64}[/image]\n\n{question}"

        # Appel à l'API deepseek-chat
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "user",
                    "content": prompt_texte
                }
            ],
            "temperature": 0.1,
            "max_tokens": 6000,
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"📡 Envoi à deepseek-chat")
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Nettoyer la réponse
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'\s*```', '', content)
            content = content.strip()

            try:
                data = json.loads(content)

                # S'assurer que tous les champs existent
                if "angles" not in data:
                    data["angles"] = []
                if "dimensions" not in data:
                    data["dimensions"] = []
                if "textes" not in data:
                    data["textes"] = []
                if "objets" not in data:
                    data["objets"] = []
                if "elements_principaux" not in data:
                    data["elements_principaux"] = []

                logger.info(f"✅ Analyse schéma réussie")
                logger.info(f"   - Type: {data.get('type_schema', 'inconnu')}")
                logger.info(f"   - Description: {len(data.get('description', ''))} caractères")
                logger.info(f"   - Angles: {len(data.get('angles', []))}")
                logger.info(f"   - Dimensions: {len(data.get('dimensions', []))}")
                logger.info(f"   - Textes: {len(data.get('textes', []))}")

                return data

            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur parsing JSON: {e}")
                # Fallback minimal
                return {
                    "type_schema": "inconnu",
                    "description": content[:500] if content else "Erreur d'analyse",
                    "elements_principaux": [],
                    "angles": [],
                    "dimensions": [],
                    "textes": [],
                    "objets": [],
                    "interpretation": ""
                }
        else:
            logger.error(f"❌ Erreur API: {response.status_code}")
            return {
                "type_schema": "inconnu",
                "description": "",
                "elements_principaux": [],
                "angles": [],
                "dimensions": [],
                "textes": [],
                "objets": [],
                "interpretation": "",
                "error": f"api_error_{response.status_code}"
            }

    except Exception as e:
        logger.error(f"❌ Exception analyser_schema: {e}")
        import traceback
        traceback.print_exc()
        return {
            "type_schema": "inconnu",
            "description": "",
            "elements_principaux": [],
            "angles": [],
            "dimensions": [],
            "textes": [],
            "objets": [],
            "interpretation": "",
            "error": str(e)
        }


def extraire_schemas_du_document(fichier_path: str, demande=None) -> list:
    """
    Extrait et analyse tous les schémas d'un document.
    Version améliorée avec détection intelligente et descriptions riches.

    Args:
        fichier_path: Chemin vers le fichier (PDF ou image)
        demande: Objet DemandeCorrection optionnel

    Returns:
        list: Liste des schémas détectés avec leur page et données
              [{"page": 1, "schemas": [...], "nombre": n}, ...]
    """
    logger.info(f"📑 Extraction des schémas du document: {fichier_path}")

    schemas_detectes = []
    ext = os.path.splitext(fichier_path)[1].lower()
    temp_files = []

    try:
        # === CAS 1: Fichier PDF ===
        if ext == '.pdf':
            from pdf2image import convert_from_path

            logger.info("📄 Conversion PDF en images...")
            images = convert_from_path(fichier_path, dpi=150)

            logger.info(f"   {len(images)} page(s) converties")

            for page_num, image in enumerate(images, 1):
                logger.info(f"   🔍 Analyse page {page_num}/{len(images)}...")

                # Sauvegarder temporairement
                temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_img.close()
                image.save(temp_img.name, 'PNG', quality=85, optimize=True)
                temp_files.append(temp_img.name)

                # Détection rapide si la page contient probablement un schéma
                a_schema = _detection_rapide_schema(temp_img.name)

                if a_schema:
                    # Analyse approfondie
                    schema_data = analyser_schema_avec_deepseek_vl(temp_img.name)

                    if schema_data.get('description') and len(schema_data.get('description', '')) > 30:
                        schemas_detectes.append({
                            "page": page_num,
                            "schemas": [schema_data],
                            "nombre": 1
                        })
                        logger.info(f"   ✅ Schéma détecté page {page_num}: {schema_data.get('type_schema', 'inconnu')}")
                    else:
                        logger.info(f"   ⚠️ Page {page_num}: pas de schéma clair")
                else:
                    logger.info(f"   ⚠️ Page {page_num}: probablement pas de schéma")

                # Petite pause pour éviter surcharge API
                time.sleep(0.3)

        # === CAS 2: Image simple ===
        elif ext in ['.png', '.jpg', '.jpeg']:
            logger.info("🖼️ Analyse image unique")

            schema_data = analyser_schema_avec_deepseek_vl(fichier_path)

            if schema_data.get('description'):
                schemas_detectes.append({
                    "page": 1,
                    "schemas": [schema_data],
                    "nombre": 1
                })
                logger.info(f"✅ Schéma détecté: {schema_data.get('type_schema', 'inconnu')}")

        logger.info(f"📊 Bilan: {len(schemas_detectes)} page(s) avec schémas")
        return schemas_detectes

    except Exception as e:
        logger.error(f"❌ Erreur extraction schémas: {e}")
        import traceback
        traceback.print_exc()
        return []

    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass


def _detection_rapide_schema(image_path: str) -> bool:
    """
    Détection rapide si une image contient probablement un schéma.
    Utilise des heuristiques simples pour éviter d'analyser des pages sans schéma.
    """
    try:

        # Lire l'image
        img = cv2.imread(image_path)
        if img is None:
            return True  # En cas d'erreur, on analyse quand même

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Si l'image est trop petite, probablement pas un schéma détaillé
        if h < 100 or w < 100:
            return False

        # Détection de contours
        edges = cv2.Canny(gray, 50, 150)

        # Détection de lignes
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

        # Compter les lignes
        n_lines = len(lines) if lines is not None else 0

        # Détection de cercles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=5, maxRadius=200)
        has_circles = circles is not None

        # Calculer la densité de contours (pour distinguer texte pur vs schéma)
        cell_size = 50
        cells_with_edges = 0
        n_cells_h = h // cell_size + 1
        n_cells_w = w // cell_size + 1

        for i in range(0, h, cell_size):
            for j in range(0, w, cell_size):
                cell = edges[i:min(i + cell_size, h), j:min(j + cell_size, w)]
                if np.sum(cell) > 1000:
                    cells_with_edges += 1

        density = cells_with_edges / (n_cells_h * n_cells_w) if (n_cells_h * n_cells_w) > 0 else 0

        # Heuristique: un schéma a généralement pas mal de lignes,
        # et une densité de contours modérée (pas trop dense comme du texte)
        est_schema = (n_lines > 8 or has_circles) and 0.1 < density < 0.7

        if est_schema:
            logger.debug(f"   ✅ Détection rapide: schéma probable (lignes={n_lines}, densité={density:.2f})")
        else:
            logger.debug(f"   ⚠️ Détection rapide: probablement pas schéma (lignes={n_lines}, densité={density:.2f})")

        return est_schema

    except Exception as e:
        logger.warning(f"⚠️ Erreur détection rapide: {e}")
        return True  # En cas d'erreur, on analyse quand même


# ============== TÂCHE ASYNCHRONE ==============

@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(demande_id, matiere_id=None):
    from correction.models import DemandeCorrection, SoumissionIA
    from resources.models import Matiere

    try:
        # Récupération de la demande et création de la soumission IA
        demande = DemandeCorrection.objects.get(id=demande_id)
        soumission = SoumissionIA.objects.get(demande=demande)

        # Étape 1 : Extraction du texte brut AVEC VISION
        soumission.statut = 'extraction'
        soumission.progression = 20
        soumission.save()

        donnees_vision_complete = None
        texte_brut = ""

        if demande.fichier:
            # 1) Sauvegarde locale
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, os.path.basename(demande.fichier.name))
            with open(local_path, "wb") as f:
                for chunk in demande.fichier.chunks():
                    f.write(chunk)

            # 2) Appel unique d'analyse scientifique
            analyse_complete = analyser_document_scientifique(local_path, demande)
            donnees_vision_complete = {
                "elements_visuels": analyse_complete.get("elements_visuels", []),
                "formules_latex": analyse_complete.get("formules_latex", []),
                "graphs": analyse_complete.get("graphs", []),
                "angles": analyse_complete.get("angles", []),
                "numbers": analyse_complete.get("numbers", []),
                "structure_exercices": analyse_complete.get("structure_exercices", [])
            }
            texte_brut = analyse_complete.get("texte_complet", "")

            # 3) Nettoyage
            try:
                os.unlink(local_path)
            except:
                pass
        else:
            texte_brut = demande.enonce_texte or ""

        logger.info("📥 TEXTE BRUT AVEC VISION (premiers 500 chars) :")
        logger.info(texte_brut[:500].replace("\n", "\\n"), "...\n")

        # Étape 1b : Extraire les exercices et stocker les données
        exercices_data = separer_exercices_avec_titres(texte_brut)
        logger.info(f"✅ {len(exercices_data)} exercice(s) détecté(s)")

        # Stocker les données des exercices dans la demande
        demande.exercices_data = json.dumps([
            {
                'titre': ex['titre'],
                'titre_complet': ex['titre_complet'],
                'contenu': ex['contenu'][:500] + '...' if len(ex['contenu']) > 500 else ex['contenu']
            }
            for ex in exercices_data
        ])
        demande.save()

        # Étape 2 : Texte final pour l'IA
        texte_enonce = texte_brut

        # Étape 3 : Lancement du traitement IA AVEC DONNÉES VISION
        soumission.statut = 'analyse_ia'
        soumission.progression = 40
        soumission.save()

        matiere = Matiere.objects.get(id=matiere_id) if matiere_id else demande.matiere
        contexte = f"Exercice de {matiere.nom} - {demande.classe.nom if demande.classe else ''}"

        # Étape 4 : Génération graphique (si département scientifique)
        departement = demande.departement
        if is_departement_scientifique(departement):
            print(f"⚗️ Département scientifique : {departement.nom}")
            soumission.statut = 'generation_graphiques'
            soumission.progression = 60
            soumission.save()
        else:
            print(f"⚡ Département non scientifique ({departement.nom if departement else 'inconnu'}), skip graphiques")

        # APPEL AVEC DONNÉES VISION
        corrige_txt, graph_list = generer_corrige_ia_et_graphique(
            texte_enonce,
            contexte,
            matiere=matiere,
            donnees_vision=donnees_vision_complete,
            demande=demande
        )

        # Étape 5 : Génération PDF
        soumission.statut = 'formatage_pdf'
        soumission.progression = 80
        soumission.save()

        pdf_path = generer_pdf_corrige(
            {
                "titre_corrige": contexte,
                "corrige_html": corrige_txt,
                "soumission_id": demande_id,
                "exercices_data": exercices_data  # Passer les données des exercices
            },
            demande_id
        )

        # Débit de crédit
        from abonnement.services import debiter_credit_abonnement
        if not debiter_credit_abonnement(demande.user):
            soumission.statut = 'erreur_credit'
            soumission.save()
            return False

        # Étape 6 : Mise à jour du statut et sauvegarde
        soumission.statut = 'termine'
        soumission.progression = 100
        soumission.resultat_json = {
            'corrige_text': corrige_txt,
            'pdf_url': pdf_path,
            'graphiques': graph_list or [],
            'analyse_vision': donnees_vision_complete,
            'exercices_detectes': len(exercices_data),
            'exercices_titres': [ex['titre'] for ex in exercices_data]
        }
        soumission.save()

        demande.corrigé = corrige_txt
        demande.save()

        logger.info("🎉 TRAITEMENT AVEC VISION TERMINÉ AVEC SUCCÈS!")
        logger.info(f"   Exercices détectés: {len(exercices_data)}")
        for i, ex in enumerate(exercices_data, 1):
            logger.info(f"   {i}. {ex['titre'][:50]}...")

        return True

    except Exception as e:
        logger.info(f"❌ ERREUR dans la tâche IA: {e}")
        import traceback
        traceback.print_exc()
        try:
            soumission.statut = 'erreur'
            soumission.save()
        except:
            pass
        return False


@shared_task(name='correction.ia_utils.generer_corrige_exercice_async',
             bind=True,
             max_retries=3,
             default_retry_delay=60)
def generer_corrige_exercice_async(self, soumission_id):
    """
    Tâche asynchrone pour corriger UN exercice isolé.
    Version robuste avec retries automatiques, timeout gérés et logging détaillé.
    Utilise les données pré-stockées dans exercices_data (texte + schémas)
    """

    task_start = time.time()
    logger.info(f"\n{'=' * 70}")
    logger.info(f"🎯 DÉBUT TÂCHE ASYNC - {datetime.now().strftime('%H:%M:%S')}")
    logger.info(f"   Soumission ID: {soumission_id}")
    logger.info(f"{'=' * 70}")

    try:
        # 1) RÉCUPÉRATION DE LA SOUMISSION
        recovery_start = time.time()
        soum = SoumissionIA.objects.get(id=soumission_id)
        dem = soum.demande
        recovery_time = time.time() - recovery_start

        logger.info(f"✅ Soumission récupérée ({recovery_time:.1f}s)")
        logger.info(f"   - Demande ID: {dem.id}")
        logger.info(f"   - Exercice index: {soum.exercice_index}")
        logger.info(f"   - Département: {dem.departement.nom if dem.departement else 'Non spécifié'}")
        logger.info(f"   - Statut initial: {soum.statut}")

        # Vérification DeepSeek/Mathpix disponible
        deepseek_configure = bool(os.getenv("DEEPSEEK_API_KEY"))
        mathpix_configure = bool(os.getenv("MATHPIX_APP_ID") and os.getenv("MATHPIX_APP_KEY"))

        if dem.departement and is_departement_scientifique(dem.departement):
            print(f"   - Département scientifique → DeepSeek: {'Activé' if deepseek_configure else 'Non configuré'}")

        # 2) MISE À JOUR STATUT IMMÉDIATE
        soum.statut = 'analyse_ia'
        soum.progression = 20
        soum.save()
        logger.info(f"📊 Statut mis à jour: analyse_ia (20%)")

        # 3) RÉCUPÉRATION OPTIMISÉE DU CONTENU DEPUIS exercices_data (AVEC SCHÉMAS)
        extraction_start = time.time()
        fragment = None
        source = "unknown"
        idx = soum.exercice_index or 0
        methode_extraction = "standard"  # Pour le suivi
        donnees_vision_exercice = {}  # ← NOUVEAU : stocker les schémas spécifiques

        # Tentative 1: Récupération depuis exercices_data (avec schémas)
        if dem.exercices_data:
            try:
                exercices_list = json.loads(dem.exercices_data)
                for ex in exercices_list:
                    if ex.get('index') == idx:
                        fragment = ex.get('contenu_complet') or ex.get('contenu', '')
                        source = ex.get('source_extraction', 'exercices_data')
                        methode_extraction = source

                        # ✅ RÉCUPÉRATION DES DONNÉES VISION SPÉCIFIQUES À CET EXERCICE
                        donnees_vision_exercice = {
                            "elements_visuels": ex.get('schemas', []),
                            "formules_latex": ex.get('formules', []),
                            "graphs": ex.get('graphs', []),
                            "angles": ex.get('angles', []),
                            "numbers": ex.get('numbers', [])
                        }

                        logger.info(f"✅ Contenu récupéré depuis exercices_data")
                        logger.info(f"   - Source: {source}")
                        logger.info(f"   - Longueur: {len(fragment)} caractères")
                        logger.info(f"   - Schémas: {len(donnees_vision_exercice['elements_visuels'])}")
                        logger.info(f"   - Formules: {len(donnees_vision_exercice['formules_latex'])}")

                        # Afficher les schémas pour debug
                        if donnees_vision_exercice['elements_visuels']:
                            for i, s in enumerate(donnees_vision_exercice['elements_visuels'][:2]):
                                print(
                                    f"      Schéma {i + 1}: {s.get('type', 'inconnu')} - {s.get('description', '')[:50]}...")

                        break
            except json.JSONDecodeError as e:
                logger.info(f"⚠️  Erreur JSON exercices_data: {e}")

        # Tentative 2: Fallback extraction fichier AVEC DEEPSEEK CONDITIONNEL
        if not fragment and dem.fichier:
            logger.info(f"🔄 Fallback: extraction depuis fichier")
            try:
                # Extraction complète avec DeepSeek (si département scientifique)
                analyse_complete = extraire_texte_fichier(dem.fichier, dem)

                texte_complet = analyse_complete.get("texte_complet", "") if isinstance(analyse_complete,
                                                                                        dict) else analyse_complete

                if texte_complet and len(texte_complet.strip()) > 50:
                    exercices_data = separer_exercices_avec_titres(texte_complet)

                    if idx >= len(exercices_data):
                        print(f"⚠️  Index {idx} hors limites, ajustement")
                        idx = len(exercices_data) - 1 if exercices_data else 0

                    ex_data = exercices_data[idx] if exercices_data else {}
                    fragment = ex_data.get('contenu', '')
                    source = "extraction_fraiche"

                    # Récupérer les exercices structurés si disponibles
                    exercices_struct = analyse_complete.get("exercices_struct", []) if isinstance(analyse_complete,
                                                                                                  dict) else []

                    if exercices_struct and idx < len(exercices_struct):
                        ex_vision = exercices_struct[idx]
                        donnees_vision_exercice = {
                            "elements_visuels": ex_vision.get("schemas", []),
                            "formules_latex": ex_vision.get("formules", []),
                            "graphs": ex_vision.get("graphs", []),
                            "angles": ex_vision.get("angles", []),
                            "numbers": ex_vision.get("numbers", [])
                        }

                    logger.info(f"✅ Contenu extrait via fallback")
                    logger.info(f"   - Source: {source}")
                    logger.info(f"   - Longueur: {len(fragment)} caractères")
                    logger.info(f"   - Schémas: {len(donnees_vision_exercice.get('elements_visuels', []))}")

                    # Enregistrer la méthode d'extraction
                    if isinstance(analyse_complete, dict):
                        methode_extraction = analyse_complete.get('source_extraction', 'standard')
                    else:
                        methode_extraction = "standard"

                    logger.info(f"   - Méthode extraction: {methode_extraction}")

                else:
                    logger.info(f"⚠️  Texte extrait trop court: {len(texte_complet or '')} caractères")
            except Exception as e:
                logger.info(f"❌ Erreur extraction fichier: {type(e).__name__}: {str(e)[:100]}")

        extraction_time = time.time() - extraction_start

        # 4) VALIDATION DU FRAGMENT
        if not fragment or len(fragment.strip()) < 20:
            error_msg = f"Fragment invalide (longueur: {len(fragment or '')} chars, source: {source})"
            logger.info(f"❌ {error_msg}")
            logger.info(f"⏱️  Temps extraction: {extraction_time:.1f}s")

            # Mise à jour statut erreur
            soum.statut = 'erreur'
            soum.save()

            raise ValueError(error_msg)

        logger.info(f"✅ Fragment validé")
        logger.info(f"⏱️  Extraction totale: {extraction_time:.1f}s")
        logger.info(f"📝 Début fragment: {fragment[:100].replace(chr(10), ' ')}...")
        logger.info(f"🔧 Méthode extraction: {methode_extraction}")

        # 5) PRÉPARATION CONTEXTE IA
        mat = dem.matiere if dem.matiere else Matiere.objects.first()
        titre_exercice = f"Exercice {idx + 1}"

        # Récupération titre depuis exercices_data si disponible
        if dem.exercices_data and source == "exercices_data":
            try:
                exercices_list = json.loads(dem.exercices_data)
                for ex in exercices_list:
                    if ex.get('index') == idx:
                        titre_exercice = ex.get('titre_complet', ex.get('titre', titre_exercice))
                        break
            except:
                pass

        contexte = f"Exercice de {mat.nom if mat else 'Matière'} – {titre_exercice}"
        if dem.departement:
            contexte += f" – Département {dem.departement.nom}"
        logger.info(f"🎯 Contexte IA: {contexte}")

        # 6) GÉNÉRATION IA AVEC GESTION D'ERREURS ROBUSTE
        ia_start = time.time()
        logger.info(f"\n{'─' * 40}")
        logger.info(f"🤖 DÉBUT GÉNÉRATION IA AVEC SCHÉMAS")
        logger.info(f"{'─' * 40}")

        try:
            # Appel IA avec les données vision pré-filtrées
            corrige_txt, _ = generer_corrige_ia_et_graphique(
                texte_enonce=fragment,
                contexte=contexte,
                matiere=mat,
                donnees_vision=donnees_vision_exercice if donnees_vision_exercice else None,  # ← Données filtrées !
                demande=dem
            )

            ia_time = time.time() - ia_start
            logger.info(f"✅ Génération IA réussie ({ia_time:.1f}s)")
            logger.info(f"📝 Longueur corrigé: {len(corrige_txt or '')} caractères")

            # Validation basique du corrigé
            if not corrige_txt or len(corrige_txt.strip()) < 50:
                error_msg = f"Corrigé trop court: {len(corrige_txt or '')} caractères"
                print(f"⚠️  {error_msg}")
                raise ValueError(error_msg)

        except Exception as ia_error:
            ia_time = time.time() - ia_start
            logger.error(f"\n❌ ÉCHEC GÉNÉRATION IA ({ia_time:.1f}s)")
            logger.info(f"   Type erreur: {type(ia_error).__name__}")
            logger.info(f"   Message: {str(ia_error)[:200]}")
            logger.info(f"{'─' * 40}")

            # Retry automatique après délai
            logger.info(f"🔄 Retry automatique dans 60s...")
            raise self.retry(exc=ia_error, countdown=60)

        # 7) MISE À JOUR STATUT INTERMÉDIAIRE
        soum.statut = 'formatage_pdf'
        soum.progression = 60
        soum.save()
        logger.info(f"📊 Statut mis à jour: formatage_pdf (60%)")

        # 8) GÉNÉRATION PDF
        pdf_start = time.time()
        logger.info(f"\n{'─' * 40}")
        logger.info(f"📄 DÉBUT GÉNÉRATION PDF")
        logger.info(f"{'─' * 40}")

        try:
            pdf_url = generer_pdf_corrige(
                {
                    "titre_corrige": contexte,
                    "corrige_html": corrige_txt,
                    "soumission_id": soum.id,
                    "titre_exercice": titre_exercice,
                    "methode_extraction": methode_extraction  # Ajout pour suivi
                },
                soum.id
            )

            pdf_time = time.time() - pdf_start
            print(f"✅ Génération PDF réussie ({pdf_time:.1f}s)")
            print(f"📎 URL PDF: {pdf_url}")

        except Exception as pdf_error:
            pdf_time = time.time() - pdf_start
            logger.error(f"❌ Échec génération PDF ({pdf_time:.1f}s)")
            logger.info(f"   Erreur: {type(pdf_error).__name__}: {str(pdf_error)[:200]}")
            raise pdf_error

        # 9) DÉBIT CRÉDIT
        debit_start = time.time()
        logger.info(f"\n{'─' * 40}")
        logger.info(f"💳 DÉBIT CRÉDIT UTILISATEUR")
        logger.info(f"{'─' * 40}")

        try:
            if not debiter_credit_abonnement(dem.user):
                error_msg = "Crédits insuffisants"
                logger.info(f"❌ {error_msg}")

                soum.statut = 'erreur_credit'
                soum.save()

                raise ValueError(error_msg)

            debit_time = time.time() - debit_start
            logger.info(f"✅ Débit crédit réussi ({debit_time:.1f}s)")

        except Exception as debit_error:
            print(f"❌ Erreur débit crédit: {type(debit_error).__name__}")
            raise debit_error

        # 10) CRÉATION CORRIGEPARTIEL
        corrige_start = time.time()
        logger.info(f"\n{'─' * 40}")
        logger.info(f"📁 CRÉATION CORRIGEPARTIEL")
        logger.info(f"{'─' * 40}")

        try:
            # Préparation titre
            titre_reel = titre_exercice
            if len(titre_reel) > 200:
                titre_reel = titre_reel[:197] + "..."

            # Récupération chemin PDF
            pdf_relative_path = pdf_url.replace(settings.MEDIA_URL, '')
            pdf_absolute_path = os.path.join(settings.MEDIA_ROOT, pdf_relative_path)

            if not os.path.exists(pdf_absolute_path):
                error_msg = f"Fichier PDF non trouvé: {pdf_absolute_path}"
                logger.error(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)

            # Création CorrigePartiel avec info d'extraction
            with open(pdf_absolute_path, 'rb') as f:
                corrige = CorrigePartiel.objects.create(
                    soumission=soum,
                    titre_exercice=titre_reel,
                )
                corrige.fichier_pdf.save(
                    f"corrige_{dem.id}_ex{idx + 1}_{soum.id}_{int(time.time())}.pdf",
                    File(f)
                )
                corrige.save()

            corrige_time = time.time() - corrige_start
            logger.info(f"✅ CorrigePartiel créé ({corrige_time:.1f}s)")
            logger.info(f"   - ID: {corrige.id}")
            logger.info(f"   - Titre: {titre_reel}")
            logger.info(f"   - Méthode extraction: {methode_extraction}")

        except Exception as corrige_error:
            corrige_time = time.time() - corrige_start
            logger.info(f"❌ Erreur création CorrigePartiel ({corrige_time:.1f}s)")
            logger.info(f"   Erreur: {type(corrige_error).__name__}: {str(corrige_error)[:200]}")
            raise corrige_error

        # 11) FINALISATION
        total_time = time.time() - task_start

        # Préparer le résultat JSON avec les informations des schémas
        resultat_json = {
            "exercice_index": idx,
            "exercice_titre": titre_reel,
            "corrige_text": corrige_txt,
            "pdf_url": pdf_url,
            "timestamp": datetime.now().isoformat(),
            "processing_time": total_time,
            "source_content": source,
            "methode_extraction": methode_extraction,
            "departement": dem.departement.nom if dem.departement else None,
            "schemas_utilises": len(
                donnees_vision_exercice.get('elements_visuels', [])) if donnees_vision_exercice else 0
        }

        soum.statut = 'termine'
        soum.progression = 100
        soum.resultat_json = resultat_json
        soum.save()

        logger.info(f"\n{'=' * 70}")
        logger.info(f"✅ TÂCHE TERMINÉE AVEC SUCCÈS!")
        logger.info(f"   Temps total: {total_time:.1f}s")
        logger.info(f"   Exercice: {titre_reel}")
        logger.info(f"   Source contenu: {source}")
        logger.info(f"   Méthode extraction: {methode_extraction}")
        logger.info(f"   Schémas utilisés: {resultat_json['schemas_utilises']}")
        logger.info(f"   Département: {dem.departement.nom if dem.departement else 'Non spécifié'}")
        logger.info(f"   Corrigé: {len(corrige_txt)} caractères")
        logger.info(f"   {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'=' * 70}")

        return True

    except Exception as e:
        total_time = time.time() - task_start

        logger.info(f"\n{'=' * 70}")
        logger.error(f"❌ ERREUR CRITIQUE DANS LA TÂCHE")
        logger.info(f"   Temps écoulé: {total_time:.1f}s")
        logger.info(f"   Type erreur: {type(e).__name__}")
        logger.info(f"   Message: {str(e)[:300]}")
        logger.info(f"   Soumission ID: {soumission_id}")
        logger.info(f"   {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'=' * 70}")

        # Log détaillé de l'erreur
        import traceback
        error_details = traceback.format_exc()
        logger.info(f"\n📋 TRACEBACK COMPLET:")
        logger.info(error_details[:1000])  # Limité pour éviter logs trop longs

        # Mise à jour statut erreur si possible
        try:
            soum = SoumissionIA.objects.get(id=soumission_id)
            soum.statut = 'erreur'
            soum.save()
        except:
            pass

        # Si c'est une erreur réseau/timeout, on retry
        error_type = type(e).__name__
        if error_type in ['Timeout', 'ConnectionError', 'ReadTimeout', 'ConnectTimeout']:
            logger.error(f"🔄 Erreur réseau détectée, retry automatique...")
            raise self.retry(exc=e, countdown=120)

        # Pour les autres erreurs, on ne retry pas
        return False