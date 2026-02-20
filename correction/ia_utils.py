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

# ── NOUVELLE FONCTION : Analyse scientifique avancée ────

def analyser_document_scientifique(fichier_path: str, demande=None) -> dict:
    """
    Analyse scientifique avancée avec choix du moteur selon département.
    Pour les départements scientifiques : Mathpix (formules LaTeX précises)
    Sinon : OCR standard + DeepSeek Vision
    """
    logger.info(f"🔍 Début analyse scientifique pour {fichier_path}")

    # 1) DÉTECTION DU DÉPARTEMENT POUR CHOIX DE LA MÉTHODE
    use_mathpix = False
    dept_nom = "inconnu"

    if demande and demande.departement:
        dept_nom = demande.departement.nom
        use_mathpix = is_departement_scientifique(demande.departement)
        logger.info(f"📊 Département '{dept_nom}' → Mathpix = {use_mathpix}")

    # 2) EXTRACTION AVEC MATHPIX POUR DÉPARTEMENTS SCIENTIFIQUES
    if use_mathpix:
        logger.info("🧮 Extraction avec Mathpix (département scientifique)")

        # Vérifier que la configuration Mathpix existe
        if not os.getenv("MATHPIX_APP_ID") or not os.getenv("MATHPIX_APP_KEY"):
            logger.warning("⚠️ Configuration Mathpix manquante, fallback standard")
            use_mathpix = False
        else:
            resultat_mathpix = extraire_avec_mathpix(fichier_path)

            if resultat_mathpix.get("text") and len(resultat_mathpix["text"]) > 100:
                logger.info(f"✅ Mathpix réussi: {len(resultat_mathpix['text'])} caractères, "
                            f"{len(resultat_mathpix.get('latex_blocks', []))} formules LaTeX")

                return {
                    "texte_complet": resultat_mathpix["text"],
                    "elements_visuels": [],
                    "formules_latex": resultat_mathpix.get("latex_blocks", []),
                    "graphs": [],
                    "angles": [],
                    "numbers": [],
                    "structure_exercices": [],
                    "source_extraction": "mathpix",
                    "departement": dept_nom
                }
            else:
                logger.warning("⚠️ Mathpix échec ou résultat trop court (<100 chars), fallback standard")
                use_mathpix = False

    # 3) FALLBACK: ANALYSE STANDARD (OCR + DeepSeek Vision)
    logger.info("🔤 Extraction standard (OCR + DeepSeek Vision)")

    # 3a) OCR fallback pour avoir un premier texte
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

        else:
            raise ValueError(f"Format non supporté pour OCR: {fichier_path}")

    except Exception as e:
        logger.error(f"❌ Erreur pendant OCR/PDF: {e}")
        # On continue avec texte_ocr vide

    # 3b) Appel deepseek-vl2 pour tout : texte + schémas + JSON
    try:
        vision_json = call_deepseek_vision(fichier_path)

        # 3c) Texte complet : fallback sur OCR si résultat trop court
        texte_json = vision_json.get("text", "") or ""
        if len(texte_json) < 50:
            texte_json = texte_ocr

        # 3d) Récupération des blocs
        captions = vision_json.get("captions", [])
        latex_blocks = vision_json.get("latex_blocks", [])
        graphs = vision_json.get("graphs", [])
        angles = vision_json.get("angles", [])
        numbers = vision_json.get("numbers", [])
        struct_exos = vision_json.get("structure_exercices", [])

        logger.info(f"✅ DeepSeek Vision OK : texte {len(texte_json)} chars, "
                    f"{len(captions)} schémas, {len(latex_blocks)} formules, "
                    f"{len(angles)} angles, {len(numbers)} nombres")

        return {
            "texte_complet": texte_json,
            "elements_visuels": captions,
            "formules_latex": latex_blocks,
            "graphs": graphs,
            "angles": angles,
            "numbers": numbers,
            "structure_exercices": struct_exos,
            "source_extraction": "standard",
            "departement": dept_nom
        }

    except Exception as e:
        logger.error(f"❌ Erreur DeepSeek Vision: {e}")
        # Fallback minimal
        return {
            "texte_complet": texte_ocr,
            "elements_visuels": [],
            "formules_latex": [],
            "graphs": [],
            "angles": [],
            "numbers": [],
            "structure_exercices": [],
            "source_extraction": "fallback_ocr",
            "departement": dept_nom
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
    Génère le corrigé pour un seul exercice.
    Version améliorée avec intégration des descriptions de schémas.
    """
    import time
    from datetime import datetime

    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"🤖 DÉBUT generer_corrige_par_exercice - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 70}")

    if demande:
        print(f"📋 Informations demande:")
        print(f"   - ID: {demande.id}")
        print(f"   - Matière: {demande.matiere.nom if demande.matiere else 'Non spécifiée'}")
        print(f"   - Classe: {demande.classe.nom if demande.classe else 'Non spécifiée'}")
        print(f"   - Département: {demande.departement.nom if demande.departement else 'Non spécifiée'}")

    print(f"📊 Métriques:")
    print(f"   - Longueur exercice: {len(texte_exercice)} caractères")
    print(f"   - Contexte: {contexte}")
    print(f"   - Données vision: {'PRÉSENTES' if donnees_vision else 'ABSENTES'}")

    try:
        # 1) RÉCUPÉRATION DU PROMPT MÉTIER
        promptia = get_best_promptia(demande)
        print(f"✅ Prompt trouvé: {'OUI' if promptia else 'NON (DEFAULT)'}")

        # 2) CONSTRUCTION DES MESSAGES
        msg_system, msg_user = build_promptia_messages(promptia, contexte)

        # 3) ENRICHISSEMENT AVEC TEXTE + SCHÉMAS
        user_blocks = [
            msg_user["content"],
            "----- EXERCICE À CORRIGER -----",
            texte_exercice.strip()
        ]

        vision_elements_count = 0

        # ========== NOUVEAU: RÉCUPÉRATION DES SCHÉMAS DEPUIS LA DEMANDE ==========
        schemas_exercice = []
        if demande and demande.exercices_data:
            try:
                exercices_list = json.loads(demande.exercices_data)
                # Récupérer l'index depuis la soumission si disponible
                idx_courant = getattr(demande, 'exercice_index_courant', 0)

                for ex in exercices_list:
                    if ex.get('index') == idx_courant:
                        schemas_exercice = ex.get('schemas', [])
                        break
            except Exception as e:
                print(f"⚠️ Erreur lecture schémas: {e}")

        # Ajouter les descriptions des schémas au prompt
        if schemas_exercice:
            user_blocks.append("")
            user_blocks.append("----- SCHÉMAS / CROQUIS DÉTECTÉS DANS L'EXERCICE -----")

            for idx, schema in enumerate(schemas_exercice, 1):
                if not isinstance(schema, dict):
                    continue

                user_blocks.append(f"\n--- SCHÉMA {idx} ---")

                # Description générale
                if schema.get('description'):
                    user_blocks.append(f"Description: {schema['description']}")

                # Angles détectés
                if schema.get('angles') and len(schema['angles']) > 0:
                    angles_text = []
                    for angle in schema['angles']:
                        if isinstance(angle, dict):
                            val = angle.get('valeur', '?')
                            unite = angle.get('unite', '°')
                            desc = angle.get('description', '')
                            if desc:
                                angles_text.append(f"{val}{unite} ({desc})")
                            else:
                                angles_text.append(f"{val}{unite}")
                    if angles_text:
                        user_blocks.append(f"Angles: {', '.join(angles_text)}")

                # Dimensions
                if schema.get('dimensions') and len(schema['dimensions']) > 0:
                    dims_text = []
                    for dim in schema['dimensions']:
                        if isinstance(dim, dict):
                            val = dim.get('valeur', '?')
                            unite = dim.get('unite', '')
                            desc = dim.get('description', '')
                            if desc:
                                dims_text.append(f"{val}{unite} ({desc})")
                            else:
                                dims_text.append(f"{val}{unite}")
                    if dims_text:
                        user_blocks.append(f"Dimensions: {', '.join(dims_text)}")

                # Textes/Annotations
                if schema.get('textes') and len(schema['textes']) > 0:
                    textes = schema['textes']
                    if isinstance(textes, list):
                        user_blocks.append(f"Textes/Annotations: {' - '.join(textes[:5])}")
                        if len(textes) > 5:
                            user_blocks.append(f"  ... et {len(textes) - 5} autres annotations")

                # Objets géométriques
                if schema.get('objets') and len(schema['objets']) > 0:
                    objets = schema['objets']
                    if isinstance(objets, list):
                        user_blocks.append(f"Éléments géométriques: {', '.join(objets[:8])}")
                        if len(objets) > 8:
                            user_blocks.append(f"  ... et {len(objets) - 8} autres éléments")

                # Interprétation scientifique
                if schema.get('interpretation'):
                    user_blocks.append(f"Interprétation: {schema['interpretation']}")

                vision_elements_count += 1

        # Anciennes données vision (formules, etc.)
        if donnees_vision:
            if donnees_vision.get("elements_visuels"):
                elements = donnees_vision["elements_visuels"]
                user_blocks.append(f"----- SCHÉMAS IDENTIFIÉS ({len(elements)}) -----")
                for element in elements[:5]:
                    desc = element.get("description", "")
                    user_blocks.append(f"- {desc}")
                    vision_elements_count += 1

            if donnees_vision.get("formules_latex"):
                formules = donnees_vision["formules_latex"]
                user_blocks.append(f"----- FORMULES DÉTECTÉES ({len(formules)}) -----")
                for formule in formules[:10]:
                    user_blocks.append(f"- {formule}")
                    vision_elements_count += 1

        # Assembler le message final
        msg_user["content"] = "\n\n".join(user_blocks)

        print(f"\n📦 Message IA construit:")
        print(f"   - Longueur système: {len(msg_system['content'])} caractères")
        print(f"   - Longueur utilisateur: {len(msg_user['content'])} caractères")
        print(f"   - Éléments visuels (schémas inclus): {vision_elements_count}")
        print(f"   - Schémas de l'exercice: {len(schemas_exercice)}")

        # 4) APPEL API (identique à avant)
        api_url = "https://api.deepseek.com/v1/chat/completions"
        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            error_msg = "❌ API KEY DeepSeek non configurée"
            print(f"\n{error_msg}")
            return error_msg, None

        data = {
            "model": "deepseek-chat",
            "messages": [msg_system, msg_user],
            "temperature": 0.1,
            "max_tokens": 8192,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CIS-Education/1.0"
        }

        # 5) APPEL AVEC RETRIES
        output = None
        final_response_data = None
        last_error = None

        for tentative in range(3):
            print(f"\n   🔄 TENTATIVE {tentative + 1}/3")
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data,
                    timeout=120,
                    verify=True
                )

                if response.status_code == 200:
                    response_data = response.json()
                    output = response_data['choices'][0]['message']['content']
                    final_response_data = response_data

                    if verifier_qualite_corrige(output, texte_exercice):
                        print(f"   ✅ Qualité validée")
                        break
                    else:
                        print(f"   🔄 Qualité insuffisante, nouvelle tentative...")
                        last_error = "Qualité insuffisante"
                        time.sleep(2 * (tentative + 1))
                else:
                    last_error = f"HTTP {response.status_code}"
                    time.sleep(5 * (tentative + 1))

            except Exception as e:
                last_error = str(e)
                time.sleep(5 * (tentative + 1))

        if not output:
            return f"Erreur IA: {last_error}", None

        # 6) POST-TRAITEMENT
        output = flatten_multiline_latex_blocks(output)
        output_structured = format_corrige_pdf_structure(output)

        # Extraction des graphiques
        json_blocks = extract_json_blocks(output_structured)
        graph_list = []

        if json_blocks:
            json_blocks = sorted(json_blocks, key=lambda x: x[1], reverse=True)
            for idx, (graph_dict, start, end) in enumerate(json_blocks, 1):
                try:
                    output_name = f"graphique_{idx}_{int(time.time())}.png"
                    img_path = tracer_graphique(graph_dict, output_name)
                    if img_path:
                        abs_path = os.path.join(settings.MEDIA_ROOT, img_path)
                        img_tag = f'<img src="file://{abs_path}" alt="Graphique {idx}" style="max-width:100%;margin:10px 0;" />'
                        output_structured = output_structured[:start] + img_tag + output_structured[end:]
                        graph_list.append(graph_dict)
                except:
                    continue

        total_time = time.time() - start_time
        print(f"\n✅ SUCCÈS - Temps: {total_time:.1f}s")
        print(f"   Corrigé: {len(output_structured)} caractères")
        print(f"   Schémas intégrés: {len(schemas_exercice)}")

        return output_structured.strip(), graph_list

    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ ERREUR: {type(e).__name__}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return f"Erreur traitement IA: {str(e)[:200]}", None


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
    Extraction robuste avec support Mathpix conditionnel
    """
    if not fichier_field:
        return ""

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

        logger.info(f"📄 Extraction terminée: {len(texte)} caractères "
                    f"(source: {analyse.get('source_extraction', 'inconnu')})")

        # Stocker la méthode d'extraction dans la demande si disponible
        if demande and hasattr(demande, 'methode_extraction'):
            demande.methode_extraction = analyse.get('source_extraction', 'standard')
            demande.save()

    except Exception as e:
        logger.error(f"❌ Analyse échouée: {e}")
        texte = ""

    # Nettoyage
    try:
        os.unlink(local_path)
    except:
        pass

    return texte.strip()

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
    print("🎯 Traitement DIRECT avec analyse vision")
    print("\n[DEBUG] --> generer_corrige_direct called avec demande:", getattr(demande, 'id', None),
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
                                    demande=None, donnees_vision=None):
    """
    Version SIMPLIFIÉE pour les exercices uniques.
    Appelle directement generer_corrige_par_exercice sans logique de décision.
    """
    print("\n" + "=" * 60)
    print("🚀 DÉBUT TRAITEMENT IA POUR EXERCICE UNIQUE")
    print("=" * 60)
    print(f"📏 Longueur texte: {len(texte_enonce)} caractères")
    print(f"📝 Contexte: {contexte}")

    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    # Données vision
    if donnees_vision:
        print(f"🔬 Données vision disponibles:")
        print(f"   - Éléments visuels: {len(donnees_vision.get('elements_visuels', []))}")
        print(f"   - Formules LaTeX: {len(donnees_vision.get('formules_latex', []))}")

    # POUR LES EXERCICES UNIQUES : APPEL DIRECT
    print("🎯 Appel direct à generer_corrige_par_exercice")

    return generer_corrige_par_exercice(
        texte_exercice=texte_enonce,
        contexte=contexte,
        matiere=matiere,
        donnees_vision=donnees_vision,
        demande=demande
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
            print(f"❌ [extraire_exercice_par_index] Erreur JSON: {e}")

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
    Analyse un schéma/image avec deepseek-chat (capacités vision) et retourne une description structurée.

    Args:
        image_path: Chemin vers l'image (fichier temporaire)
        question: Question spécifique à poser (None pour description générale)

    Returns:
        dict: {
            'description': str,  # Description générale du schéma
            'angles': list,      # Angles détectés [{"valeur": 30, "unite": "°", "description": "..."}]
            'dimensions': list,  # Dimensions [{"valeur": 5, "unite": "cm", "description": "..."}]
            'textes': list,      # Textes/légendes lus
            'objets': list,      # Types d'objets géométriques détectés
            'interpretation': str # Interprétation scientifique
        }
    """
    logger.info(f"🖼️ Analyse schéma avec deepseek-chat: {image_path}")

    # Vérifier que la clé API est configurée
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("❌ DEEPSEEK_API_KEY non configurée")
        return {"description": "", "error": "api_key_missing"}

    try:
        # Encodage de l'image en base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # Vérifier la taille (limite ~5Mo)
        img_size = len(img_b64) * 3 / 4  # Approximation taille réelle en octets
        if img_size > 5 * 1024 * 1024:  # 5Mo
            logger.warning(f"⚠️ Image trop grande ({img_size / 1024 / 1024:.1f}Mo), redimensionnement automatique")
            # Redimensionner l'image si trop grande
            from PIL import Image
            import io

            # Ouvrir et redimensionner
            img = Image.open(image_path)
            img.thumbnail((1200, 1200))  # Max 1200px de côté

            # Ré-encoder en base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", quality=85, optimize=True)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            logger.info(f"✅ Image redimensionnée: {len(img_b64) * 3 / 4 / 1024:.1f}Ko")

        # Construction du prompt avec la balise [image]
        if not question:
            question = """
            Analyse ce schéma/croquis en détail et retourne UNIQUEMENT un JSON structuré avec :
            {
                "description": "description détaillée du schéma (ce qu'il représente, les éléments principaux)",
                "angles": [{"valeur": 30, "unite": "°", "description": "angle entre quels éléments"}],
                "dimensions": [{"valeur": 5, "unite": "cm", "description": "quelle dimension/mesure"}],
                "textes": ["texte1", "texte2"],  # Tous les textes/légendes/annotations lus
                "objets": ["cercle", "triangle", "ligne", "fleche", ...],  # Types d'objets géométriques
                "interpretation": "interprétation scientifique/mathématique du schéma (loi, théorème, concept)"
            }

            RÈGLES IMPORTANTES:
            - Sois extrêmement précis sur les angles et dimensions si visibles
            - Si une valeur exacte n'est pas claire, mets "≈" devant (ex: "≈45°")
            - Ne retourne que du JSON valide, pas de texte avant/après
            - Utilise des guillemets doubles, pas simples
            """

        # IMPORTANT: Format correct pour deepseek-chat - utilisation de la balise [image] dans le texte
        prompt_texte = f"[image]{img_b64}[/image]\n\n{question}"

        # Appel à l'API deepseek-chat
        payload = {
            "model": "deepseek-chat",  # ← CHANGEMENT: deepseek-vl → deepseek-chat
            "messages": [
                {
                    "role": "user",
                    "content": prompt_texte  # ← CHANGEMENT: plus de tableau, un simple string avec balise [image]
                }
            ],
            "temperature": 0.1,  # Bas pour précision
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}  # Forcer JSON
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"📡 Envoi à deepseek-chat (taille image: {len(img_b64) / 1024:.1f}Ko)")
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",  # Même URL
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Nettoyer la réponse (enlever les ```json si présents)
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'\s*```', '', content)
            content = content.strip()

            # Tenter de parser le JSON
            try:
                data = json.loads(content)

                # Logging des résultats
                logger.info(f"✅ Analyse schéma réussie")
                logger.info(f"   - Description: {len(data.get('description', ''))} caractères")
                logger.info(f"   - Angles: {len(data.get('angles', []))}")
                logger.info(f"   - Dimensions: {len(data.get('dimensions', []))}")
                logger.info(f"   - Textes: {len(data.get('textes', []))}")
                logger.info(f"   - Objets: {len(data.get('objets', []))}")

                return data

            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur parsing JSON: {e}")
                logger.error(f"Contenu reçu: {content[:500]}")

                # Fallback: extraire avec regex si possible
                description_match = re.search(r'"description"\s*:\s*"([^"]+)"', content)
                description = description_match.group(1) if description_match else "Erreur d'analyse"

                return {
                    "description": description,
                    "angles": [],
                    "dimensions": [],
                    "textes": [],
                    "objets": [],
                    "interpretation": "",
                    "raw_response": content[:1000],
                    "error": "json_parse_error"
                }
        else:
            logger.error(f"❌ Erreur API deepseek-chat: {response.status_code} - {response.text[:200]}")
            return {
                "description": "",
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
            "description": "",
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
    Pour les PDF multi-pages, convertit chaque page en image et l'analyse.

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
    temp_files = []  # Pour nettoyage

    try:
        # === CAS 1: Fichier PDF ===
        if ext == '.pdf':
            from pdf2image import convert_from_path

            logger.info("📄 Conversion PDF en images...")
            # Convertir avec résolution modérée pour économiser
            images = convert_from_path(fichier_path, dpi=150)  # 150 dpi suffisant pour l'analyse

            logger.info(f"   {len(images)} page(s) converties")

            for page_num, image in enumerate(images, 1):
                logger.info(f"   🔍 Analyse page {page_num}/{len(images)}...")

                # Sauvegarder temporairement
                temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_img.close()
                image.save(temp_img.name, 'PNG', quality=85, optimize=True)
                temp_files.append(temp_img.name)

                # Question spécifique pour détecter les schémas
                question = """
                Analyse cette page et réponds UNIQUEMENT par JSON:

                {
                    "a_des_schemas": true/false,
                    "schemas": [
                        {
                            "description": "description du schéma",
                            "angles": [...],
                            "dimensions": [...],
                            "textes": [...],
                            "objets": [...],
                            "interpretation": "..."
                        }
                    ],
                    "nombre_schemas": 0
                }

                Ne retourne que le JSON, rien d'autre.
                """

                schema_data = analyser_schema_avec_deepseek_vl(temp_img.name, question)

                # Vérifier si des schémas ont été détectés
                a_des_schemas = False
                if isinstance(schema_data, dict):
                    # Différents formats possibles selon la réponse
                    if schema_data.get('a_des_schemas') is True:
                        a_des_schemas = True
                        schemas_page = schema_data.get('schemas', [])
                    elif schema_data.get('description') and len(schema_data.get('description', '')) > 20:
                        a_des_schemas = True
                        schemas_page = [schema_data]  # Un seul schéma
                    else:
                        schemas_page = []

                    if a_des_schemas:
                        schemas_detectes.append({
                            "page": page_num,
                            "schemas": schemas_page,
                            "nombre": len(schemas_page)
                        })
                        logger.info(f"   ✅ {len(schemas_page)} schéma(s) détecté(s) page {page_num}")

                # Petite pause pour éviter de surcharger l'API
                time.sleep(0.5)

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
                logger.info(f"✅ Schéma détecté dans l'image")

        logger.info(f"📊 Bilan: {len(schemas_detectes)} page(s) avec schémas")
        return schemas_detectes

    except Exception as e:
        logger.error(f"❌ Erreur extraction schémas: {e}")
        import traceback
        traceback.print_exc()
        return []

    finally:
        # Nettoyage des fichiers temporaires
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

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

        print("📥 TEXTE BRUT AVEC VISION (premiers 500 chars) :")
        print(texte_brut[:500].replace("\n", "\\n"), "...\n")

        # Étape 1b : Extraire les exercices et stocker les données
        exercices_data = separer_exercices_avec_titres(texte_brut)
        print(f"✅ {len(exercices_data)} exercice(s) détecté(s)")

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

        from .pdf_utils import generer_pdf_corrige
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

        print("🎉 TRAITEMENT AVEC VISION TERMINÉ AVEC SUCCÈS!")
        print(f"   Exercices détectés: {len(exercices_data)}")
        for i, ex in enumerate(exercices_data, 1):
            print(f"   {i}. {ex['titre'][:50]}...")

        return True

    except Exception as e:
        print(f"❌ ERREUR dans la tâche IA: {e}")
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
    Support Mathpix pour départements scientifiques.
    """

    task_start = time.time()
    print(f"\n{'=' * 70}")
    print(f"🎯 DÉBUT TÂCHE ASYNC - {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Soumission ID: {soumission_id}")
    print(f"{'=' * 70}")

    try:
        # 1) RÉCUPÉRATION DE LA SOUMISSION
        recovery_start = time.time()
        soum = SoumissionIA.objects.get(id=soumission_id)
        dem = soum.demande
        recovery_time = time.time() - recovery_start

        print(f"✅ Soumission récupérée ({recovery_time:.1f}s)")
        print(f"   - Demande ID: {dem.id}")
        print(f"   - Exercice index: {soum.exercice_index}")
        print(f"   - Département: {dem.departement.nom if dem.departement else 'Non spécifié'}")
        print(f"   - Statut initial: {soum.statut}")

        # Vérification Mathpix disponible
        mathpix_configure = bool(os.getenv("MATHPIX_APP_ID") and os.getenv("MATHPIX_APP_KEY"))
        if dem.departement and is_departement_scientifique(dem.departement):
            print(f"   - Département scientifique → Mathpix: {'Activé' if mathpix_configure else 'Non configuré'}")

        # 2) MISE À JOUR STATUT IMMÉDIATE
        soum.statut = 'analyse_ia'
        soum.progression = 20
        soum.save()
        print(f"📊 Statut mis à jour: analyse_ia (20%)")

        # 3) RÉCUPÉRATION OPTIMISÉE DU CONTENU AVEC SUPPORT MATHPIX
        extraction_start = time.time()
        fragment = None
        source = "unknown"
        idx = soum.exercice_index or 0
        methode_extraction = "standard"  # Pour le suivi

        # Tentative 1: Récupération depuis exercices_data
        if dem.exercices_data:
            try:
                exercices_list = json.loads(dem.exercices_data)
                for ex in exercices_list:
                    if ex.get('index') == idx:
                        fragment = ex.get('contenu_complet') or ex.get('contenu', '')
                        source = "exercices_data"
                        print(f"✅ Contenu récupéré depuis exercices_data")
                        print(f"   - Source: {source}")
                        print(f"   - Longueur: {len(fragment)} caractères")
                        break
            except json.JSONDecodeError as e:
                print(f"⚠️  Erreur JSON exercices_data: {e}")

        # Tentative 2: Fallback extraction fichier AVEC MATHPIX CONDITIONNEL
        if not fragment and dem.fichier:
            print(f"🔄 Fallback: extraction depuis fichier")
            try:
                # MODIFICATION CLÉ : Passer la demande pour choix Mathpix
                texte_complet = extraire_texte_fichier(dem.fichier, dem)  # ← dem ajouté comme 2ème paramètre

                if texte_complet and len(texte_complet.strip()) > 50:
                    exercices_data = separer_exercices_avec_titres(texte_complet)

                    if idx >= len(exercices_data):
                        print(f"⚠️  Index {idx} hors limites, ajustement")
                        idx = len(exercices_data) - 1 if exercices_data else 0

                    ex_data = exercices_data[idx] if exercices_data else {}
                    fragment = ex_data.get('contenu', '')
                    source = "extraction_fraiche"
                    print(f"✅ Contenu extrait via fallback")
                    print(f"   - Source: {source}")
                    print(f"   - Longueur: {len(fragment)} caractères")

                    # Enregistrer la méthode d'extraction depuis la demande si disponible
                    if hasattr(dem, 'methode_extraction') and dem.methode_extraction:
                        methode_extraction = dem.methode_extraction
                        print(f"   - Méthode extraction: {methode_extraction}")
                    else:
                        # Déterminer la méthode basée sur le département
                        if dem.departement and is_departement_scientifique(dem.departement):
                            methode_extraction = "mathpix" if mathpix_configure else "standard"
                        else:
                            methode_extraction = "standard"
                        print(f"   - Méthode déduite: {methode_extraction}")

                else:
                    print(f"⚠️  Texte extrait trop court: {len(texte_complet or '')} caractères")
            except Exception as e:
                print(f"❌ Erreur extraction fichier: {type(e).__name__}: {str(e)[:100]}")

        extraction_time = time.time() - extraction_start

        # 4) VALIDATION DU FRAGMENT
        if not fragment or len(fragment.strip()) < 20:
            error_msg = f"Fragment invalide (longueur: {len(fragment or '')} chars, source: {source})"
            print(f"❌ {error_msg}")
            print(f"⏱️  Temps extraction: {extraction_time:.1f}s")

            # Mise à jour statut erreur
            soum.statut = 'erreur'
            soum.save()

            raise ValueError(error_msg)

        print(f"✅ Fragment validé")
        print(f"⏱️  Extraction totale: {extraction_time:.1f}s")
        print(f"📝 Début fragment: {fragment[:100].replace(chr(10), ' ')}...")
        print(f"🔧 Méthode extraction: {methode_extraction}")

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
        print(f"🎯 Contexte IA: {contexte}")

        # 6) GÉNÉRATION IA AVEC GESTION D'ERREURS ROBUSTE
        ia_start = time.time()
        print(f"\n{'─' * 40}")
        print(f"🤖 DÉBUT GÉNÉRATION IA")
        print(f"{'─' * 40}")

        try:
            # Appel IA avec timeout global
            corrige_txt, _ = generer_corrige_ia_et_graphique(
                texte_enonce=fragment,
                contexte=contexte,
                matiere=mat,
                demande=dem
            )

            ia_time = time.time() - ia_start
            print(f"✅ Génération IA réussie ({ia_time:.1f}s)")
            print(f"📝 Longueur corrigé: {len(corrige_txt or '')} caractères")

            # Validation basique du corrigé
            if not corrige_txt or len(corrige_txt.strip()) < 50:
                error_msg = f"Corrigé trop court: {len(corrige_txt or '')} caractères"
                print(f"⚠️  {error_msg}")
                raise ValueError(error_msg)

        except Exception as ia_error:
            ia_time = time.time() - ia_start
            print(f"\n❌ ÉCHEC GÉNÉRATION IA ({ia_time:.1f}s)")
            print(f"   Type erreur: {type(ia_error).__name__}")
            print(f"   Message: {str(ia_error)[:200]}")
            print(f"{'─' * 40}")

            # Retry automatique après délai
            print(f"🔄 Retry automatique dans 60s...")
            raise self.retry(exc=ia_error, countdown=60)

        # 7) MISE À JOUR STATUT INTERMÉDIAIRE
        soum.statut = 'formatage_pdf'
        soum.progression = 60
        soum.save()
        print(f"📊 Statut mis à jour: formatage_pdf (60%)")

        # 8) GÉNÉRATION PDF
        pdf_start = time.time()
        print(f"\n{'─' * 40}")
        print(f"📄 DÉBUT GÉNÉRATION PDF")
        print(f"{'─' * 40}")

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
            print(f"❌ Échec génération PDF ({pdf_time:.1f}s)")
            print(f"   Erreur: {type(pdf_error).__name__}: {str(pdf_error)[:200]}")
            raise pdf_error

        # 9) DÉBIT CRÉDIT
        debit_start = time.time()
        print(f"\n{'─' * 40}")
        print(f"💳 DÉBIT CRÉDIT UTILISATEUR")
        print(f"{'─' * 40}")

        try:
            if not debiter_credit_abonnement(dem.user):
                error_msg = "Crédits insuffisants"
                print(f"❌ {error_msg}")

                soum.statut = 'erreur_credit'
                soum.save()

                raise ValueError(error_msg)

            debit_time = time.time() - debit_start
            print(f"✅ Débit crédit réussi ({debit_time:.1f}s)")

        except Exception as debit_error:
            print(f"❌ Erreur débit crédit: {type(debit_error).__name__}")
            raise debit_error

        # 10) CRÉATION CORRIGEPARTIEL
        corrige_start = time.time()
        print(f"\n{'─' * 40}")
        print(f"📁 CRÉATION CORRIGEPARTIEL")
        print(f"{'─' * 40}")

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
                print(f"❌ {error_msg}")
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
                # Optionnel: stocker la méthode d'extraction dans les métadonnées
                corrige.save()

            corrige_time = time.time() - corrige_start
            print(f"✅ CorrigePartiel créé ({corrige_time:.1f}s)")
            print(f"   - ID: {corrige.id}")
            print(f"   - Titre: {titre_reel}")
            print(f"   - Méthode extraction: {methode_extraction}")

        except Exception as corrige_error:
            corrige_time = time.time() - corrige_start
            print(f"❌ Erreur création CorrigePartiel ({corrige_time:.1f}s)")
            print(f"   Erreur: {type(corrige_error).__name__}: {str(corrige_error)[:200]}")
            raise corrige_error

        # 11) FINALISATION
        total_time = time.time() - task_start

        soum.statut = 'termine'
        soum.progression = 100
        soum.resultat_json = {
            "exercice_index": idx,
            "exercice_titre": titre_reel,
            "corrige_text": corrige_txt,
            "pdf_url": pdf_url,
            "timestamp": datetime.now().isoformat(),
            "processing_time": total_time,
            "source_content": source,
            "methode_extraction": methode_extraction,  # Ajout pour suivi
            "departement": dem.departement.nom if dem.departement else None
        }
        soum.save()

        print(f"\n{'=' * 70}")
        print(f"✅ TÂCHE TERMINÉE AVEC SUCCÈS!")
        print(f"   Temps total: {total_time:.1f}s")
        print(f"   Exercice: {titre_reel}")
        print(f"   Source contenu: {source}")
        print(f"   Méthode extraction: {methode_extraction}")
        print(f"   Département: {dem.departement.nom if dem.departement else 'Non spécifié'}")
        print(f"   Corrigé: {len(corrige_txt)} caractères")
        print(f"   {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 70}")

        return True

    except Exception as e:
        total_time = time.time() - task_start

        print(f"\n{'=' * 70}")
        print(f"❌ ERREUR CRITIQUE DANS LA TÂCHE")
        print(f"   Temps écoulé: {total_time:.1f}s")
        print(f"   Type erreur: {type(e).__name__}")
        print(f"   Message: {str(e)[:300]}")
        print(f"   Soumission ID: {soumission_id}")
        print(f"   {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 70}")

        # Log détaillé de l'erreur
        import traceback
        error_details = traceback.format_exc()
        print(f"\n📋 TRACEBACK COMPLET:")
        print(error_details[:1000])  # Limité pour éviter logs trop longs

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
            print(f"🔄 Erreur réseau détectée, retry automatique...")
            raise self.retry(exc=e, countdown=120)

        # Pour les autres erreurs, on ne retry pas
        return False

