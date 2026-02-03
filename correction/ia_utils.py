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
from resources.models import PromptIA, Matiere
from .pdf_utils import generer_pdf_corrige
from .models import SoumissionIA
from resources.models import Matiere
from abonnement.services import debiter_credit_abonnement
from .models import CorrigePartiel
from django.core.files import File

try:
    from .latex_utils import convertir_balises_latex_mathpix
except ImportError:
    # Fallback si fichier non créé
    def convertir_balises_latex_mathpix(texte):
        return texte

logger = logging.getLogger(__name__)

# ========== CONFIGURATION MATHPIX ==========
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")


# ========== FONCTIONS DE PRÉTRAITEMENT IMAGE ==========
def preprocess_image_for_ocr(pil_image):
    """Convertit une PIL.Image en image binaire nettoyée pour Tesseract."""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    kernel = np.ones((1, 1), np.uint8)
    clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    return clean


# ========== DÉTECTION DÉPARTEMENT SCIENTIFIQUE AVANCÉE ==========
def detecter_departement_scientifique_avance(departement):
    """
    Version ROBUSTE avec gestion d'erreurs.
    """
    if not departement:
        return False

    try:
        # Si c'est un string direct
        if isinstance(departement, str):
            nom_dep = departement.upper()
        # Si c'est un objet avec attribut nom
        elif hasattr(departement, 'nom'):
            nom_dep = str(departement.nom).upper()
        else:
            nom_dep = str(departement).upper()

        # Liste complète des départements scientifiques
        SCIENTIFIQUES = [
            'MATHEMATIQUES', 'MATHS', 'MATHÉMATIQUES',
            'PHYSIQUE', 'PHYSIQUE-CHIMIE', 'PHYSIQUE CHIMIE',
            'CHIMIE', 'SCIENCES PHYSIQUES',
            'SCIENCES', 'SCIENCE', 'SVT',
            'TECHNOLOGIE', 'SCIENCES DE LINGENIEUR',
            'INFORMATIQUE', 'SCIENCES NUMERIQUES', 'SCIENCES NUMÉRIQUES',
            'BIOLOGIE', 'SCIENCES DE LA VIE'
        ]

        # Vérifier correspondance
        for dep_sci in SCIENTIFIQUES:
            if dep_sci in nom_dep:
                return True

        return False

    except Exception as e:
        print(f"⚠️ Erreur détection département scientifique: {e}")
        return False  # En cas d'erreur, considérer comme non scientifique


# ========== OCR MATHPIX (extraction scientifique) ==========
def ocr_mathpix(path_image: str) -> dict:
    """
    Appelle l'API Mathpix pour extraire texte + LaTeX.
    """
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        logger.error("❌ MathPix credentials manquants")
        return {"text": "", "latex_simplified": ""}

    try:
        with open(path_image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        headers = {
            "app_id": MATHPIX_APP_ID,
            "app_key": MATHPIX_APP_KEY,
            "Content-type": "application/json"
        }

        payload = {
            "src": f"data:image/png;base64,{img_b64}",
            "formats": ["text", "latex_simplified"],
            "math_inline_delimiters": ["$", "$"],
            "rm_spaces": True
        }

        resp = requests.post(
            "https://api.mathpix.com/v3/text",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()

        result = resp.json()
        logger.info(f"✅ MathPix OCR réussi: {len(result.get('text', ''))} caractères")
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur API MathPix: {e}")
        return {"text": "", "latex_simplified": ""}
    except Exception as e:
        logger.error(f"❌ Erreur inattendue MathPix: {e}")
        return {"text": "", "latex_simplified": ""}


def extraire_avec_mathpix(fichier_path: str) -> str:
    """
    Extraction complète avec MathPix pour un fichier.
    Gère PDF (converti en images) et images directes.
    """
    if not fichier_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        logger.error(f"Format non supporté par MathPix: {fichier_path}")
        return ""

    texte_complet = ""

    try:
        if fichier_path.lower().endswith('.pdf'):
            pages = convert_from_path(fichier_path, dpi=300)
            logger.info(f"📄 PDF converti en {len(pages)} pages pour MathPix")

            for i, page in enumerate(pages, 1):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    page.save(tmp.name, 'PNG')

                    result = ocr_mathpix(tmp.name)
                    page_text = result.get('text', '')
                    latex = result.get('latex_simplified', '')

                    if page_text:
                        texte_complet += f"\n\n--- Page {i} ---\n{page_text}"
                    if latex:
                        texte_complet += f"\n[Formules page {i}]: {latex}"

                    os.unlink(tmp.name)

        else:
            result = ocr_mathpix(fichier_path)
            texte_complet = result.get('text', '')
            latex = result.get('latex_simplified', '')

            if latex:
                texte_complet += f"\n\n[Formules LaTeX]: {latex}"

    except Exception as e:
        logger.error(f"❌ Erreur extraction MathPix: {e}")
    texte_complet = convertir_balises_latex_mathpix(texte_complet)
    return texte_complet.strip()


# ========== CONFIGURATION DEEPSEEK ==========
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.api_base = "https://api.deepseek.com"
DEEPSEEK_VISION_MODEL = "deepseek-reasoner"


# ========== APPEL DEEPSEEK VISION ==========
def call_deepseek_vision(path_fichier: str) -> dict:
    """
    Envoie un PDF ou une image à DeepSeek Vision.
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
        with open(path_fichier, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")

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
        logger.error(f"❌ Erreur call_deepseek_vision: {e}")
        return {"text": "", "latex_blocks": [], "captions": [], "graphs": []}


# ========== FUSION EXTRACTIONS SCIENTIFIQUES ==========
def fusionner_extractions_scientifiques(texte_mathpix: str, analyse_vision: dict) -> str:
    """
    Fusionne intelligemment l'extraction MathPix et l'analyse DeepSeek Vision.
    """
    if not texte_mathpix and not analyse_vision:
        return ""

    parties = []

    if texte_mathpix:
        parties.append(f"=== TEXTE ET FORMULES ===\n{texte_mathpix}")

    if analyse_vision.get("captions"):
        parties.append("\n=== SCHÉMAS DÉTECTÉS ===")
        for idx, element in enumerate(analyse_vision["captions"], 1):
            desc = element if isinstance(element, str) else f"Schéma {idx}"
            parties.append(f"[SCHÉMA {idx}]: {desc}")

    if analyse_vision.get("numbers"):
        parties.append("\n=== DONNÉES NUMÉRIQUES ===")
        for num in analyse_vision["numbers"]:
            val = num.get("valeur", "")
            unit = num.get("unité", "")
            if val:
                parties.append(f"- {val} {unit}")

    if analyse_vision.get("angles"):
        parties.append("\n=== ANGLES DÉTECTÉS ===")
        for angle in analyse_vision["angles"]:
            val = angle.get("valeur", "")
            unit = angle.get("unité", "°")
            if val:
                parties.append(f"- {val}{unit}")

    return "\n".join(parties)


# ========== ANALYSE DOCUMENT SCIENTIFIQUE UNIFIÉE ==========
def analyser_document_scientifique(fichier_path: str, departement=None) -> dict:
    """
    Analyse scientifique avancée avec sélection automatique du pipeline.
    """
    logger.info(f"🔍 Analyse scientifique pour {fichier_path}")

    # Décision du pipeline
    besoin_scientifique = detecter_departement_scientifique_avance(departement)

    texte_mathpix = ""
    analyse_vision = {"text": "", "captions": [], "graphs": []}

    if besoin_scientifique:
        logger.info("⚡ Département scientifique détecté → Pipeline MathPix + DeepSeek Vision")

        # 1. Extraction MathPix
        texte_mathpix = extraire_avec_mathpix(fichier_path)

        # 2. Analyse DeepSeek Vision
        try:
            analyse_vision = call_deepseek_vision(fichier_path)
            if isinstance(analyse_vision, str):
                analyse_vision = json.loads(analyse_vision)
        except Exception as e:
            logger.error(f"❌ Erreur DeepSeek Vision: {e}")
            analyse_vision = {"text": "", "captions": [], "graphs": []}

        # 3. Fusion intelligente
        texte_fusionne = fusionner_extractions_scientifiques(texte_mathpix, analyse_vision)

        logger.info(
            f"✅ Pipeline scientifique: {len(texte_fusionne)} caractères, {len(analyse_vision.get('captions', []))} schémas")

        return {
            "texte_complet": texte_fusionne,
            "elements_visuels": analyse_vision.get("captions", []),
            "formules_latex": analyse_vision.get("latex_blocks", []),
            "graphs": analyse_vision.get("graphs", []),
            "angles": analyse_vision.get("angles", []),
            "numbers": analyse_vision.get("numbers", []),
            "pipeline": "scientifique",
            "mathpix_used": True
        }

    else:
        logger.info("📄 Département non scientifique → Pipeline standard OCR")

        # Pipeline standard
        try:
            vision_json = call_deepseek_vision(fichier_path)
            if isinstance(vision_json, str):
                vision_json = json.loads(vision_json)

            texte_json = vision_json.get("text", "") or ""

            if len(texte_json) < 100:
                texte_json = extraire_texte_robuste(fichier_path)

            logger.info(f"✅ Pipeline standard: {len(texte_json)} caractères")

            return {
                "texte_complet": texte_json,
                "elements_visuels": vision_json.get("captions", []),
                "formules_latex": vision_json.get("latex_blocks", []),
                "graphs": vision_json.get("graphs", []),
                "angles": vision_json.get("angles", []),
                "numbers": vision_json.get("numbers", []),
                "pipeline": "standard",
                "mathpix_used": False
            }

        except Exception as e:
            logger.error(f"❌ Erreur pipeline standard: {e}")
            texte_fallback = extraire_texte_robuste(fichier_path)
            return {
                "texte_complet": texte_fallback,
                "elements_visuels": [],
                "formules_latex": [],
                "graphs": [],
                "angles": [],
                "numbers": [],
                "pipeline": "fallback",
                "mathpix_used": False
            }


# ========== ENRICHISSEMENT AVEC DEEPSEEK REASONER ==========
def enrichir_exercice_avec_reasoner(texte_exercice: str, donnees_vision: dict, index_exercice: int = 1) -> str:
    """
    Utilise DeepSeek Reasoner pour reconstituer un énoncé scientifique clair.
    """
    elements_visuels = donnees_vision.get("elements_visuels", [])
    formules = donnees_vision.get("formules_latex", [])
    numbers = donnees_vision.get("numbers", [])
    angles = donnees_vision.get("angles", [])

    system_prompt = """Tu es un expert en reconstitution d'énoncés scientifiques.
Ta mission : à partir d'un texte extrait (parfois incomplet ou mal structuré) et de données visuelles,
reconstituer un énoncé d'exercice scientifique CLAIR, COMPLET et BIEN STRUCTURÉ.

RÈGLES ABSOLUES :
1. Garde TOUTES les informations techniques et numériques
2. Reconstruis la logique de l'exercice si nécessaire
3. Intègre les schémas sous forme de descriptions textuelles précises
4. Formate les formules mathématiques en LaTeX : \[ ... \] pour les blocs, \( ... \) pour l'inline
5. Numérote clairement les questions si elles sont détectées

FORMAT DE SORTIE :
- Titre de l'exercice (si détecté)
- Énoncé principal reconstitué
- Questions numérotées (1., 2., etc.)
- [SCHÉMA X] : Description précise à insérer dans le texte
- [DONNÉE Y] : Valeur numérique à utiliser

Ne réponds QUE par l'énoncé reconstitué, sans commentaires."""

    user_prompt = f"""RECONSTITUTION DE L'EXERCICE {index_exercice}

=== TEXTE EXTRACT (potentiellement incomplet) ===
{texte_exercice}

=== DONNÉES VISUELLES DÉTECTÉES ===
"""

    if elements_visuels:
        user_prompt += "\nSCHÉMAS :\n"
        for i, elem in enumerate(elements_visuels, 1):
            desc = elem if isinstance(elem, str) else elem.get("description", f"Schéma {i}")
            user_prompt += f"[SCHÉMA {i}]: {desc}\n"

    if formules:
        user_prompt += "\nFORMULES DÉTECTÉES :\n"
        for i, formule in enumerate(formules, 1):
            user_prompt += f"[FORMULE {i}]: {formule}\n"

    if numbers:
        user_prompt += "\nDONNÉES NUMÉRIQUES :\n"
        for i, num in enumerate(numbers, 1):
            val = num.get("valeur", "")
            unit = num.get("unité", "")
            user_prompt += f"[DONNÉE {i}]: {val} {unit}\n"

    if angles:
        user_prompt += "\nANGLES DÉTECTÉS :\n"
        for i, angle in enumerate(angles, 1):
            val = angle.get("valeur", "")
            unit = angle.get("unité", "°")
            user_prompt += f"[ANGLE {i}]: {val}{unit}\n"

    user_prompt += "\n---\nRECONSTITUE UN ÉNONCÉ CLAIR ET COMPLET :"

    try:
        response = openai.ChatCompletion.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000,
            stream=False
        )

        enonce_reconstitue = response.choices[0].message.content
        logger.info(f"✅ Exercice {index_exercice} enrichi: {len(enonce_reconstitue)} caractères")
        return enonce_reconstitue

    except Exception as e:
        logger.error(f"❌ Erreur enrichissement Reasoner: {e}")
        return texte_exercice


# ========== FILTRAGE DONNÉES VISION ==========
def filtrer_donnees_vision_par_exercice(donnees_vision_complete: dict, texte_exercice: str) -> dict:
    """
    Filtre les données de vision pour ne garder que celles pertinentes.
    """
    if not donnees_vision_complete or not texte_exercice:
        return {}

    mots = set(texte_exercice.lower().split()[:50])

    resultat = {
        "elements_visuels": [],
        "formules_latex": [],
        "graphs": [],
        "angles": [],
        "numbers": []
    }

    for elem in donnees_vision_complete.get("elements_visuels", []):
        if isinstance(elem, str):
            desc = elem
        else:
            desc = elem.get("description", "")

        desc_lower = desc.lower()
        pertinence = any(mot in desc_lower for mot in mots if len(mot) > 3)

        if pertinence:
            resultat["elements_visuels"].append(elem)

    resultat["formules_latex"] = donnees_vision_complete.get("formules_latex", [])
    resultat["graphs"] = donnees_vision_complete.get("graphs", [])
    resultat["angles"] = donnees_vision_complete.get("angles", [])
    resultat["numbers"] = donnees_vision_complete.get("numbers", [])

    return resultat


# ========== FONCTIONS EXISTANTES (conservées) ==========
def is_departement_scientifique(departement):
    """Ancienne fonction conservée pour compatibilité."""
    DEPARTEMENTS_SCIENTIFIQUES = [
        'MATHEMATIQUES', 'PHYSIQUE', 'CHIMIE', 'biologie', 'svt', 'sciences', 'informatique'
    ]
    if departement and departement.nom:
        dep_name = departement.nom.lower()
        return any(dep_name.startswith(sc) or sc in dep_name for sc in DEPARTEMENTS_SCIENTIFIQUES)
    return False


def get_best_promptia(demande):
    """Retourne le PromptIA le plus spécifique pour la demande."""
    filtra = {
        'pays': demande.pays,
        'sous_systeme': demande.sous_systeme,
        'classe': demande.classe,
        'matiere': demande.matiere,
        'departement': demande.departement,
        'type_exercice': demande.type_exercice,
    }
    filtra = {k: v for k, v in filtra.items() if v is not None}

    if filtra:
        qs = PromptIA.objects.filter(**filtra)
        if qs.exists():
            return qs.first()

        for champ in ['type_exercice', 'departement', 'classe', 'sous_systeme', 'pays']:
            if champ in filtra:
                filtra2 = dict(filtra)
                filtra2.pop(champ)
                if filtra2:
                    qs2 = PromptIA.objects.filter(**filtra2)
                    if qs2.exists():
                        return qs2.first()

        if 'matiere' in filtra:
            qs3 = PromptIA.objects.filter(matiere=demande.matiere)
            if qs3.exists():
                return qs3.first()

    return None


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


# ========== PATTERNS DE STRUCTURE ==========
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
    r'^\d{1,2}[.\-]', r'^\d{1,2}[.]\d{1,2}[.-]?', r'^\d{1,2,3}[a-z]{1}[.]',
    r'^[ivxIVX]{1,4}[.)-]', r'^[a-z]{1}[.)]', r'^[A-Z]{1}[.)]',
    r'^\d{1,2}[.][a-z]{1}[.]', r'^\d{1,2}[.][A-Z]{1}[.]',
    r'^\(\d+\)', r'^\([a-z]\)', r'^\([ivxIVX]+\)',
]


# ========== SÉPARATION EXERCICES ==========
def separer_exercices_avec_titres(texte_epreuve, min_caracteres=60):
    """Version avec hiérarchie parent-enfant pour les titres."""
    if not texte_epreuve:
        return []

    lignes = texte_epreuve.splitlines()

    mots_cles_exercices = [
        'EXERCICE', 'EXERICE', 'PROBLÈME', 'PROBLEME',
        'PARTIE.*EVALUATION DES COMPETENCES',
        'SITUATION PROBLÈME',
        'SECTION', 'PART', 'EXERCISE', 'QUESTION',
        'TASK', 'ACTIVITY',
        'EJERCICIO', 'PRUEBA',
        'AUFGABE', 'TEIL',
        '[A-D][\\s\\-\\.:]*É?VALUATION[\\s\\-]*DES[\\s\\-]*COMPÉTENCES',
        '[A-D][\\s\\-\\.:]*É?VALUATION[\\s\\-]*DES[\\s\\-]*COMPETENCES',
        '[IVXL]+[\\s\\-\\.:]*É?VALUATION[\\s\\-]*DES[\\s\\-]*COMPÉTENCES',
        '[IVXL]+[\\s\\-\\.:]*É?VALUATION[\\s\\-]*DES[\\s\\-]*COMPETENCES',
        'SUJET[\\s\\-]*DE[\\s\\-]*TYPE[\\s\\-]*[\\dIVXL]+',
        'SUJET[\\s\\-]*TYPE[\\s\\-]*[\\dIVXL]+',
        'SUJET[\\s\\-]*[\\dIVXL]+',
    ]

    patterns = []
    for mot_cle in mots_cles_exercices:
        pattern_str = f'^{mot_cle}[\\s\\-]*[\\dA-ZIVXL]*[\\s\\-:\\.]'
        patterns.append(re.compile(pattern_str, re.IGNORECASE))

    tous_les_blocs = []
    current_block = []
    current_title = None
    current_start_index = 0

    for i, ligne in enumerate(lignes):
        ligne_stripped = ligne.strip()

        est_titre_potentiel = False
        for pattern in patterns:
            if pattern.match(ligne_stripped.upper()):
                est_titre_potentiel = True
                break

        if not est_titre_potentiel:
            if re.search(r'\(\s*\d+[\s,\.]*(?:point|pt|mark|marque|note)s?\s*\)', ligne_stripped.upper()):
                est_titre_potentiel = True

        if est_titre_potentiel:
            if current_block and current_title:
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

            current_title = ligne_stripped
            current_block = [ligne]
            current_start_index = i
        else:
            if current_block:
                current_block.append(ligne)

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

    groupes = []
    groupe_courant = []

    for bloc in tous_les_blocs:
        if not bloc['has_enough_content']:
            groupe_courant.append(bloc)
        else:
            if groupe_courant:
                groupes.append(groupe_courant + [bloc])
                groupe_courant = []
            else:
                groupes.append([bloc])

    if groupe_courant:
        if groupes:
            groupes[-1].extend(groupe_courant)
        else:
            groupes.append(groupe_courant)

    resultats = []

    for groupe in groupes:
        if not groupe:
            continue

        if len(groupe) == 1:
            bloc = groupe[0]
            titre_final = bloc['title']
            lignes_finales = bloc['lines']
        else:
            titres = [bloc['title'] for bloc in groupe]
            titre_final = " / ".join(titres)

            lignes_finales = []
            for bloc in groupe:
                lignes_finales.extend(bloc['lines'])
                if bloc != groupe[-1]:
                    lignes_finales.append("")

        titre_affichage = titre_final
        if len(titre_affichage) > 150:
            mots = titre_affichage.split()
            if len(mots) > 8:
                titre_affichage = ' '.join(mots[:4]) + " ... " + ' '.join(mots[-4:])
            else:
                titre_affichage = titre_affichage[:147] + "..."

        lignes_limitees = lignes_finales[:300]
        contenu = '\n'.join(lignes_limitees)
        longueur_totale = sum(len(l) for l in lignes_limitees[1:] if len(lignes_limitees) > 1)

        resultats.append({
            'titre': titre_affichage,
            'contenu': contenu,
            'titre_complet': titre_final,
            'longueur_contenu': longueur_totale,
            'nombre_parents': len(groupe) - 1 if len(groupe) > 1 else 0
        })

    if not resultats:
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
            contenu_lines = lignes[:100]
            contenu = '\n'.join(contenu_lines)
            resultats.append({
                'titre': "Document complet",
                'contenu': contenu,
                'titre_complet': "Document complet",
                'longueur_contenu': len(contenu)
            })

    return resultats


def separer_exercices(texte_epreuve):
    """Version simple maintenue pour compatibilité."""
    resultats = separer_exercices_avec_titres(texte_epreuve)
    return [ex['contenu'] for ex in resultats]


# ========== EXTRACTION TEXTE ROBUSTE ==========
def extraire_texte_robuste(fichier_path: str) -> str:
    """Extraction simple : OCR direct → Analyse IA."""
    print("🔄 Extraction simple...")

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


# ============== EXTRACTION TEXTE/FICHIER ==============

def extraire_texte_pdf(fichier_path):
    try:
        texte = extract_text(fichier_path)
        print(f"📄 PDF extrait: {len(texte)} caractères")
        return texte.strip() if texte else ""
    except Exception as e:
        print(f"❌ Erreur extraction PDF: {e}")
        return ""


# ========== EXTRACTION TEXTE FICHIER OPTIMISÉE ==========
def extraire_texte_fichier(fichier_field, departement=None):
    """
    Extraction robuste avec sélection automatique du pipeline.
    Optimisé pour extraction UNE FOIS par sujet.
    """
    if not fichier_field:
        return ""

    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))

    with open(local_path, "wb") as f:
        for chunk in fichier_field.chunks():
            f.write(chunk)

    ext = os.path.splitext(local_path)[1].lower()

    try:
        analyse = analyser_document_scientifique(local_path, departement)
        texte = analyse.get("texte_complet", "")

        pipeline = analyse.get("pipeline", "standard")
        print(f"✅ Extraction ({pipeline}): {len(texte)} caractères")

    except Exception as e:
        print(f"❌ Erreur extraction: {e}")
        texte = ""

    try:
        os.unlink(local_path)
    except:
        pass

    return texte.strip()


# ========== GÉNÉRATION CORRIGÉ PAR EXERCICE ==========
def generer_corrige_par_exercice(texte_exercice, contexte, matiere=None, donnees_vision=None, demande=None,
                                 exercice_index=1):
    """
    Génère le corrigé pour un seul exercice avec enrichissement scientifique si besoin.
    """
    print(f"🎯 Génération corrigé exercice {exercice_index}...")

    # Vérifier si besoin d'enrichissement scientifique
    besoin_enrichissement = False
    if demande and demande.departement:
        besoin_enrichissement = detecter_departement_scientifique_avance(demande.departement)

    # Étape d'enrichissement si département scientifique
    if besoin_enrichissement and donnees_vision:
        print(f"🔬 Département scientifique → Enrichissement avec Reasoner...")
        texte_exercice = enrichir_exercice_avec_reasoner(
            texte_exercice,
            donnees_vision,
            index_exercice=exercice_index
        )
        print(f"✅ Énoncé enrichi: {len(texte_exercice)} caractères")

    # Récupérer prompt métier
    promptia = get_best_promptia(demande)

    # Construire les messages
    msg_system, msg_user = build_promptia_messages(promptia, contexte)

    # Enrichir le user_message
    user_blocks = [
        msg_user["content"],
        "----- EXERCICE À CORRIGER -----",
        texte_exercice.strip()
    ]

    if donnees_vision:
        if donnees_vision.get("elements_visuels"):
            user_blocks.append("----- SCHÉMAS IDENTIFIÉS -----")
            for element in donnees_vision["elements_visuels"]:
                desc = element.get("description", "") if isinstance(element, dict) else str(element)
                user_blocks.append(f"- {desc}")

        if donnees_vision.get("formules_latex"):
            user_blocks.append("----- FORMULES DÉTECTÉES -----")
            for formule in donnees_vision["formules_latex"]:
                user_blocks.append(f"- {formule}")

        if donnees_vision.get("graphs"):
            user_blocks.append("----- DONNÉES GRAPHIQUES (JSON) -----")
            user_blocks.append(json.dumps(donnees_vision["graphs"], ensure_ascii=False, indent=2))

        if donnees_vision.get("angles"):
            user_blocks.append("----- ANGLES IDENTIFIÉS -----")
            for angle in donnees_vision["angles"]:
                val = angle.get("valeur", "")
                unit = angle.get("unité", "")
                coord = angle.get("coord", "")
                user_blocks.append(f"- {val}{unit} à coord {coord}")

        if donnees_vision.get("numbers"):
            user_blocks.append("----- NOMBRES ET UNITÉS -----")
            for num in donnees_vision["numbers"]:
                val = num.get("valeur", "")
                unit = num.get("unité", "")
                coord = num.get("coord", "")
                user_blocks.append(f"- {val}{unit} à coord {coord}")

    msg_user["content"] = "\n\n".join(user_blocks)

    # Appel API
    data = {
        "model": "deepseek-chat",
        "messages": [msg_system, msg_user],
        "temperature": 0.1,
        "max_tokens": 6000,
        "top_p": 0.9,
        "frequency_penalty": 0.1
    }

    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json"
    }

    try:
        print("📡 Appel API DeepSeek avec analyse vision...")

        for tentative in range(2):
            response = requests.post(api_url, headers=headers, json=data, timeout=90)
            response_data = response.json()

            if response.status_code != 200:
                error_msg = f"Erreur API: {response_data.get('message', 'Pas de détail')}"
                print(f"❌ {error_msg}")
                return error_msg, None

            output = response_data['choices'][0]['message']['content']
            print(f"✅ Réponse IA brute (tentative {tentative + 1}): {len(output)} caractères")

            if verifier_qualite_corrige(output, texte_exercice):
                print("✅ Qualité du corrigé validée")
                break
            else:
                print(f"🔄 Tentative {tentative + 1} - Qualité insuffisante, régénération...")
                data["messages"][1][
                    "content"] += "\n\n⚠️ ATTENTION : Sois plus rigoureux ! Exploite mieux les schémas identifiés. Vérifie tous tes calculs."

                if tentative == 0:
                    import time
                    time.sleep(2)
        else:
            print("❌ Échec après 2 tentatives - qualité insuffisante")
            return "Erreur: Qualité du corrigé insuffisante après plusieurs tentatives", None

        # Traitement de la réponse
        output = flatten_multiline_latex_blocks(output)
        output_structured = format_corrige_pdf_structure(output)

        # Initialisation
        corrige_txt = output_structured
        graph_list = []

        # Extraction graphique
        json_blocks = extract_json_blocks(output_structured)
        print(f"🔍 JSON blocks détectés : {len(json_blocks)}")

        json_blocks = sorted(json_blocks, key=lambda x: x[1], reverse=True)

        for idx, (graph_dict, start, end) in enumerate(json_blocks, start=1):
            try:
                output_name = f"graphique_{idx}.png"
                img_path = tracer_graphique(graph_dict, output_name)
                if img_path is None:
                    raise ValueError("tracer_graphique a retourné None")

                abs_path = os.path.join(settings.MEDIA_ROOT, img_path)
                img_tag = (
                    f'<img src="file://{abs_path}" alt="Graphique {idx}" '
                    f'style="max-width:100%;margin:10px 0;" />'
                )
                corrige_txt = corrige_txt[:start] + img_tag + corrige_txt[end:]
                graph_list.append(graph_dict)
                print(f"✅ Graphique {idx} inséré")
            except Exception as e:
                print(f"❌ Erreur génération graphique {idx}: {e}")
                continue

        return corrige_txt.strip(), graph_list

    except Exception as e:
        error_msg = f"Erreur: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, None


# ========== FONCTIONS UTILITAIRES (conservées) ==========
def build_promptia_messages(promptia, contexte):
    """Construit les messages système et utilisateur."""
    DEFAULT_SYSTEM_PROMPT = r"""Tu es un professeur expert en Mathématiques, physique, chimie, biologie,francais,histoire
géographie...bref, tu es un professeur de l'enseignement secondaire.

🔬 **CAPACITÉ VISION ACTIVÉE** - Tu peux maintenant analyser les schémas scientifiques !

RÈGLES ABSOLUES POUR L'ANALYSE DES SCHÉMAS :
1. ✅ Identifie le TYPE de schéma (plan incliné, circuit électrique, molécule, graphique)
2. ✅ Extrait les DONNÉES NUMÉRIQUES (angles, masses, distances, forces, tensions)
3. ✅ Décris les RELATIONS SPATIALES entre les éléments
4. ✅ Explique le CONCEPT SCIENTIFIQUE illustré

RÈGLES GÉNÉRALES DE CORRECTION :
- Sois EXTRÊMEMENT RIGOUREUX dans tous les calculs
- Vérifie systématiquement tes résultats intermédiaires  
- Ne laisse JAMAIS une question sans réponse complète
- Donne TOUTES les étapes de calcul détaillées
- Les réponses doivent être NUMÉRIQUEMENT EXACTES
... [Le reste de votre prompt] ..."""

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
    user_content = contexte.strip()

    return {"role": "system", "content": system_content}, \
        {"role": "user", "content": user_content}


def verifier_qualite_corrige(corrige_text, exercice_original):
    """Vérifie si le corrigé généré est de bonne qualité."""
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

    problemes_trouves = sum(1 for indicateur in indicateurs_problemes
                            if indicateur.lower() in corrige_text.lower())

    if problemes_trouves >= 2:
        print(f"🔄 Qualité insuffisante détectée ({problemes_trouves} indicateurs)")
        return False

    if len(corrige_text) < len(exercice_original) * 0.3:
        print("🔄 Corrigé trop court par rapport à l'énoncé")
        return False

    return True


def estimer_tokens(texte):
    """Estimation simple du nombre de tokens."""
    mots = len(texte.split())
    tokens = int(mots / 0.75)
    print(f"📊 Estimation tokens: {mots} mots → {tokens} tokens")
    return tokens


def extract_json_blocks(text: str):
    """Extrait les blocs JSON pour les graphiques."""
    decoder = json.JSONDecoder()
    idx = 0
    blocks = []

    while True:
        start = text.find('{', idx)
        if start == -1:
            break

        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and 'graphique' in obj:
                blocks.append((obj, start, start + end))
            idx = start + end
        except ValueError:
            idx = start + 1

    return blocks


def format_corrige_pdf_structure(texte_corrige_raw):
    """Nettoie et structure le corrigé pour le PDF/HTML."""
    texte = re.sub(r"(#+\s*)", "", texte_corrige_raw)
    texte = re.sub(r"(\*{2,})", "", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)

    lignes = texte.strip().split('\n')
    html_output = []
    in_bloc = False

    for line in lignes:
        line = line.strip()
        if not line:
            continue

        if any(re.search(pat, line, re.IGNORECASE) for pat in PATTERNS_BLOCS):
            if in_bloc: html_output.append("</div>")
            html_output.append(
                f'<div class="bloc-exercice" style="margin-top:60px;"><h1 class="titre-exercice">{line}</h1>')
            in_bloc = True
            continue

        if any(re.match(pat, line) for pat in PATTERNS_QUESTIONS):
            html_output.append(f'<h2 class="titre-question">{line}</h2>')
            continue

        if line.lower().startswith(("algorithme", "début", "fin", "code")):
            html_output.append(f'<div class="code-block">{line}</div>')
            continue

        html_output.append(f'<p class="reponse-question">{line}</p>')

    if in_bloc: html_output.append("</div>")
    return "".join(html_output)


def flatten_multiline_latex_blocks(text):
    """Fusionne les blocs LaTeX multilignes."""
    if not text:
        return ""

    text = re.sub(
        r'\\\[\s*([\s\S]+?)\s*\\\]',
        lambda m: r'\[' + " ".join(m.group(1).splitlines()).strip() + r'\]',
        text,
        flags=re.DOTALL
    )

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

    text = re.sub(
        r'\$\$([\s\S]+?)\$\$',
        lambda m: r'\[' + " ".join(m.group(1).splitlines()).strip() + r'\]',
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
        lambda m: r'\(' + m.group(1).replace('\n', ' ').strip() + r'\)',
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r'\\\[\s*([\s\S]+?)\s*\\\]',
        lambda m: r'\[' + " ".join(m.group(1).splitlines()).strip() + r'\]',
        text,
        flags=re.DOTALL
    )

    text = re.sub(r'\\\\\s*\[', r'\[', text)
    text = re.sub(r'\\\\\s*\]', r'\]', text)
    text = text.replace('\\backslash', '\\').replace('\xa0', ' ')
    return text


def extract_and_process_graphs(output: str):
    """Extrait et traite les graphiques d'un corrigé."""
    print("🖼️ Extraction des graphiques (via JSONDecoder)…")

    graphs_data = []
    final_text = output

    json_blocks = extract_json_blocks(output)
    print(f"🔍 JSON blocks détectés dans extract_and_process_graphs: {len(json_blocks)}")

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

                s, e = start + offset, end + offset
                final_text = final_text[:s] + img_tag + final_text[e:]
                offset += len(img_tag) - (end - start)

                graphs_data.append(graph_dict)
                print(f"✅ Graphique {idx} inséré.")
            else:
                s, e = start + offset, end + offset
                final_text = final_text[:s] + "[Erreur génération graphique]" + final_text[e:]
                offset += len("[Erreur génération graphique]") - (end - start)
                print(f"❌ Graphique {idx} : erreur de tracé.")

        except Exception as e:
            print(f"❌ Exception sur bloc graphique {idx}: {e}")
            continue

    print(f"🎯 Extraction terminée: {len(graphs_data)} graphique(s) traité(s)")
    return final_text, graphs_data


def generate_corrige_html(corrige_text):
    """Transforme le corrigé brut en HTML stylisé."""
    if not corrige_text:
        return ""

    lines = corrige_text.strip().split('\n')
    pattern_exercice = re.compile(r'^(EXERCICE\s*\d+|PARTIE\s*[IVXLCDM]+|Exercice\s*\d+|Partie\s*[IVXLCDM]+)',
                                  re.IGNORECASE)
    html_output = []
    i = 0

    html_output.append(
        '<div class="cis-message"><strong>SUJET CORRIGÉ PAR L\'APPLICATION CIS, DISPO SUR PLAYSTORE</strong></div>')

    in_bloc_exercice = False

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if pattern_exercice.match(line):
            if in_bloc_exercice: html_output.append("</div>")
            html_output.append(f'<div class="bloc-exercice"><h1 class="titre-exercice">{line}</h1>')
            in_bloc_exercice = True
            i += 1
            continue

        if re.match(r'^Question\s*\d+', line, re.IGNORECASE):
            html_output.append(f'<h2 class="titre-question">{line}</h2>')
            i += 1
            continue

        if re.match(r'^\d+\.', line):
            html_output.append(f'<h3 class="titre-question">{line}</h3>')
            i += 1
            continue

        if re.match(r'^[a-z]\)', line):
            html_output.append(f'<p><strong>{line}</strong></p>')
            i += 1
            continue

        if line.startswith('•') or line.startswith('-'):
            html_output.append(f'<p>{line}</p>')
            i += 1
            continue

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

        if '\\(' in line or '\\[' in line:
            html_output.append(f'<p class="reponse-question mathjax">{line}</p>')
            i += 1
            continue

        html_output.append(f'<p class="reponse-question">{line}</p>')
        i += 1

    if in_bloc_exercice:
        html_output.append("</div>")

    return mark_safe("".join(html_output))


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


# ========== FONCTIONS DE GRAPHIQUES (conservées) ==========
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
                print("Erreur safe_float cast direct:", expr, e2);
                return None

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


# ========== TÂCHES ASYNCHRONES OPTIMISÉES ==========
@shared_task(name='correction.ia_utils.generer_corrige_exercice_async')
def generer_corrige_exercice_async(soumission_id):
    """
    Tâche asynchrone optimisée pour corriger UN exercice isolé.
    Utilise les données déjà extraites quand disponibles.
    """
    print(f"🚀 === DÉBUT TÂCHE ASYNCHRONE === soumission_id={soumission_id}")

    try:
        soum = SoumissionIA.objects.get(id=soumission_id)
        dem = soum.demande

        print(f"✅ Soumission trouvée: ID={soum.id}")
        print(f"📋 Demande: ID={dem.id}, Fichier={'OUI' if dem.fichier else 'NON'}")
        print(f"🔢 Exercice index: {soum.exercice_index}")
        print(f"🎓 Département: {dem.departement.nom if dem.departement else 'AUCUN'}")

        # Vérifier si on a des données déjà extraites
        if dem.exercices_data:
            print("📦 Données extraction PRÉSENTES dans la demande")
            try:
                data = json.loads(dem.exercices_data)
                nb_ex = len(data.get('exercices', [])) if isinstance(data, dict) else len(data)
                print(f"   → {nb_ex} exercice(s) stocké(s)")
            except:
                print("   → Format JSON invalide")
        else:
            print("📦 Données extraction ABSENTES, extraction nécessaire")

        # Étape 1 : Vérifier si données déjà extraites
        donnees_vision_complete = None
        texte_complet = ""

        if dem.exercices_data:
            try:
                exercices_data = json.loads(dem.exercices_data)
                if isinstance(exercices_data, dict) and "texte_complet" in exercices_data:
                    # Données structurées avec extraction complète
                    texte_complet = exercices_data.get("texte_complet", "")
                    donnees_vision_complete = exercices_data.get("donnees_vision", {})
                    print("✅ Utilisation données extraction déjà stockées")
                else:
                    # Ancien format (juste liste d'exercices)
                    texte_complet = " ".join([ex.get("contenu", "") for ex in exercices_data])
                    print("✅ Utilisation contenu exercices déjà stockés")
            except json.JSONDecodeError:
                print("⚠️ Données exercices mal formatées, re-extraction")

        # Si pas de données stockées, extraction nécessaire
        if not texte_complet and dem.fichier:
            print("🔄 Données non stockées, extraction nécessaire...")
            soum.statut = 'extraction'
            soum.progression = 10
            soum.save()

            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, os.path.basename(dem.fichier.name))
            with open(local_path, "wb") as f:
                for chunk in dem.fichier.chunks():
                    f.write(chunk)

            analyse_complete = analyser_document_scientifique(local_path, departement=dem.departement)

            donnees_vision_complete = {
                "elements_visuels": analyse_complete.get("elements_visuels", []),
                "formules_latex": analyse_complete.get("formules_latex", []),
                "graphs": analyse_complete.get("graphs", []),
                "angles": analyse_complete.get("angles", []),
                "numbers": analyse_complete.get("numbers", []),
                "pipeline_utilise": analyse_complete.get("pipeline", "inconnu")
            }

            texte_complet = analyse_complete.get("texte_complet", "")

            try:
                os.unlink(local_path)
            except:
                pass
        else:
            texte_complet = dem.enonce_texte or ""

        # Étape 2 : Séparation et extraction du fragment
        soum.statut = 'decoupage'
        soum.progression = 30
        soum.save()

        exercices_data = separer_exercices_avec_titres(texte_complet)
        idx = soum.exercice_index or 0

        if idx >= len(exercices_data):
            print(f"⚠️ Index {idx} hors limites, utilisation du dernier exercice")
            idx = len(exercices_data) - 1

        ex_data = exercices_data[idx]
        fragment = ex_data['contenu']

        print(f"✅ Exercice {idx + 1} extrait: {ex_data.get('titre', 'Sans titre')}")
        print(f"   Longueur contenu: {len(fragment)} caractères")

        # Étape 3 : Préparer données vision pour cet exercice
        donnees_vision_exercice = None
        if dem.departement and detecter_departement_scientifique_avance(dem.departement):
            if donnees_vision_complete:
                donnees_vision_exercice = filtrer_donnees_vision_par_exercice(
                    donnees_vision_complete,
                    fragment
                )
                print(f"🔬 Enrichissement scientifique activé pour exercice {idx + 1}")
                print(f"   Schémas pertinents: {len(donnees_vision_exercice.get('elements_visuels', []))}")

        # Étape 4 : Génération du corrigé
        soum.statut = 'analyse_ia'
        soum.progression = 50
        soum.save()

        mat = dem.matiere if dem.matiere else Matiere.objects.first()
        contexte = f"Exercice de {mat.nom} – {ex_data.get('titre', f'Exercice {idx + 1}')}"

        corrige_txt, _ = generer_corrige_par_exercice(
            texte_exercice=fragment,
            contexte=contexte,
            matiere=mat,
            donnees_vision=donnees_vision_exercice,
            demande=dem,
            exercice_index=idx + 1
        )

        # Étape 5 : Génération PDF
        soum.statut = 'formatage_pdf'
        soum.progression = 80
        soum.save()

        from .pdf_utils import generer_pdf_corrige
        pdf_url = generer_pdf_corrige(
            {
                "titre_corrige": contexte,
                "corrige_html": corrige_txt,
                "soumission_id": soum.id,
                "titre_exercice": ex_data.get('titre_complet', f"Exercice {idx + 1}")
            },
            soum.id
        )

        # Étape 6 : Débit de crédit
        if not debiter_credit_abonnement(dem.user):
            soum.statut = 'erreur_credit'
            soum.save()
            return False

        # Étape 7 : Création CorrigePartiel
        pdf_relative_path = pdf_url.replace(settings.MEDIA_URL, '')
        pdf_absolute_path = os.path.join(settings.MEDIA_ROOT, pdf_relative_path)

        titre_reel = ex_data.get('titre_complet', ex_data.get('titre', f"Exercice {idx + 1}"))
        if len(titre_reel) > 200:
            titre_reel = titre_reel[:197] + "..."

        with open(pdf_absolute_path, 'rb') as f:
            corrige = CorrigePartiel.objects.create(
                soumission=soum,
                titre_exercice=titre_reel,
            )
            corrige.fichier_pdf.save(
                f"corrige_{dem.id}_ex{idx + 1}_{soum.id}.pdf",
                File(f)
            )
            corrige.save()

        # Étape 8 : Finalisation
        soum.statut = 'termine'
        soum.progression = 100
        soum.resultat_json = {
            "exercice_index": idx,
            "exercice_titre": titre_reel,
            "corrige_text": corrige_txt,
            "pdf_url": pdf_url,
            "exercice_data": ex_data,
            "pipeline_utilise": donnees_vision_complete.get("pipeline_utilise",
                                                            "standard") if donnees_vision_complete else "standard"
        }
        soum.save()

        print(f"🎉 Exercice {idx + 1} traité avec succès!")
        return True

    except Exception as e:
        print(f"❌ Erreur dans generer_corrige_exercice_async: {e}")
        import traceback
        traceback.print_exc()
        try:
            soum = SoumissionIA.objects.get(id=soumission_id)
            soum.statut = 'erreur'
            soum.save()
        except:
            pass
        return False


@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(demande_id, matiere_id=None):
    """
    Tâche pour traitement complet (conservée pour compatibilité).
    """
    try:
        from correction.models import DemandeCorrection
        demande = DemandeCorrection.objects.get(id=demande_id)
        soumission = SoumissionIA.objects.get(demande=demande)

        # Extraction complète UNE FOIS
        if demande.fichier and not demande.exercices_data:
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, os.path.basename(demande.fichier.name))
            with open(local_path, "wb") as f:
                for chunk in demande.fichier.chunks():
                    f.write(chunk)

            analyse_complete = analyser_document_scientifique(local_path, departement=demande.departement)

            # Stocker pour réutilisation future
            donnees_completes = {
                "texte_complet": analyse_complete.get("texte_complet", ""),
                "donnees_vision": {
                    "elements_visuels": analyse_complete.get("elements_visuels", []),
                    "formules_latex": analyse_complete.get("formules_latex", []),
                    "graphs": analyse_complete.get("graphs", []),
                    "angles": analyse_complete.get("angles", []),
                    "numbers": analyse_complete.get("numbers", []),
                    "pipeline_utilise": analyse_complete.get("pipeline", "inconnu")
                }
            }

            demande.exercices_data = json.dumps(donnees_completes, ensure_ascii=False)
            demande.save()

            try:
                os.unlink(local_path)
            except:
                pass

        soumission.statut = 'termine'
        soumission.progression = 100
        soumission.save()

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


# ========== FONCTIONS UTILITAIRES SUPPLEMENTAIRES ==========
def extraire_exercice_par_index(texte_epreuve, index=0):
    """Fonction utilitaire pour extraire un exercice spécifique."""
    exercices_data = separer_exercices_avec_titres(texte_epreuve)

    if index < 0 or index >= len(exercices_data):
        return None

    ex_data = exercices_data[index]
    ex_data.update({
        'index': index,
        'total_exercices': len(exercices_data),
        'extraction_date': datetime.now().isoformat()
    })

    return ex_data


def obtenir_liste_exercices(texte_epreuve, avec_preview=False):
    """Retourne la liste de tous les exercices détectés."""
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
            lignes = ex['contenu'].split('\n')[:3]
            preview_text = ' '.join([l[:100] for l in lignes if l.strip()])
            item['preview'] = (preview_text[:200] + '...') if len(preview_text) > 200 else preview_text

        result.append(item)

    return result

