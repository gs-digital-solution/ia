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

import math
from typing import Dict, List, Optional, Tuple
from .mathpix_extractor import (
    validate_mathpix_config,
    analyze_scientific_document as mathpix_analyze
)
#from .tasks import generer_un_exercice
#from celery import group
import logging
# Logger dédié
logger = logging.getLogger(__name__)

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
    'MATHEMATIQUES', 'PHYSIQUE', 'CHIMIE', 'biologie', 'svt', 'sciences', 'informatique'
]
def is_departement_scientifique(departement):
    """
    Renvoie True si le département fait partie des filières scientifiques définies globalement.
    """
    if departement and departement.nom:
        dep_name = departement.nom.lower()
        return any(dep_name.startswith(sc) or sc in dep_name for sc in DEPARTEMENTS_SCIENTIFIQUES)
    return False


def extract_with_scientific_workflow(file_path: str, departement) -> Dict:
    """
    Workflow d'extraction scientifique avec Mathpix et analyse avancée.

    Args:
        file_path: Chemin vers le fichier
        departement: Objet département pour vérification

    Returns:
        Dict: Résultats d'analyse enrichis pour DeepSeek Reasoner
    """
    logger.info(f"🔬 Début workflow scientifique pour département: {departement.nom if departement else 'Inconnu'}")

    # 1. Vérifier si Mathpix est configuré
    if not validate_mathpix_config():
        logger.warning("⚠️ Mathpix non configuré, fallback sur OCR standard")
        return {"text": "", "elements_visuels": [], "formules_latex": []}

    try:
        # 2. Analyse scientifique avec Mathpix
        mathpix_results = mathpix_analyze(file_path)

        # 3. Post-traitement pour structurer les résultats
        # Extraire le texte principal
        texte_complet = mathpix_results.get("text", "")

        # Préparer les éléments visuels (équations et tableaux)
        elements_visuels = []

        # Ajouter les équations comme éléments visuels
        for eq in mathpix_results.get("equations", []):
            elements_visuels.append({
                "type": "equation",
                "latex": eq.get("latex", ""),
                "confidence": eq.get("confidence", 0),
                "position": eq.get("bounds", {}),
                "description": f"Équation mathématique (confiance: {eq.get('confidence', 0):.2f})"
            })

        # Ajouter les tableaux comme éléments visuels
        for table in mathpix_results.get("tables", []):
            elements_visuels.append({
                "type": "table",
                "latex": table.get("latex", ""),
                "text": table.get("text", ""),
                "confidence": table.get("confidence", 0),
                "description": f"Tableau détecté (confiance: {table.get('confidence', 0):.2f})"
            })

        # Extraire les blocs LaTeX
        formules_latex = mathpix_results.get("latex_blocks", [])

        # 4. Analyse des schémas avec BLIP (si disponible)
        captions = []
        try:
            from PIL import Image
            img = Image.open(file_path)

            # Détection de schémas basique par analyse d'image
            # Vous pouvez ajouter une détection plus sophistiquée ici
            img_array = np.array(img)

            # Détection de contours pour identifier les schémas
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            edges = cv2.Canny(gray, 50, 150)
            contour_ratio = np.sum(edges > 0) / (img_array.shape[0] * img_array.shape[1])

            if contour_ratio > 0.05:  # Seuil pour détecter un schéma
                try:
                    # Utiliser BLIP pour générer une description
                    blip_processor, blip_model = get_blip_model()
                    inputs = blip_processor(img, return_tensors="pt").to(blip_model.device)
                    out = blip_model.generate(**inputs, max_new_tokens=50)
                    caption = blip_processor.decode(out[0], skip_special_tokens=True)

                    captions.append({
                        "type": "schema",
                        "description": caption,
                        "confidence": 0.7,
                        "contour_density": contour_ratio
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Erreur BLIP: {e}")
                    captions.append({
                        "type": "schema",
                        "description": "Schéma scientifique détecté (non analysé)",
                        "confidence": 0.5,
                        "contour_density": contour_ratio
                    })

        except Exception as e:
            logger.warning(f"⚠️ Erreur analyse image: {e}")

        # 5. Structurer le résultat final
        result = {
            "texte_complet": texte_complet,
            "elements_visuels": elements_visuels + captions,
            "formules_latex": formules_latex,
            "confidence": mathpix_results.get("confidence", 0),
            "extraction_method": "mathpix_scientific",
            "mathpix_raw": mathpix_results,  # Données brutes pour débogage
            "metadata": {
                "departement": departement.nom if departement else "Inconnu",
                "is_scientific": True,
                "file_type": os.path.splitext(file_path)[1],
                "extraction_timestamp": datetime.now().isoformat()
            }
        }

        logger.info(f"✅ Workflow scientifique terminé: {len(texte_complet)} caractères, "
                    f"{len(elements_visuels)} éléments visuels, {len(formules_latex)} formules")

        return result

    except Exception as e:
        logger.error(f"❌ Erreur workflow scientifique: {e}")
        # Fallback sur l'analyse standard
        return analyser_document_scientifique(file_path)

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


def call_deepseek_reasoner(exercice_data: Dict, contexte: str, promptia=None) -> str:
    """
    Appel à DeepSeek Reasoner pour reconstituer rigoureusement l'exercice scientifique.

    Args:
        exercice_data: Données de l'exercice (texte, formules, éléments visuels)
        contexte: Contexte de l'exercice
        promptia: Prompt IA spécifique (optionnel)

    Returns:
        str: Exercice reconstitué et structuré pour DeepSeek Chat
    """
    logger.info("🧠 Appel DeepSeek Reasoner pour reconstruction scientifique")

    # Construire le prompt système spécialisé pour Reasoner
    system_prompt = """Tu es DeepSeek Reasoner, un expert en analyse scientifique rigoureuse.

TON RÔLE : Reconstituer de manière EXHAUSTIVE et STRUCTURÉE un exercice scientifique à partir des données d'extraction.

CONSIGNES ABSOLUES :
1. NE PAS faire le corrigé, seulement reconstituer l'énoncé
2. INCLURE TOUS les éléments : texte, formules, schémas, tableaux
3. DÉCRIRE LES SCHÉMAS de manière précise et complète
4. STRUCTURER avec des sections claires
5. PRÉSERVER la rigueur mathématique/scientifique
6. GÉRER les ambiguïtés de manière logique

FORMAT DE SORTIE OBLIGATOIRE :

=== EXERCICE RECONSTITUÉ ===

[Texte principal de l'exercice exactement comme dans le document]

=== FORMULES MATHÉMATIQUES ===

[Pour chaque formule LaTeX détectée :
• Formule : \[ ... \]
• Contexte : Description de son usage dans l'exercice
• Position : Si disponible, indication de localisation]

=== SCHÉMAS ET FIGURES ===

[Pour chaque élément visuel :
• Type : (schéma, tableau, graphique, diagramme)
• Description détaillée : [Décrire TOUS les éléments, leurs relations, annotations]
• Données numériques : [Extraire toutes les valeurs numériques]
• Légendes : [Inclure toutes les légendes détectées]
• Relation avec l'exercice : [Expliquer comment le schéma s'intègre]]

=== DONNÉES SUPPLÉMENTAIRES ===

[Angles, valeurs numériques, unités spécifiques]

=== CONTEXTE ET PRÉCISIONS ===

[Clarifier les ambiguïtés, préciser les hypothèses raisonnables]

=== STRUCTURE PROPOSÉE POUR LE CORRIGÉ ===

[Suggérer une organisation pour la correction, sans la faire]

FIN DE LA RECONSTITUTION
"""

    # Construire le prompt utilisateur avec toutes les données
    user_prompt_parts = [
        f"CONTEXTE : {contexte}",
        "\n=== DONNÉES BRUTES D'EXTRACTION ===\n"
    ]

    # Ajouter le texte extrait
    if exercice_data.get("texte_complet"):
        user_prompt_parts.append("TEXTE EXTRAIT :")
        user_prompt_parts.append(exercice_data["texte_complet"][:5000])  # Limiter la taille

    # Ajouter les formules LaTeX
    if exercice_data.get("formules_latex"):
        user_prompt_parts.append("\nFORMULES LaTeX DÉTECTÉES :")
        for i, formule in enumerate(exercice_data["formules_latex"][:20], 1):  # Limiter à 20 formules
            user_prompt_parts.append(f"{i}. {formule}")

    # Ajouter les éléments visuels
    if exercice_data.get("elements_visuels"):
        user_prompt_parts.append("\nÉLÉMENTS VISUELS DÉTECTÉS :")
        for i, element in enumerate(exercice_data["elements_visuels"][:15], 1):  # Limiter à 15 éléments
            desc = element.get("description", "Élément sans description")
            elem_type = element.get("type", "inconnu")
            user_prompt_parts.append(f"{i}. [{elem_type.upper()}] {desc}")

    # Ajouter les métadonnées
    metadata = exercice_data.get("metadata", {})
    if metadata:
        user_prompt_parts.append(f"\nMÉTADONNÉES : Département={metadata.get('departement')}, "
                                 f"Scientifique={metadata.get('is_scientific')}")

    user_prompt_parts.append("\n" + "=" * 50)
    user_prompt_parts.append("INSTRUCTIONS FINALES :")
    user_prompt_parts.append("1. Reconstituer l'exercice COMPLET et EXACT")
    user_prompt_parts.append("2. Décrire TOUS les schémas en détail")
    user_prompt_parts.append("3. Structurer pour faciliter la correction")
    user_prompt_parts.append("4. NE PAS faire le corrigé maintenant")

    user_prompt = "\n".join(user_prompt_parts)

    # Préparation de l'appel API
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 8000,
        "top_p": 0.9,
        "frequency_penalty": 0.0
    }

    try:
        logger.info("📡 Appel à DeepSeek Reasoner...")
        response = requests.post(api_url, headers=headers, json=data, timeout=120)
        response.raise_for_status()

        result = response.json()
        reconstructed_exercise = result['choices'][0]['message']['content']

        logger.info(f"✅ Reasoner: {len(reconstructed_exercise)} caractères générés")

        # Vérification de la qualité
        if len(reconstructed_exercise) < 100:
            logger.warning("⚠️ Reconstruction trop courte, utilisation des données brutes")
            return exercice_data.get("texte_complet", "")

        return reconstructed_exercise

    except Exception as e:
        logger.error(f"❌ Erreur DeepSeek Reasoner: {e}")
        # Fallback: retourner le texte brut
        return exercice_data.get("texte_complet", "")

# ── NOUVELLE FONCTION : Analyse scientifique avancée ────

def analyser_document_scientifique(fichier_path: str, departement=None) -> dict:
    """
    Analyse scientifique avancée avec choix automatique du workflow.

    Args:
        fichier_path: Chemin vers le fichier
        departement: Objet département (optionnel, pour décision)

    Returns:
        dict: Résultats d'analyse
    """
    logger.info(f"🔍 Début analyse scientifique pour {fichier_path}")

    # Décision du workflow
    use_scientific_workflow = False
    if departement:
        use_scientific_workflow = is_departement_scientifique(departement)
        logger.info(f"   Département: {departement.nom} → Workflow scientifique: {use_scientific_workflow}")

    # Si département scientifique et Mathpix configuré, utiliser le workflow scientifique
    if use_scientific_workflow and validate_mathpix_config():
        logger.info("   → Utilisation du workflow scientifique avec Mathpix")
        return extract_with_scientific_workflow(fichier_path, departement)

    # Sinon, utiliser le workflow standard avec DeepSeek Vision
    logger.info("   → Utilisation du workflow standard avec DeepSeek Vision")

    # 1) OCR fallback pour avoir un premier texte
    config_tesseract = r'--oem 3 --psm 6 -l fra+eng+digits'
    texte_ocr = ""
    try:
        if fichier_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(fichier_path)
            clean = preprocess_image_for_ocr(img)
            texte_ocr = pytesseract.image_to_string(clean, config=config_tesseract)
            logger.info("    ✓ OCR image brut extrait %d caractères", len(texte_ocr))

        elif fichier_path.lower().endswith('.pdf'):
            texte_ocr = extraire_texte_pdf(fichier_path)
            logger.info("    ✓ PDFMiner extrait %d caractères", len(texte_ocr))
            if len(texte_ocr) < 50:
                logger.warning("    ⚠️ OCR PDFMiner trop court, fallback page à page")
                pages = convert_from_path(fichier_path, dpi=300)
                txts = []
                for page in pages:
                    clean = preprocess_image_for_ocr(page)
                    txts.append(pytesseract.image_to_string(clean, config=config_tesseract))
                texte_ocr = "\n".join(txts)
                logger.info("    ✓ fallback OCR pages donne %d caractères", len(texte_ocr))

        else:
            raise ValueError(f"Format non supporté pour OCR : {fichier_path}")

    except Exception:
        logger.exception("❌ Erreur pendant OCR/PDF pour %s", fichier_path)
        # on ne stoppe pas, on continue avec texte_ocr vide

    # 2) Appel deepseek-vl2 pour tout : texte + schémas + JSON
    try:
        vision_json = call_deepseek_vision(fichier_path)

        # 2a) Texte complet : fallback sur OCR si résultat trop court
        texte_json = vision_json.get("text", "") or ""
        if len(texte_json) < 50:
            texte_json = texte_ocr

        # 2b) Récupération des blocs
        captions = vision_json.get("captions", [])
        latex_blocks = vision_json.get("latex_blocks", [])
        graphs = vision_json.get("graphs", [])
        angles = vision_json.get("angles", [])
        numbers = vision_json.get("numbers", [])
        struct_exos = vision_json.get("structure_exercices", [])

        logger.info("✅ deepseek-vl2 OK : texte %d chars, %d schémas, %d formules, %d angles, %d nombres",
                    len(texte_json), len(captions), len(latex_blocks), len(angles), len(numbers))

        return {
            "texte_complet": texte_json,
            "elements_visuels": captions,
            "formules_latex": latex_blocks,
            "graphs": graphs,
            "angles": angles,
            "numbers": numbers,
            "structure_exercices": struct_exos,
            "extraction_method": "deepseek_vision",
            "metadata": {
                "departement": departement.nom if departement else "Inconnu",
                "is_scientific": False,
                "file_type": os.path.splitext(fichier_path)[1]
            }
        }

    except Exception as e:
        logger.exception("❌ Erreur deepseek-vl2 pour %s: %s", fichier_path, e)
        # fallback minimal
        return {
            "texte_complet": texte_ocr,
            "elements_visuels": [],
            "formules_latex": [],
            "graphs": [],
            "angles": [],
            "numbers": [],
            "structure_exercices": [],
            "extraction_method": "ocr_fallback",
            "metadata": {
                "departement": departement.nom if departement else "Inconnu",
                "is_scientific": False,
                "file_type": os.path.splitext(fichier_path)[1]
            }
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

def generer_corrige_par_exercice(texte_exercice, contexte, matiere=None, donnees_vision=None,demande=None):
    """
    Génère le corrigé pour un seul exercice en exploitant les données vision.

    Args:
        texte_exercice: Texte de l'exercice
        contexte: Contexte de l'exercice
        matiere: Matière concernée
        donnees_vision: Données d'analyse vision (schémas, formules, etc.)

    Returns:
        Tuple (corrige_text, graph_list)
    """
    print("🎯 Génération corrigé avec analyse vision...")
    print("\n[DEBUG] ==> generer_corrige_par_exercice avec demande:",
          getattr(demande, 'id', None), "/", type(demande))

    # 1) Récupère le prompt métier (ou None)
    promptia = get_best_promptia(demande)

    # 2) Construit les deux messages
    contexte = f"Contexte : Exercice de {matiere.nom} – {getattr(demande.classe, 'nom', '')}"
    msg_system, msg_user = build_promptia_messages(promptia, contexte)

    # 3) Enrichir le user_message avec l'exercice et la vision
    user_blocks = [
        msg_user["content"],
        "----- EXERCICE À CORRIGER -----",
        texte_exercice.strip()
    ]
    if donnees_vision:
        # Schémas identifiés
        if donnees_vision.get("elements_visuels"):
            user_blocks.append("----- SCHÉMAS IDENTIFIÉS -----")
            for element in donnees_vision["elements_visuels"]:
                desc = element.get("description", "")
                user_blocks.append(f"- {desc}")

        # Formules LaTeX
        if donnees_vision.get("formules_latex"):
            user_blocks.append("----- FORMULES DÉTECTÉES -----")
            for formule in donnees_vision["formules_latex"]:
                user_blocks.append(f"- {formule}")

        # Données graphiques brutes (JSON)
        if donnees_vision.get("graphs"):
            user_blocks.append("----- DONNÉES GRAPHIQUES (JSON) -----")
            user_blocks.append(
                json.dumps(donnees_vision["graphs"], ensure_ascii=False, indent=2)
            )

        # Angles détectés
        if donnees_vision.get("angles"):
            user_blocks.append("----- ANGLES IDENTIFIÉS -----")
            for angle in donnees_vision["angles"]:
                val = angle.get("valeur", "")
                unit = angle.get("unité", "")
                coord = angle.get("coord", "")
                user_blocks.append(f"- {val}{unit} à coord {coord}")

        # Nombres détectés
        if donnees_vision.get("numbers"):
            user_blocks.append("----- NOMBRES ET UNITÉS -----")
            for num in donnees_vision["numbers"]:
                val = num.get("valeur", "")
                unit = num.get("unité", "")
                coord = num.get("coord", "")
                user_blocks.append(f"- {val}{unit} à coord {coord}")

    # On reconstitue le contenu utilisateur final
    msg_user["content"] = "\n\n".join(user_blocks)

    # 4) Préparation de l’appel API avec deux messages
    data = {
        "model": "deepseek-chat",
        "messages": [msg_system, msg_user],
        "temperature": 0.1,
        "max_tokens": 6000,
        "top_p": 0.9,
        "frequency_penalty": 0.1
    }
    # URL et en-têtes pour l'appel DeepSeek
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",  # Assurez-vous que DEEPSEEK_API_KEY est dans vos env vars
        "Content-Type": "application/json"
    }
    try:
        print("📡 Appel API DeepSeek avec analyse vision...")

        # Tentative avec vérification de qualité
        output = None
        for tentative in range(2):  # Maximum 2 tentatives
            response = requests.post(api_url, headers=headers, json=data, timeout=90)
            response_data = response.json()

            if response.status_code != 200:
                error_msg = f"Erreur API: {response_data.get('message', 'Pas de détail')}"
                print(f"❌ {error_msg}")
                return error_msg, None

            # Récupération de la réponse
            output = response_data['choices'][0]['message']['content']
            print(f"✅ Réponse IA brute (tentative {tentative + 1}): {len(output)} caractères")

            # Vérification de la qualité
            if verifier_qualite_corrige(output, texte_exercice):
                print("✅ Qualité du corrigé validée")
                break
            else:
                print(f"🔄 Tentative {tentative + 1} - Qualité insuffisante, régénération...")
                # Ajouter une consigne de rigueur pour la prochaine tentative
                data["messages"][1][
                    "content"] += "\n\n⚠️ ATTENTION : Sois plus rigoureux ! Exploite mieux les schémas identifiés. Vérifie tous tes calculs."

                if tentative == 0:  # Attendre un peu avant la 2ème tentative
                    import time
                    time.sleep(2)
        else:
            print("❌ Échec après 2 tentatives - qualité insuffisante")
            return "Erreur: Qualité du corrigé insuffisante après plusieurs tentatives", None

        # Traitement de la réponse (identique à avant)
        output = response_data['choices'][0]['message']['content']
        print("✅ Réponse IA brute (début):")
        print(output[:500].replace("\n", "\\n"))
        print("… (total", len(output), "caractères)\n")

        output = flatten_multiline_latex_blocks(output)
        print("🛠️ Après flatten_multiline_latex_blocks (début):")
        print(output[:500].replace("\n", "\\n"))
        print("… (total", len(output), "caractères)\n")

        output_structured = format_corrige_pdf_structure(output)
        print("🧩 output_structured après format_corrige_pdf_structure:")
        print(output_structured[:500].replace("\n", "\\n"), "\n…\n")

        # Initialisation des variables de retour
        corrige_txt = output_structured
        graph_list = []

        # Extraction graphique
        json_blocks = extract_json_blocks(output_structured)
        print(f"🔍 JSON blocks détectés : {len(json_blocks)}")

        # Afficher chaque JSON brut
        for i, (graph_dict, start, end) in enumerate(json_blocks, start=1):
            raw_json = output_structured[start:end]
            print(f"   ▶️ Bloc JSON {i} brut:")
            print(raw_json.replace("\n", "\\n"))
            print("   ▶️ Parsed Python dict :", graph_dict)

        # Traitement des graphiques (identique à avant)
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

        print("📝 Corrigé final (début) :")
        print(corrige_txt[:1000].replace("\n", "\\n"))
        print("… fin extrait Corrigé\n")

        return corrige_txt.strip(), graph_list

    except Exception as e:
        error_msg = f"Erreur: {str(e)}"
        print(f"❌ {error_msg}")
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


def format_table_statistique(table_data, title=None):
    """
    Formate un tableau statistique en HTML à partir de données structurées.

    Args:
        table_data: dict ou list avec les données du tableau
        title: titre optionnel du tableau

    Returns:
        str: HTML formaté du tableau
    """
    if not table_data:
        return ""

    # Si c'est un dict avec des listes
    if isinstance(table_data, dict):
        headers = list(table_data.keys())
        # Trouver la longueur maximale des listes
        max_len = max(len(v) if isinstance(v, list) else 1 for v in table_data.values())

        html = ['<div class="stat-table-container">']
        if title:
            html.append(f'<h4 class="table-title">{title}</h4>')

        html.append('<table class="stat-table">')
        html.append('<thead><tr>')
        for header in headers:
            html.append(f'<th>{header}</th>')
        html.append('</tr></thead>')
        html.append('<tbody>')

        for i in range(max_len):
            html.append('<tr>')
            for header in headers:
                value = table_data[header]
                if isinstance(value, list) and i < len(value):
                    cell = str(value[i])
                elif not isinstance(value, list):
                    cell = str(value)
                else:
                    cell = ""
                html.append(f'<td>{cell}</td>')
            html.append('</tr>')

        html.append('</tbody></table></div>')
        return "".join(html)

    # Si c'est déjà du HTML ou du markdown, le renvoyer tel quel
    return str(table_data)


def generate_corrige_html(corrige_text):
    """Transforme le corrigé brut en HTML stylisé, aéré, avec blocs d'exercices,
    titres mis en valeur, formatage MathJax et tableaux conservés, et branding CIS au début."""
    if not corrige_text:
        return ""

    # Formatage des expressions mathématiques (Latex) et tableaux
    lines = corrige_text.strip().split('\n')

    # Pattern pour détecter les débuts d'exercice/partie
    pattern_exercice = re.compile(r'^(EXERCICE\s*\d+|PARTIE\s*[IVXLCDM]+|Exercice\s*\d+|Partie\s*[IVXLCDM]+)',
                                  re.IGNORECASE)
    html_output = []
    i = 0

    # Variables d'état
    in_bloc_exercice = False
    in_stat_table = False
    in_variation_block = False
    in_table_markdown = False
    table_lines = []
    variation_lines = []
    markdown_table_lines = []

    # Branding CIS en haut
    html_output.append(
        '<div class="cis-message"><strong>SUJET CORRIGÉ PAR L\'APPLICATION CIS, DISPO SUR PLAYSTORE</strong></div>')

    while i < len(lines):
        line = lines[i].strip()

        # ========== GESTION DES TABLEAUX MARKDOWN DÉTECTÉS ==========
        # Détecter le début d'un tableau markdown (ligne avec | et ligne suivante avec |- ou |:)
        if line.startswith('|') and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith('|') and ('---' in next_line or '===' in next_line or ':|' in next_line):
                in_table_markdown = True
                markdown_table_lines = [line]
                i += 1
                continue

        # Collecter les lignes du tableau markdown
        if in_table_markdown:
            if line.startswith('|'):
                markdown_table_lines.append(line)
                i += 1
                continue
            else:
                # Fin du tableau markdown - le formater
                if len(markdown_table_lines) >= 2:  # Au moins l'en-tête et la ligne de séparation
                    html_table = format_table_markdown('\n'.join(markdown_table_lines))
                    # Détecter si c'est un tableau statistique
                    is_stat_table = any(keyword in markdown_table_lines[0].lower()
                                        for keyword in
                                        ['classe', 'effectif', 'fréquence', 'ecc', 'ecd', 'centre', 'note'])
                    if is_stat_table:
                        html_output.append('<div class="statistical-table-wrapper">')
                        html_output.append('<p class="table-title"><strong>📊 Tableau statistique :</strong></p>')
                        html_output.append(html_table)
                        html_output.append('</div>')
                    else:
                        html_output.append('<div class="general-table-wrapper">')
                        html_output.append(html_table)
                        html_output.append('</div>')
                in_table_markdown = False
                markdown_table_lines = []
                # Ne pas incrémenter i, traiter cette ligne normalement
                continue

        # ========== GESTION DES TABLEAUX STATISTIQUES SANS FORMAT MARKDOWN ==========
        # Détecter les tableaux statistiques décrits en texte
        stat_keywords = ['classes', 'effectifs', 'fréquences', 'ecc', 'ecd', 'centre de classe', 'notes', 'valeurs']
        if (any(keyword in line.lower() for keyword in stat_keywords) and
                len(line) > 30 and
                not line.startswith('\\') and
                'tableau' not in line.lower()):  # Éviter les titres "Tableau 1:"

            # Vérifier si les prochaines lignes contiennent des données tabulaires
            j = i + 1
            data_lines = []
            while j < len(lines) and j < i + 10:  # Regarder les 10 lignes suivantes
                if lines[j].strip() and ('|' in lines[j] or ';' in lines[j] or
                                         re.match(r'^\[.*\]\s+\d+', lines[j].strip()) or
                                         re.match(r'^\d+\s*[-–]\s*\d+\s+\d+', lines[j].strip())):
                    data_lines.append(lines[j].strip())
                j += 1

            if data_lines:
                # C'est probablement un tableau statistique - le formater proprement
                html_output.append('<div class="statistical-table-wrapper">')
                html_output.append('<p class="table-title"><strong>📊 Tableau statistique :</strong></p>')

                # Ajouter la ligne de description
                html_output.append(f'<p class="table-description">{line}</p>')

                # Formater les données en tableau HTML
                html_output.append('<table class="statistical-table"><tbody>')

                for data_line in data_lines:
                    html_output.append('<tr>')
                    # Différents formats de données
                    if '|' in data_line:
                        cells = [cell.strip() for cell in data_line.split('|') if cell.strip()]
                    elif ';' in data_line:
                        cells = [cell.strip() for cell in data_line.split(';')]
                    else:
                        # Essayer de parser d'autres formats
                        cells = re.split(r'\s{2,}', data_line)

                    for cell in cells:
                        html_output.append(f'<td>{cell}</td>')
                    html_output.append('</tr>')

                html_output.append('</tbody></table></div>')
                i = j  # Sauter toutes les lignes traitées
                continue

        # ========== GESTION DES DESCRIPTIONS DE VARIATIONS ==========
        # Détecter le début d'une description de variations/signe
        variation_keywords = ['croissant', 'décroissant', 'signe', 'variation', 'monotonie',
                              'positive', 'négative', 's\'annule', 'minimum', 'maximum', 'extremum',
                              'strictement']

        if (any(keyword in line.lower() for keyword in variation_keywords) and
                not line.startswith('|') and
                not line.startswith('\\') and
                len(line) > 15 and  # Éviter les très courtes lignes
                'tableau' not in line.lower()):  # Éviter les références aux tableaux

            in_variation_block = True
            variation_lines = [line]
            i += 1
            continue

        # Collecter les lignes de description de variations
        if in_variation_block:
            # Continuer tant que la ligne n'est pas vide ou ne commence pas un nouveau bloc
            if (line and not pattern_exercice.match(line) and
                    not line.startswith('Question') and not line.startswith('|')):
                variation_lines.append(line)
                i += 1
                continue
            else:
                # Fin du bloc de variations - le formater
                if variation_lines:
                    variation_text = ' '.join(variation_lines)
                    html_output.append('<div class="variation-description-wrapper">')
                    html_output.append(
                        '<p class="variation-title"><strong>📈 Étude des variations et du signe :</strong></p>')

                    # Séparer en paragraphes si c'est long
                    if len(variation_text) > 150:
                        sentences = re.split(r'[.!?]', variation_text)
                        for sentence in sentences:
                            if sentence.strip():
                                html_output.append(f'<p class="variation-content">{sentence.strip()}.</p>')
                    else:
                        html_output.append(f'<p class="variation-content">{variation_text}</p>')

                    html_output.append('</div>')
                in_variation_block = False
                variation_lines = []
                # Ne pas incrémenter i, traiter cette ligne normalement
                continue

        # ========== TRAITEMENT NORMAL DES LIGNES ==========
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

        # Listes avec puces
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            html_output.append(f'<p class="list-item">{line}</p>')
            i += 1
            continue

        # Énumérations numérotées
        if re.match(r'^\d+[\).]', line):
            html_output.append(f'<p class="numbered-item">{line}</p>')
            i += 1
            continue

        # Formules LaTeX (inline)
        if '\\(' in line and '\\)' in line:
            html_output.append(f'<p class="reponse-question mathjax-inline">{line}</p>')
            i += 1
            continue

        # Formules LaTeX (display)
        if '\\[' in line and '\\]' in line:
            html_output.append(f'<div class="mathjax-display">{line}</div>')
            i += 1
            continue

        # Titres de sections dans les réponses
        if line.endswith(':') and len(line) < 50 and not line.startswith('http'):
            html_output.append(f'<p class="section-title"><strong>{line}</strong></p>')
            i += 1
            continue

        # Cas général : paragraphe de réponse ou explication
        html_output.append(f'<p class="reponse-question">{line}</p>')
        i += 1

    # ========== FERMETURE DES BLOCS EN COURS ==========
    # Fermer le dernier tableau markdown si toujours ouvert
    if in_table_markdown and markdown_table_lines:
        html_table = format_table_markdown('\n'.join(markdown_table_lines))
        is_stat_table = any(keyword in markdown_table_lines[0].lower()
                            for keyword in ['classe', 'effectif', 'fréquence', 'ecc', 'ecd'])
        if is_stat_table:
            html_output.append('<div class="statistical-table-wrapper">')
            html_output.append('<p class="table-title"><strong>📊 Tableau statistique :</strong></p>')
            html_output.append(html_table)
            html_output.append('</div>')
        else:
            html_output.append('<div class="general-table-wrapper">')
            html_output.append(html_table)
            html_output.append('</div>')

    # Fermer le dernier bloc de variations si toujours ouvert
    if in_variation_block and variation_lines:
        variation_text = ' '.join(variation_lines)
        html_output.append('<div class="variation-description-wrapper">')
        html_output.append('<p class="variation-title"><strong>📈 Étude des variations et du signe :</strong></p>')

        if len(variation_text) > 150:
            sentences = re.split(r'[.!?]', variation_text)
            for sentence in sentences:
                if sentence.strip():
                    html_output.append(f'<p class="variation-content">{sentence.strip()}.</p>')
        else:
            html_output.append(f'<p class="variation-content">{variation_text}</p>')

        html_output.append('</div>')

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
def extraire_texte_fichier(fichier_field):
    """
    Extraction robuste via analyse scientifique avec fallback OCR pour images.
    """
    if not fichier_field:
        return ""

    # 1) Sauvegarde locale
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))
    with open(local_path, "wb") as f:
        for chunk in fichier_field.chunks():
            f.write(chunk)

    # 2) Détecter le type de fichier
    ext = os.path.splitext(local_path)[1].lower()

    # 3) Pour les images, essayer d'abord un OCR simple et rapide
    texte = ""
    if ext in ['.png', '.jpg', '.jpeg']:
        print(f"🖼️  Fichier image détecté: {ext}, tentative OCR Tesseract...")
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(local_path)

            # Préprocess pour améliorer l'OCR
            image = image.convert('L')  # Niveaux de gris
            texte = pytesseract.image_to_string(image, lang='fra+eng')
            print(f"✅ OCR Tesseract réussi: {len(texte)} caractères")

            if len(texte) > 100:  # Si l'OCR a bien fonctionné
                # Nettoyer
                try:
                    os.unlink(local_path)
                except:
                    pass
                return texte.strip()
            else:
                print("⚠️  OCR Tesseract a retourné peu de texte, essai DeepSeek...")
        except Exception as e:
            print(f"⚠️  OCR Tesseract échoué: {e}, passage à DeepSeek...")

    # 4) Appel à l'analyse scientifique (DeepSeek) - pour PDF et images avec OCR faible
    try:
        analyse = analyser_document_scientifique(local_path)
        texte = analyse.get("texte_complet", "")
        print(f"🔬 Analyse scientifique: {len(texte)} caractères")
    except Exception as e:
        print(f"❌ Analyse scientifique échouée: {e}")
        texte = ""

    # 5) Fallback final pour images si tout échoue
    if not texte or len(texte) < 50:
        if ext in ['.png', '.jpg', '.jpeg']:
            print("🔄 Fallback final: OCR brut sans prétraitement...")
            try:
                import pytesseract
                from PIL import Image
                image = Image.open(local_path)
                texte = pytesseract.image_to_string(image, lang='fra+eng')
                print(f"✅ Fallback OCR: {len(texte)} caractères")
            except Exception as e:
                print(f"❌ Tous les OCR ont échoué: {e}")
                texte = "Impossible d'extraire le texte de cette image."

    # 6) Nettoyage
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

🔬 **NOUVELLES CONSIGNES POUR LES TABLEAUX :**

### POUR LES VARIATIONS ET SIGNE DE FONCTIONS :
- NE JAMAIS produire de tableau de variations ou de signe
- À la place, DÉCRIRE clairement les variations et signes par intervalles
- Utiliser un langage naturel et pédagogique

**EXEMPLE DE DESCRIPTION AU LIEU D'UN TABLEAU :**
"La fonction f est définie sur [-2, 4]. Elle est :
- Strictement croissante sur [-2, 1] avec f(-2)=3 et f(1)=8
- Strictement décroissante sur [1, 4] avec f(4)=2
Le signe de f(x) est :
- Positif sur [-2, 3] car f(x) > 0
- Négatif sur [3, 4] car f(x) < 0
- S'annule en x=3 (f(3)=0)"

### POUR LES TABLEAUX STATISTIQUES :
- Toujours présenter les tableaux statistiques de façon CLAIRE et LISIBLE
- Utiliser le format Markdown pour les tableaux avec | et -
- Inclure TOUTES les colonnes nécessaires : classes, effectifs, fréquences, ECC, ECD, etc.
- Ajouter une brève explication au-dessus du tableau

**EXEMPLE DE TABLEAU STATISTIQUE :**
"Voici le tableau statistique complet :

| Classes (en €) | Centre classe | Effectif | Fréquence (%) | ECC   | ECD   |
|----------------|---------------|----------|---------------|-------|-------|
| [20, 40[       | 30            | 4        | 8%            | 4     | 50    |
| [40, 60[       | 50            | 6        | 12%           | 10    | 46    |
| [60, 80[       | 70            | 25       | 50%           | 35    | 40    |
| [80, 100[      | 90            | 5        | 10%           | 40    | 15    |
| [100, 120[     | 110           | 10       | 20%           | 50    | 10    |

*ECC = Effectifs Cumulés Croissants, ECD = Effectifs Cumulés Décroissants*"

### RÈGLES ABSOLUES :
1. ✅ Pour variations/signes : TOUJOURS une description textuelle, JAMAIS de tableau
2. ✅ Pour statistiques : TOUJOURS un tableau Markdown clair avec explications
3. ✅ Les descriptions doivent être PÉDAGOGIQUES et EXPLICITES
4. ✅ Inclure les valeurs aux bornes quand elles sont importantes
5. ✅ Mentionner les annulations et extremums
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
    Nouvelle version avec système unifié d'extraction.
    """
    print("\n[DEBUG] --> generer_corrige_ia_et_graphique called avec demande:",
          getattr(demande, 'id', None), "/",
          type(demande))

    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    print("\n" + "=" * 60)
    print("🚀 DÉBUT TRAITEMENT INTELLIGENT AVEC VISION (SYSTÈME UNIFIÉ)")
    print("=" * 60)
    print(f"📏 Longueur texte: {len(texte_enonce)} caractères")

    # Données vision
    if donnees_vision:
        print(f"🔬 Données vision disponibles:")
        print(f"   - Éléments visuels: {len(donnees_vision.get('elements_visuels', []))}")
        print(f"   - Formules LaTeX: {len(donnees_vision.get('formules_latex', []))}")

    # 1. ESTIMER LA COMPLEXITÉ
    tokens_estimes = estimer_tokens(texte_enonce)

    # 2. DÉCISION : TRAITEMENT DIRECT OU DÉCOUPÉ
    if tokens_estimes < 1500:  # Épreuve courte
        print("🎯 Décision: TRAITEMENT DIRECT (épreuve courte)")
        return generer_corrige_direct(texte_enonce, contexte, lecons_contenus, exemples_corriges, matiere,
                                      donnees_vision, demande=demande)
    else:  # Épreuve longue
        print("🎯 Décision: DÉCOUPAGE (épreuve longue)")
        # Utiliser la nouvelle version unifiée
        return generer_corrige_decoupe(texte_enonce, contexte, matiere, donnees_vision, demande=demande)

#les fonctions utilitaires , utilisables ou non, donc optionnelles
def extraire_exercice_par_index(texte_epreuve, index=0):
    """
    Fonction utilitaire pour extraire un exercice spécifique par son index.
    Utile pour les API et le frontend.
    """
    exercices_data = separer_exercices_avec_titres(texte_epreuve)

    if index < 0 or index >= len(exercices_data):
        return None

    ex_data = exercices_data[index]

    # Ajouter des métadonnées utiles
    ex_data.update({
        'index': index,
        'total_exercices': len(exercices_data),
        'extraction_date': datetime.now().isoformat()  # ← datetime IMPORTÉ
    })

    return ex_data


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


# ============== TÂCHE ASYNCHRONE ==============
@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(demande_id, matiere_id=None):
    from correction.models import DemandeCorrection, SoumissionIA
    from resources.models import Matiere

    try:
        # Récupération de la demande et création de la soumission IA
        demande = DemandeCorrection.objects.get(id=demande_id)
        soumission = SoumissionIA.objects.get(demande=demande)

        # Étape 1 : Extraction du texte brut AVEC WORKFLOW SCIENTIFIQUE
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

            # 2) Analyse scientifique avec choix automatique du workflow
            #    La fonction analyser_document_scientifique gère automatiquement
            #    le choix entre workflow scientifique et standard
            analyse_complete = analyser_document_scientifique(
                local_path,
                departement=demande.departement
            )

            # 3) Si département scientifique, utiliser DeepSeek Reasoner pour reconstruction
            if is_departement_scientifique(demande.departement):
                logger.info(f"⚗️ Département scientifique détecté: {demande.departement.nom}")

                # Préparer le contexte
                matiere = Matiere.objects.get(id=matiere_id) if matiere_id else demande.matiere
                contexte = f"Exercice de {matiere.nom} - {demande.classe.nom if demande.classe else ''}"

                # Appeler Reasoner pour reconstruction rigoureuse
                soumission.statut = 'reconstruction_scientifique'
                soumission.progression = 30
                soumission.save()

                try:
                    exercice_reconstruit = call_deepseek_reasoner(
                        exercice_data=analyse_complete,
                        contexte=contexte,
                        promptia=get_best_promptia(demande)
                    )

                    # Remplacer le texte brut par la version reconstruite
                    texte_brut = exercice_reconstruit
                    logger.info(f"✅ Exercice reconstruit par Reasoner: {len(texte_brut)} caractères")

                    # Ajouter un flag dans les données vision
                    analyse_complete["reasoner_reconstructed"] = True
                    analyse_complete["reconstructed_text"] = texte_brut

                except Exception as e:
                    logger.error(f"❌ Erreur Reasoner, fallback sur extraction standard: {e}")
                    texte_brut = analyse_complete.get("texte_complet", "")
            else:
                # Département non-scientifique : workflow standard
                texte_brut = analyse_complete.get("texte_complet", "")

            donnees_vision_complete = {
                "elements_visuels": analyse_complete.get("elements_visuels", []),
                "formules_latex": analyse_complete.get("formules_latex", []),
                "graphs": analyse_complete.get("graphs", []),
                "angles": analyse_complete.get("angles", []),
                "numbers": analyse_complete.get("numbers", []),
                "structure_exercices": analyse_complete.get("structure_exercices", []),
                "extraction_method": analyse_complete.get("extraction_method", "unknown"),
                "metadata": analyse_complete.get("metadata", {}),
                "reasoner_used": is_departement_scientifique(demande.departement),
                "confidence": analyse_complete.get("confidence", 0)
            }

            # 4) Nettoyage du fichier temporaire
            try:
                os.unlink(local_path)
            except Exception as e:
                logger.warning(f"⚠️ Erreur suppression fichier temporaire: {e}")
        else:
            # Si pas de fichier, utiliser le texte de l'énoncé
            texte_brut = demande.enonce_texte or ""
            donnees_vision_complete = {
                "elements_visuels": [],
                "formules_latex": [],
                "graphs": [],
                "angles": [],
                "numbers": [],
                "structure_exercices": [],
                "extraction_method": "direct_text",
                "metadata": {
                    "departement": demande.departement.nom if demande.departement else "Inconnu",
                    "is_scientific": is_departement_scientifique(demande.departement)
                },
                "reasoner_used": False
            }

        logger.info(f"📥 TEXTE FINAL POUR CORRECTION: {len(texte_brut)} caractères")
        if texte_brut:
            logger.info(f"   Extrait (500 premiers chars): {texte_brut[:500].replace(chr(10), '\\n')}...")

        # Étape 1b : Extraire les exercices et stocker les données (uniquement si texte suffisant)
        if len(texte_brut) > 100:
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
        else:
            exercices_data = []
            demande.exercices_data = json.dumps([])

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
        is_scientific = is_departement_scientifique(departement)

        if is_scientific:
            logger.info(f"⚗️ Département scientifique : {departement.nom}")
            soumission.statut = 'generation_graphiques'
            soumission.progression = 60
            soumission.save()
        else:
            logger.info(
                f"⚡ Département non scientifique ({departement.nom if departement else 'inconnu'}), skip graphiques")

        # APPEL AVEC DONNÉES VISION
        corrige_txt, graph_list = generer_corrige_ia_et_graphique(
            texte_enonce=texte_enonce,
            contexte=contexte,
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
            logger.error("❌ Crédits insuffisants pour l'utilisateur")
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
            'exercices_titres': [ex['titre'] for ex in exercices_data],
            'is_scientific': is_scientific,
            'reasoner_used': donnees_vision_complete.get('reasoner_used', False)
        }
        soumission.save()

        demande.corrigé = corrige_txt
        demande.save()

        logger.info("🎉 TRAITEMENT AVEC VISION TERMINÉ AVEC SUCCÈS!")
        logger.info(f"   Exercices détectés: {len(exercices_data)}")
        logger.info(f"   Département scientifique: {is_scientific}")
        logger.info(f"   Reasoner utilisé: {donnees_vision_complete.get('reasoner_used', False)}")

        for i, ex in enumerate(exercices_data, 1):
            logger.info(f"   {i}. {ex['titre'][:50]}...")

        return True

    except Exception as e:
        logger.exception(f"❌ ERREUR dans la tâche IA: {e}")
        try:
            soumission.statut = 'erreur'
            soumission.save()
        except:
            pass
        return False


@shared_task(name='correction.ia_utils.generer_corrige_exercice_async')
def generer_corrige_exercice_async(soumission_id):
    """
    Tâche asynchrone pour corriger UN exercice isolé.
    Version mise à jour avec workflow scientifique intégré.
    """
    try:
        soum = SoumissionIA.objects.get(id=soumission_id)
        dem = soum.demande

        # Étape 1 : Préparation de l'extraction
        soum.statut = 'extraction'
        soum.progression = 10
        soum.save()

        donnees_vision_complete = None
        texte_complet = ""

        if dem.fichier:
            # 1) Sauvegarde locale
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, os.path.basename(dem.fichier.name))
            with open(local_path, "wb") as f:
                for chunk in dem.fichier.chunks():
                    f.write(chunk)

            # 2) Analyse scientifique avec choix automatique du workflow
            analyse_complete = analyser_document_scientifique(
                local_path,
                departement=dem.departement
            )

            # 3) Si département scientifique, utiliser DeepSeek Reasoner pour reconstruction
            if is_departement_scientifique(dem.departement):
                logger.info(f"⚗️ Département scientifique détecté: {dem.departement.nom}")

                # Préparer le contexte
                mat = dem.matiere if dem.matiere else Matiere.objects.first()
                contexte = f"Exercice isolé de {mat.nom} - {dem.classe.nom if dem.classe else ''}"

                # Appeler Reasoner pour reconstruction rigoureuse
                soum.statut = 'reconstruction_scientifique'
                soum.progression = 20
                soum.save()

                try:
                    exercice_reconstruit = call_deepseek_reasoner(
                        exercice_data=analyse_complete,
                        contexte=contexte,
                        promptia=get_best_promptia(dem)
                    )

                    # Utiliser la version reconstruite
                    texte_complet = exercice_reconstruit
                    logger.info(f"✅ Exercice reconstruit par Reasoner: {len(texte_complet)} caractères")

                    # Ajouter un flag dans les données vision
                    analyse_complete["reasoner_reconstructed"] = True
                    analyse_complete["reconstructed_text"] = texte_complet

                except Exception as e:
                    logger.error(f"❌ Erreur Reasoner, fallback sur extraction standard: {e}")
                    texte_complet = analyse_complete.get("texte_complet", "")
            else:
                # Département non-scientifique : workflow standard
                texte_complet = analyse_complete.get("texte_complet", "")

            # Préparer les données vision pour la correction
            donnees_vision_complete = {
                "elements_visuels": analyse_complete.get("elements_visuels", []),
                "formules_latex": analyse_complete.get("formules_latex", []),
                "graphs": analyse_complete.get("graphs", []),
                "angles": analyse_complete.get("angles", []),
                "numbers": analyse_complete.get("numbers", []),
                "structure_exercices": analyse_complete.get("structure_exercices", []),
                "extraction_method": analyse_complete.get("extraction_method", "unknown"),
                "metadata": analyse_complete.get("metadata", {}),
                "reasoner_used": is_departement_scientifique(dem.departement),
                "confidence": analyse_complete.get("confidence", 0)
            }

            # 4) Nettoyage
            try:
                os.unlink(local_path)
            except Exception as e:
                logger.warning(f"⚠️ Erreur suppression fichier temporaire: {e}")
        else:
            # Si pas de fichier, utiliser le texte de l'énoncé
            texte_complet = dem.enonce_texte or ""
            donnees_vision_complete = {
                "elements_visuels": [],
                "formules_latex": [],
                "graphs": [],
                "angles": [],
                "numbers": [],
                "structure_exercices": [],
                "extraction_method": "direct_text",
                "metadata": {
                    "departement": dem.departement.nom if dem.departement else "Inconnu",
                    "is_scientific": is_departement_scientifique(dem.departement)
                },
                "reasoner_used": False
            }

        logger.info(f"📥 TEXTE COMPLET: {len(texte_complet)} caractères")

        # 2) Séparer et extraire le fragment spécifique
        soum.statut = 'decoupage_exercices'
        soum.progression = 30
        soum.save()

        exercices_data = separer_exercices_avec_titres(texte_complet)
        idx = soum.exercice_index or 0

        # Vérifier l'index
        if idx >= len(exercices_data):
            logger.warning(f"⚠️ Index {idx} hors limites, utilisation du dernier exercice")
            idx = len(exercices_data) - 1

        ex_data = exercices_data[idx]
        fragment = ex_data['contenu']

        logger.info(f"✅ Exercice {idx + 1} extrait: {ex_data.get('titre', 'Sans titre')}")
        logger.info(f"   Longueur contenu: {len(fragment)} caractères")

        # 3) Préparer les données vision spécifiques à cet exercice
        # Filtrer les éléments visuels pertinents pour cet exercice
        exercice_donnees_vision = None
        if donnees_vision_complete and len(fragment) > 50:
            # On pourrait implémenter une logique pour filtrer les éléments visuels
            # pertinents pour cet exercice spécifique, mais pour l'instant on garde tout
            exercice_donnees_vision = donnees_vision_complete.copy()

            # Ajouter des métadonnées spécifiques à l'exercice
            exercice_donnees_vision["exercice_specific"] = {
                "index": idx,
                "titre": ex_data.get('titre', ''),
                "titre_complet": ex_data.get('titre_complet', ''),
                "longueur_contenu": ex_data.get('longueur_contenu', 0)
            }

        # 4) Mise à jour statut pour analyse IA
        soum.statut = 'analyse_ia'
        soum.progression = 40
        soum.save()

        # 5) Lancer la génération (IA + graph) sur ce fragment avec données vision
        mat = dem.matiere if dem.matiere else Matiere.objects.first()
        contexte = f"Exercice de {mat.nom} – {ex_data.get('titre', f'Exercice {idx + 1}')}"

        corrige_txt, _ = generer_corrige_ia_et_graphique(
            texte_enonce=fragment,
            contexte=contexte,
            matiere=mat,
            donnees_vision=exercice_donnees_vision,
            demande=dem
        )

        # 6) Mise à jour PDF
        soum.statut = 'formatage_pdf'
        soum.progression = 60
        soum.save()

        pdf_url = generer_pdf_corrige(
            {
                "titre_corrige": contexte,
                "corrige_html": corrige_txt,
                "soumission_id": soum.id,
                "titre_exercice": ex_data.get('titre_complet', f"Exercice {idx + 1}"),
                "exercice_index": idx,
                "total_exercices": len(exercices_data)
            },
            soum.id
        )

        # 7) Débit de crédit
        if not debiter_credit_abonnement(dem.user):
            soum.statut = 'erreur_credit'
            soum.save()
            logger.error("❌ Crédits insuffisants pour l'utilisateur")
            return False

        # 8) CRÉATION DU CorrigePartiel - AVEC TITRE RÉEL
        pdf_relative_path = pdf_url.replace(settings.MEDIA_URL, '')
        pdf_absolute_path = os.path.join(settings.MEDIA_ROOT, pdf_relative_path)

        # Utiliser le titre réel de l'exercice
        titre_reel = ex_data.get('titre_complet', ex_data.get('titre', f"Exercice {idx + 1}"))

        # Nettoyer un peu le titre si trop long
        if len(titre_reel) > 200:
            titre_reel = titre_reel[:197] + "..."

        # Vérifier que le fichier PDF existe
        if not os.path.exists(pdf_absolute_path):
            logger.error(f"❌ Fichier PDF non trouvé: {pdf_absolute_path}")
            raise FileNotFoundError(f"PDF non généré: {pdf_absolute_path}")

        # Ouvre le fichier PDF
        with open(pdf_absolute_path, 'rb') as f:
            # Crée le CorrigePartiel avec le VRAI titre
            corrige = CorrigePartiel.objects.create(
                soumission=soum,
                titre_exercice=titre_reel,
            )
            # Attache le fichier PDF
            corrige.fichier_pdf.save(
                f"corrige_{dem.id}_ex{idx + 1}_{soum.id}.pdf",
                File(f)
            )
            corrige.save()

        # 9) Finalisation
        soum.statut = 'termine'
        soum.progression = 100
        soum.resultat_json = {
            "exercice_index": idx,
            "exercice_titre": titre_reel,
            "corrige_text": corrige_txt,
            "pdf_url": pdf_url,
            "exercice_data": ex_data,
            "total_exercices": len(exercices_data),
            "analyse_vision": exercice_donnees_vision,
            "is_scientific": is_departement_scientifique(dem.departement),
            "reasoner_used": donnees_vision_complete.get('reasoner_used', False) if donnees_vision_complete else False,
            "metadata": {
                "departement": dem.departement.nom if dem.departement else "Inconnu",
                "matiere": mat.nom if mat else "Inconnue",
                "classe": dem.classe.nom if dem.classe else "Inconnue"
            }
        }
        soum.save()

        logger.info(f"🎉 Correction exercice {idx + 1} terminée avec succès!")
        logger.info(f"   Titre: {titre_reel[:50]}...")
        logger.info(f"   PDF généré: {pdf_url}")
        logger.info(f"   Département scientifique: {is_departement_scientifique(dem.departement)}")

        return True

    except Exception as e:
        logger.exception(f"❌ Erreur dans generer_corrige_exercice_async: {e}")
        try:
            soum = SoumissionIA.objects.get(id=soumission_id)
            soum.statut = 'erreur'
            soum.save()
        except:
            pass
        return False