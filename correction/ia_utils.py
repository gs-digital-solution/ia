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
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import base64
import functools
from typing import Dict, Any
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Cache mémoire optimisé
_analyse_cache: Dict[str, Any] = {}
_api_cache: Dict[str, Any] = {}

# ── CONFIGURATION DEEPSEEK OPTIMISÉE ────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1/chat/completions"

# Configuration des modèles
MODEL_CHAT = "deepseek-chat"  # Pour la majorité des corrections
MODEL_REASONER = "deepseek-reasoner"  # Uniquement pour les problèmes complexes
MODEL_VISION = "deepseek-chat"  # deepseek-chat gère la vision

# Session HTTP avec retry strategy pour plus de robustesse
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))


def cached_analyser_document_scientifique(fichier_path: str) -> Dict[str, Any]:
    """
    Version avec cache optimisée de l'analyse scientifique
    """
    import hashlib

    with open(fichier_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    cache_key = f"{file_hash}_{os.path.getsize(fichier_path)}"

    if cache_key in _analyse_cache:
        print("✅ Utilisation du cache pour l'analyse scientifique")
        return _analyse_cache[cache_key]

    print("🔍 Analyse nouvelle (non cachée)")
    resultat = analyser_document_scientifique_optimisee(fichier_path)
    _analyse_cache[cache_key] = resultat

    # Gestion mémoire du cache
    if len(_analyse_cache) > 50:
        oldest_key = next(iter(_analyse_cache))
        del _analyse_cache[oldest_key]

    return resultat


def call_deepseek_api_optimise(messages: list, model: str = MODEL_CHAT, temperature: float = 0.1,
                               max_tokens: int = 4000) -> str:
    """
    Appel API DeepSeek optimisé avec gestion d'erreurs avancée et cache
    """
    cache_key = f"{model}_{hash(str(messages))}_{temperature}"

    if cache_key in _api_cache:
        print("✅ Utilisation du cache API")
        return _api_cache[cache_key]

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.05,
        "stream": False
    }

    try:
        print(f"📡 Appel API DeepSeek avec modèle {model}...")
        start_time = time.time()

        response = session.post(
            DEEPSEEK_API_BASE,
            headers=headers,
            json=data,
            timeout=60
        )

        if response.status_code != 200:
            error_msg = f"Erreur API ({response.status_code}): {response.text}"
            print(f"❌ {error_msg}")

            # Fallback vers un autre modèle si erreur
            if model == MODEL_REASONER:
                print("🔄 Fallback vers deepseek-chat...")
                return call_deepseek_api_optimise(messages, MODEL_CHAT, temperature, max_tokens)
            raise Exception(error_msg)

        response_data = response.json()
        content = response_data['choices'][0]['message']['content']

        end_time = time.time()
        print(f"✅ Réponse API reçue en {end_time - start_time:.2f}s - {len(content)} caractères")

        # Mise en cache
        _api_cache[cache_key] = content
        if len(_api_cache) > 100:  # Limite du cache API
            _api_cache.pop(next(iter(_api_cache)))

        return content

    except requests.exceptions.Timeout:
        print("❌ Timeout API - Réessai avec timeout réduit...")
        # Réessai avec timeout réduit
        return call_deepseek_api_optimise(messages, model, temperature, max_tokens)
    except Exception as e:
        print(f"❌ Erreur API: {e}")
        raise


def analyser_document_scientifique_optimisee(fichier_path: str) -> dict:
    """
    Analyse scientifique OPTIMISÉE avec OCR avancé et prompt engineering
    """
    print("🔍 Analyse scientifique optimisée...")

    # 1. OCR AVANCÉ avec prétraitement intelligent
    texte_ocr, metadonnees_ocr = extraire_texte_ocr_avance(fichier_path)

    # 2. ANALYSE CONTEXTUELLE OPTIMISÉE
    prompt_analyse = construire_prompt_analyse_scientifique(texte_ocr, metadonnees_ocr)

    try:
        response = call_deepseek_api_optimise(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_ANALYSE_SCIENTIFIQUE
                },
                {
                    "role": "user",
                    "content": prompt_analyse
                }
            ],
            model=MODEL_CHAT,
            temperature=0.1,
            max_tokens=3000
        )

        resultat = json.loads(response)

        # Validation et enrichissement du résultat
        if not resultat.get("texte_complet") and texte_ocr:
            resultat["texte_complet"] = texte_ocr

        resultat["metadonnees"] = metadonnees_ocr
        resultat["metadonnees"]["qualite_analyse"] = "optimisee"

        print(f"✅ Analyse terminée: {len(resultat.get('texte_complet', ''))} caractères")
        return resultat

    except Exception as e:
        print(f"❌ Erreur analyse optimisée: {e}")
        return {
            "texte_complet": texte_ocr,
            "elements_visuels": [],
            "formules_latex": [],
            "structure_exercices": [],
            "donnees_numeriques": {},
            "metadonnees": metadonnees_ocr
        }


def extraire_texte_ocr_avance(fichier_path: str) -> tuple:
    """
    Extraction OCR avancée avec prétraitement intelligent
    """
    texte_ocr = ""
    metadonnees = {
        "caracteres_speciaux_detectes": [],
        "qualite_ocr": "standard",
        "type_document": "inconnu"
    }

    try:
        if fichier_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(fichier_path)

            # Détection automatique du type de document
            metadonnees["type_document"] = detecter_type_document(image)

            # Configuration OCR adaptative
            config_ocr = get_config_ocr_adaptatif(metadonnees["type_document"])

            # Prétraitement d'image adaptatif
            image_optimisee = preprocess_image_adaptatif(image, metadonnees["type_document"])

            # OCR principal
            texte_ocr = pytesseract.image_to_string(image_optimisee, config=config_ocr)

            # OCR de secours avec configuration différente
            if len(texte_ocr.strip()) < 50:
                config_secours = r'--oem 3 --psm 11 -l fra+eng+equ'
                texte_secours = pytesseract.image_to_string(image, config=config_secours)
                if len(texte_secours) > len(texte_ocr):
                    texte_ocr = texte_secours
                    metadonnees["qualite_ocr"] = "secours"

            # Analyse des caractères spéciaux
            metadonnees["caracteres_speciaux_detectes"] = analyser_caracteres_speciaux(texte_ocr)

        elif fichier_path.lower().endswith('.pdf'):
            texte_ocr = extraire_texte_pdf_optimise(fichier_path)
            metadonnees["type_document"] = "pdf"
            metadonnees["qualite_ocr"] = "pdf_direct"

    except Exception as e:
        print(f"❌ Extraction OCR avancée échouée: {e}")
        texte_ocr = ""

    return texte_ocr, metadonnees


def detecter_type_document(image: Image.Image) -> str:
    """
    Détection automatique du type de document
    """
    try:
        # Analyse rapide de l'image
        largeur, hauteur = image.size
        ratio = largeur / hauteur

        # Conversion en niveaux de gris pour analyse
        gris = image.convert('L')
        tableau = np.array(gris)

        # Détection de densité de texte
        densite_texte = np.mean(tableau < 128)

        if densite_texte > 0.3 and ratio > 1.2:
            return "document_texte"
        elif densite_texte < 0.1:
            return "schema_diagramme"
        else:
            return "mixte"
    except:
        return "inconnu"


def get_config_ocr_adaptatif(type_document: str) -> str:
    """
    Configuration OCR adaptative selon le type de document
    """
    configs = {
        "document_texte": r'--oem 3 --psm 6 -l fra+eng',
        "schema_diagramme": r'--oem 3 --psm 11 -l fra+eng+equ',
        "mixte": r'--oem 3 --psm 6 -l fra+eng+equ',
        "inconnu": r'--oem 3 --psm 6 -l fra+eng+equ'
    }
    return configs.get(type_document, configs["inconnu"])


def preprocess_image_adaptatif(image: Image.Image, type_document: str) -> Image.Image:
    """
    Prétraitement d'image adaptatif selon le type de document
    """
    try:
        if type_document == "schema_diagramme":
            # Renforcement des contours pour les schémas
            image = image.filter(ImageFilter.SHARPEN)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
        else:
            # Amélioration standard pour le texte
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)

        return image
    except Exception as e:
        print(f"⚠️ Prétraitement adaptatif échoué: {e}")
        return image


def analyser_caracteres_speciaux(texte: str) -> list:
    """
    Analyse avancée des caractères spéciaux scientifiques
    """
    caracteres = []

    # Symboles grecs
    symboles_grecs = re.findall(r'[αβγδεζηθικλμνξπρσςτυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ]', texte)
    caracteres.extend(symboles_grecs)

    # Opérateurs mathématiques
    operateurs = re.findall(r'[∑∫∏√∞∠∆∇∂±×÷]', texte)
    caracteres.extend(operateurs)

    # Indices et exposants
    indices = re.findall(r'[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]', texte)
    caracteres.extend(indices)

    # Notation scientifique
    notation_sci = re.findall(r'[×]?10[¹²³⁴⁵⁶⁷⁸⁹]', texte)
    caracteres.extend(notation_sci)

    return list(set(caracteres))


def extraire_texte_pdf_optimise(fichier_path: str) -> str:
    """
    Extraction PDF optimisée avec fallback
    """
    try:
        # Essai 1: Extraction directe
        texte = extract_text(fichier_path)
        if len(texte.strip()) > 100:
            return texte.strip()

        # Essai 2: Conversion image + OCR
        images = convert_from_path(fichier_path, dpi=200)
        textes_images = []

        for i, image in enumerate(images):
            if i >= 3:  # Limiter aux 3 premières pages
                break
            texte_page = pytesseract.image_to_string(image, config=r'--oem 3 --psm 6 -l fra+eng')
            textes_images.append(texte_page)

        return "\n".join(textes_images).strip()

    except Exception as e:
        print(f"❌ Extraction PDF optimisée échouée: {e}")
        return ""


# ── PROMPTS OPTIMISÉS ──────────────────────────────────

SYSTEM_PROMPT_ANALYSE_SCIENTIFIQUE = """
Tu es un expert en analyse de documents scientifiques éducatifs. 
Ton rôle est d'analyser et structurer les documents avec une précision extrême.

TÂCHES PRINCIPALES :
1. CORRECTION OCR : Corrige les erreurs d'OCR, particulièrement les symboles scientifiques
2. IDENTIFICATION : Détermine la matière exacte (maths, physique, chimie, biologie, etc.)
3. EXTRACTION : Identifie toutes les données numériques, unités et formules
4. STRUCTURATION : Détecte la structure des exercices (parties, questions)
5. VISUEL : Analyse les éléments graphiques décrits dans le texte

FORMAT DE RÉPONSE STRICT (JSON) :
{
    "texte_complet": "texte corrigé et structuré",
    "matiere_principale": "maths/physique/chimie/biologie/etc",
    "elements_visuels": [
        {
            "type": "circuit|graphique|schema|diagramme|formule",
            "description": "description précise",
            "donnees_extraites": {"variable1": "valeur1", ...},
            "contexte_scientifique": "explication du concept"
        }
    ],
    "formules_latex": ["formule1", "formule2", ...],
    "structure_exercices": ["Exercice 1", "Question 1.1", ...],
    "donnees_numeriques": {
        "valeurs": [val1, val2, ...],
        "unites": ["unite1", "unite2", ...],
        "variables": ["var1", "var2", ...]
    }
}

EXIGENCES :
- Sois extrêmement précis pour les notations scientifiques
- Conserve toutes les unités de mesure
- Identifie les schémas même s'ils sont décrits textuellement
"""

DEFAULT_SYSTEM_PROMPT_CORRECTION = """
Tu es un professeur expert en correction d'exercices scolaires.
Tu corriges avec bienveillance, précision et pédagogie.

🎯 **OBJECTIFS** :
1. Identifier les points corrects de l'élève
2. Expliquer clairement les erreurs
3. Proposer des méthodes de correction
4. Donner des conseils pour progresser

📐 **POUR LES SCIENCES** :
- Sois ultra-rigoureux dans les calculs
- Vérifie toutes les unités
- Explique chaque étape de raisonnement
- Utilise la notation LaTeX pour les formules

📚 **POUR LES LITTÉRAIRES** :
- Analyse la structure et le style
- Corrige l'orthographe et la grammaire
- Propose des améliorations stylistiques
- Contextualise les références

📊 **POUR LES GRAPHIQUES** :
Quand un graphique est demandé, utilise le format :
---corrigé---
{"graphique": {"type": "fonction", "expression": "x**2", "x_min": -5, "x_max": 5, "titre": "Courbe"}}

📝 **FORMAT DE RÉPONSE** :
- Structure claire avec titres
- Explications détaillées mais concises
- Corrections bienveillantes
- Conseils pratiques
"""


def construire_prompt_analyse_scientifique(texte_ocr: str, metadonnees: dict) -> str:
    """
    Construit un prompt d'analyse scientifique optimisé
    """
    info_speciaux = ""
    if metadonnees.get("caracteres_speciaux_detectes"):
        speciaux = metadonnees["caracteres_speciaux_detectes"]
        info_speciaux = f"""
INFORMATIONS OCR DÉTECTÉES :
- Type de document: {metadonnees.get('type_document', 'inconnu')}
- Caractères scientifiques: {', '.join(speciaux)}
- Qualité OCR: {metadonnees.get('qualite_ocr', 'standard')}
"""

    return f"""
ANALYSE CE DOCUMENT SCIENTIFIQUE :

TEXTE EXTRAIT PAR OCR :
{texte_ocr}

{info_speciaux}

CONSIGNES SPÉCIFIQUES :
1. Corrige les erreurs OCR en priorité (symboles grecs, notations)
2. Identifie la matière principale avec certitude
3. Extrait TOUTES les données numériques avec leurs unités
4. Détecte les formules même incomplètes
5. Structure l'exercice en parties logiques

ATTENTION PARTICULIÈRE :
- Les notations scientifiques doivent être parfaitement restituées
- Les unités doivent être conservées et vérifiées
- Les schémas décrits doivent être analysés

Réponds UNIQUEMENT en JSON valide.
"""


def extraire_graphiques_corrige(corrige_brut):
    pass


# PAR :
def extraire_graphiques_corrige(corrige_brut: str) -> list:
    """
    Extrait les blocs graphiques du corrigé brut
    """
    print("🔍 Extraction des graphiques du corrigé...")
    graphiques = []

    try:
        # Recherche des blocs JSON dans le corrigé
        json_blocks = extract_json_blocks(corrige_brut)
        print(f"📊 {len(json_blocks)} bloc(s) JSON détecté(s)")

        for graph_dict, start, end in json_blocks:
            if isinstance(graph_dict, dict) and 'graphique' in graph_dict:
                graphiques.append(graph_dict['graphique'])
            elif isinstance(graph_dict, dict):
                graphiques.append(graph_dict)

    except Exception as e:
        print(f"❌ Erreur extraction graphiques: {e}")

    return graphiques


def generer_corrige_par_exercice_optimise(texte_exercice: str, contexte: str, matiere=None, donnees_vision=None, demande=None) -> tuple:
    """
    Génération de corrigé OPTIMISÉE avec gestion intelligente du modèle
    """
    print("🎯 Génération de corrigé optimisée...")

    # Choix intelligent du modèle
    model_choice = choisir_modele_optimal(texte_exercice, matiere, donnees_vision)
    print(f"🤖 Modèle sélectionné: {model_choice}")

    # Construction du prompt optimisé
    prompt_correction = construire_prompt_correction_optimise(
        texte_exercice, contexte, matiere, donnees_vision
    )

    try:
        # Appel API optimisé
        corrige_brut = call_deepseek_api_optimise(
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_CORRECTION},
                {"role": "user", "content": prompt_correction}
            ],
            model=model_choice,
            temperature=0.1,
            max_tokens=4000
        )

        # Vérification de qualité
        if not verifier_qualite_corrige_optimise(corrige_brut, texte_exercice):
            print("🔄 Qualité insuffisante, régénération...")
            corrige_brut = call_deepseek_api_optimise(
                messages=[
                    {"role": "system",
                     "content": DEFAULT_SYSTEM_PROMPT_CORRECTION + "\n⚠️ SOIS PLUS PRÉCIS ET DÉTAILLÉ !"},
                    {"role": "user", "content": prompt_correction}
                ],
                model=model_choice,
                temperature=0.1,
                max_tokens=5000
            )

        # Post-traitement
        corrige_traite = post_traiter_corrige(corrige_brut)
        graphiques = extraire_graphiques_corrige(corrige_brut)

        return corrige_traite, graphiques

    except Exception as e:
        print(f"❌ Erreur génération corrigé: {e}")
        return f"Erreur lors de la génération du corrigé: {str(e)}", []


def choisir_modele_optimal(texte_exercice: str, matiere, donnees_vision: dict) -> str:
    """
    Choisit le modèle optimal selon le contexte
    """
    # Si matière scientifique avec éléments complexes → reasoner
    if matiere and hasattr(matiere, 'nom'):
        nom_matiere = matiere.nom.lower()
        if any(mot in nom_matiere for mot in ['math', 'physique', 'chimie']):
            # Vérifier la complexité
            if est_exercice_complexe(texte_exercice, donnees_vision):
                return MODEL_REASONER

    # Par défaut → deepseek-chat (meilleur équilibre)
    return MODEL_CHAT


def est_exercice_complexe(texte_exercice: str, donnees_vision: dict) -> bool:
    """
    Détermine si l'exercice est complexe (nécessite deepseek-reasoner)
    """
    indicateurs_complexite = [
        # Mots-clés de complexité
        'démontrer', 'prouver', 'calculer', 'résoudre', 'déterminer',
        'équation', 'intégrale', 'dérivée', 'théorème', 'formule',
        # Éléments visuels complexes
        'circuit', 'schéma', 'diagramme', 'graphique', 'figure'
    ]

    texte_lower = texte_exercice.lower()

    # Vérifier les mots-clés
    mots_complexes = sum(1 for mot in indicateurs_complexite if mot in texte_lower)

    # Vérifier les données vision
    elements_complexes = donnees_vision and len(donnees_vision.get('elements_visuels', [])) > 0

    return mots_complexes >= 2 or elements_complexes


def construire_prompt_correction_optimise(texte_exercice: str, contexte: str, matiere, donnees_vision: dict) -> str:
    """
    Construit un prompt de correction optimisé
    """
    # En-tête contextuel
    entete = f"""
CONTEXTE : {contexte}
MATIÈRE : {getattr(matiere, 'nom', 'Non spécifiée')}
"""

    # Section vision si disponible
    section_vision = ""
    if donnees_vision:
        section_vision = "\n## 🔬 ÉLÉMENTS VISUELS DÉTECTÉS :\n"

        # Éléments visuels
        elements = donnees_vision.get('elements_visuels', [])
        for i, element in enumerate(elements, 1):
            section_vision += f"\n**Élément {i} - {element.get('type', 'Type inconnu')}:**\n"
            section_vision += f"- Description: {element.get('description', '')}\n"

            donnees_extr = element.get('donnees_extraites', {})
            if donnees_extr:
                section_vision += "- Données extraites:\n"
                for key, value in donnees_extr.items():
                    section_vision += f"  • {key}: {value}\n"

        # Formules LaTeX
        formules = donnees_vision.get('formules_latex', [])
        if formules:
            section_vision += "\n## 📐 FORMULES IDENTIFIÉES :\n"
            for formule in formules:
                section_vision += f"- {formule}\n"

    # Instructions adaptatives
    instructions = get_instructions_adaptatives(matiere)

    return f"""
{entete}

### 📝 EXERCICE À CORRIGER :
{texte_exercice.strip()}

{section_vision}

### 🎯 CONSIGNES DE CORRECTION :
{instructions}

### ✨ EXIGENCES ABSOLUES :
- Sois EXTRÊMEMENT PRÉCIS dans tes explications
- Vérifie systématiquement tous les calculs
- Donne TOUTES les étapes de raisonnement
- Sois BIENVEILLANT et PÉDAGOGIQUE
- Exploite les éléments visuels détectés

Réponds avec une structure claire et aérée.
"""


def get_instructions_adaptatives(matiere) -> str:
    """
    Retourne des instructions adaptées à la matière
    """
    if not matiere or not hasattr(matiere, 'nom'):
        return "Corrige avec précision et pédagogie."

    nom_matiere = matiere.nom.lower()

    if any(mot in nom_matiere for mot in ['math', 'physique', 'chimie']):
        return """
• Vérifie toutes les unités et conversions
• Donne les calculs intermédiaires détaillés
• Utilise la notation LaTeX pour les formules
• Explique le raisonnement étape par étape
"""
    elif any(mot in nom_matiere for mot in ['français', 'lettre', 'littérature']):
        return """
• Analyse la structure et le style
• Corrige l'orthographe et la grammaire
• Propose des améliorations stylistiques
• Contextualise les références culturelles
"""
    elif any(mot in nom_matiere for mot in ['histoire', 'géographie']):
        return """
• Vérifie la précision des dates et faits
• Contextualise les événements
• Structure la réponse de manière logique
• Cite les sources implicites
"""
    else:
        return "Corrige avec précision, structure clairement et sois pédagogique."


def verifier_qualite_corrige_optimise(corrige: str, exercice_original: str) -> bool:
    """
    Vérification avancée de la qualité du corrigé
    """
    if not corrige or len(corrige.strip()) < 50:
        return False

    # Indicateurs de mauvaise qualité
    indicateurs_problemes = [
        "je ne peux pas", "impossible de", "manque d'information",
        "énoncé incomplet", "donnée manquante", "je ne sais pas",
        "ambigu", "imprécis", "incertain"
    ]

    # Compter les problèmes
    problemes = sum(1 for indicateur in indicateurs_problemes
                    if indicateur.lower() in corrige.lower())

    if problemes >= 2:
        return False

    # Vérifier le ratio longueur corrigé/énoncé
    ratio = len(corrige) / len(exercice_original) if exercice_original else 1
    if ratio < 0.3:  # Corrigé trop court
        return False

    return True


def post_traiter_corrige(corrige_brut: str) -> str:
    """
    Post-traitement intelligent du corrigé
    """
    # Nettoyage de base
    corrige = re.sub(r'#+\s*', '', corrige_brut)  # Remove markdown headers
    corrige = re.sub(r'\*{2,}', '', corrige)  # Remove excessive asterisks
    corrige = re.sub(r'\n{3,}', '\n\n', corrige)  # Normalize line breaks

    # Fusion des blocs LaTeX
    corrige = flatten_multiline_latex_blocks(corrige)

    # Formatage structurel
    corrige = format_corrige_pdf_structure(corrige)

    return corrige.strip()


# ── FONCTIONS EXISTANTES CONSERVÉES MAIS OPTIMISÉES ─────────────────

def extraire_texte_fichier_optimise(fichier_field):
    """
    EXTRACTION MULTIMODALE OPTIMISÉE
    """
    if not fichier_field:
        return ""

    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))

    with open(local_path, "wb") as f:
        for chunk in fichier_field.chunks():
            f.write(chunk)

    try:
        # Extraction robuste avec cache
        texte_principal = extraire_texte_robuste(local_path)

        # Analyse scientifique avec cache
        analyse_complete = cached_analyser_document_scientifique(local_path)

        # Construction du texte enrichi optimisé
        texte_enrichi = construire_texte_enrichi(texte_principal, analyse_complete)

        return texte_enrichi.strip()

    except Exception as e:
        print(f"❌ Erreur extraction optimisée: {e}")
        return ""
    finally:
        try:
            os.unlink(local_path)
        except:
            pass


def construire_texte_enrichi(texte_principal: str, analyse_complete: dict) -> str:
    """
    Construit le texte enrichi de manière optimisée
    """
    sections = []

    # Texte principal
    if texte_principal:
        sections.append("## 📝 TEXTE DU DOCUMENT")
        sections.append(texte_principal)

    # Éléments visuels
    elements_visuels = analyse_complete.get("elements_visuels", [])
    if elements_visuels:
        sections.append("\n## 🔬 ÉLÉMENTS VISUELS IDENTIFIÉS")
        for i, element in enumerate(elements_visuels, 1):
            sections.append(f"\n### Schéma {i}: {element.get('type', 'Non spécifié')}")
            sections.append(f"**Description:** {element.get('description', '')}")

            donnees = element.get('donnees_extraites', {})
            if donnees:
                sections.append("**Données extraites:**")
                for key, value in donnees.items():
                    sections.append(f"  - {key}: {value}")

    # Formules LaTeX
    formules = analyse_complete.get("formules_latex", [])
    if formules:
        sections.append("\n## 📐 FORMULES MATHÉMATIQUES")
        for formule in formules:
            sections.append(f"- {formule}")

    return "\n".join(sections)


# ── FONCTIONS EXISTANTES À CONSERVER ───────────────────
# (Ces fonctions restent identiques mais sont appelées par les nouvelles fonctions optimisées)

def extraire_texte_robuste(fichier_path: str) -> str:
    """Version optimisée de l'extraction simple"""
    print("🔄 Extraction robuste avec cache...")
    try:
        analyse = cached_analyser_document_scientifique(fichier_path)
        texte = analyse.get("texte_complet", "")
        return texte if texte and len(texte) > 50 else ""
    except Exception as e:
        print(f"❌ Extraction robuste échouée: {e}")
        return ""


def debug_ocr(fichier_path: str):
    """Debug OCR (identique)"""
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
    r'GRAMMAR', r'VOCABULARY', r'COMPREHENSION', r'ESSAY',
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

# ============== BLIP IMAGE CAPTIONING ==============
# On détecte si CUDA est dispo, sinon on reste sur CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖼️ BLIP device utilisé : {device}")

# Charger le processor et le modèle BLIP (tailles modestes pour la rapidité)
_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")\
                 .to(device).eval()
print("🖼️ Modèle BLIP chargé avec succès")

# ============== FONCTIONS DE DÉCOUPAGE INTELLIGENT ==============

def separer_exercices(texte_epreuve):
    """
    Détecte et sépare automatiquement les exercices d'une épreuve
    """
    if not texte_epreuve:
        return []

    print("🔍 Détection des exercices...")

    # Patterns pour détecter le début des exercices
    patterns_separation = [
        r'Exercice\s+\d+[:.]', r'EXERCICE\s+\d+[:.]',
        r'Partie\s+[IVXLCDM]+[:.]',
        r'\n\d+[-.)]\s', r'\n[a-z]\)\s',
        r'Question\s+\d+',
        # Nouveaux genres d’épreuves (langues, lettres, geo, etc.)
        r'COMENTARIO DEL TEXTO', r'ESTRUCTURAS DE COMUNICACIÓN',
        r'PRODUCCIÓN DE TEXTOS', r'RECEPCIÓN DE TEXTOS',
        r'EXPRESIÓN ESCRITA', r'TRADUCCIÓN',
        r'TEIL ?1\s+LESEVERSTEHEN', r'MEDIATION',
        r'SCHRIFTLICHE PRODUKTION', r'STRUKTUREN UND KOMMUNIKATION',
        r'SCHRIFTLICHER AUSDRUCK', r'Grammar', r'Vocabulary',
        r'Comprehension', r'Essay'
    ]

    exercices = []
    lignes = texte_epreuve.split('\n')
    exercice_courant = []
    dans_exercice = False

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue

        # Vérifier si cette ligne commence un nouvel exercice
        nouvel_exercice = False
        for pattern in patterns_separation:
            if re.search(pattern, ligne, re.IGNORECASE):
                nouvel_exercice = True
                break

        if nouvel_exercice and exercice_courant:
            # Sauvegarder l'exercice précédent
            exercices.append('\n'.join(exercice_courant))
            exercice_courant = []
            dans_exercice = True

        exercice_courant.append(ligne)

    # Ajouter le dernier exercice
    if exercice_courant:
        exercices.append('\n'.join(exercice_courant))

    # Si aucun exercice détecté, traiter tout comme un seul exercice
    if not exercices:
        exercices = [texte_epreuve]

    print(f"✅ {len(exercices)} exercice(s) détecté(s)")
    for i, ex in enumerate(exercices):
        print(f"   Exercice {i + 1}: {len(ex)} caractères")

    return exercices


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


def generer_corrige_par_exercice(texte_exercice, contexte, matiere=None, donnees_vision=None):
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

    system_prompt = DEFAULT_SYSTEM_PROMPT
    consignes_finales = "Format de réponse strict : LaTeX pour les exercices scientifiques, explications détaillées mais concises"

    if matiere and hasattr(matiere, 'prompt_ia'):
        promptia = matiere.prompt_ia
        system_prompt = promptia.system_prompt or system_prompt
        consignes_finales = promptia.consignes_finales or consignes_finales

    # ✅ NOUVEAU : Construction du prompt enrichi avec données vision
    prompt_vision = ""
    if donnees_vision and donnees_vision.get('elements_visuels'):
        prompt_vision = "\n\n## 🔬 SCHÉMAS IDENTIFIÉS DANS L'EXERCICE :\n"
        for i, element in enumerate(donnees_vision['elements_visuels'], 1):
            prompt_vision += f"\n**Schéma {i} - {element.get('type', 'Type inconnu')}:**\n"
            prompt_vision += f"- Description: {element.get('description', '')}\n"

            donnees_extr = element.get('donnees_extraites', {})
            if donnees_extr:
                prompt_vision += "- Données extraites:\n"
                for key, value in donnees_extr.items():
                    prompt_vision += f"  • {key}: {value}\n"

            contexte_sci = element.get('contexte_scientifique', '')
            if contexte_sci:
                prompt_vision += f"- Contexte: {contexte_sci}\n"

    # ✅ NOUVEAU : Ajout des formules LaTeX détectées
    formules_vision = ""
    if donnees_vision and donnees_vision.get('formules_latex'):
        formules_vision = "\n\n## 📐 FORMULES DÉTECTÉES :\n"
        for formule in donnees_vision['formules_latex']:
            formules_vision += f"- {formule}\n"

    prompt_ia = f"""
    {system_prompt}

    ### CONTEXTE
    {contexte}

    ### EXERCICE À CORRIGER
    {texte_exercice.strip()}

    {prompt_vision}
    {formules_vision}

    ### CONSIGNES STRICTES - À RESPECTER IMPÉRATIVEMENT
    {consignes_finales}

    **EXIGENCES ABSOLUES :**
    1. Sois EXTRÊMEMENT RIGOUREUX dans tous les calculs
    2. Vérifie systématiquement chaque résultat intermédiaire  
    3. Donne TOUTES les étapes de calcul détaillées
    4. Les réponses doivent être NUMÉRIQUEMENT EXACTES
    5. Ne laisse AUCUNE question sans réponse complète
    6. **EXPLOITE LES SCHÉMAS IDENTIFIÉS** dans tes explications

    **POUR LES SCHÉMAS :**
    - Réfère-toi aux données extraites (angles, masses, distances)
    - Utilise les descriptions des schémas dans tes explications
    - Mentionne explicitement "D'après le schéma..." ou "Le schéma montre que..."

    **FORMAT DE RÉPONSE :**
    - Réponses complètes avec justification
    - Calculs intermédiaires détaillés
    - Solutions numériques exactes
    - Références aux schémas quand ils existent
    - Ne jamais dire "je pense" ou "c'est ambigu"

    Réponds UNIQUEMENT à cet exercice avec une rigueur absolue.
    """

    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ Erreur: Clé API non configurée")
        return "Erreur: Clé API non configurée", None

    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_ia}
        ],
        "temperature": 0.1,
        "max_tokens": 6000,
        "top_p": 0.9,
        "frequency_penalty": 0.1
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
def call_deepseek_vision(local_path):
    pass


def extraire_texte_fichier(fichier_field):
    """
    EXTRACTION MULTIMODALE AVEC CACHE OPTIMISÉ
    """
    if not fichier_field:
        return ""

    # 1) Sauvegarde locale temporaire
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))

    with open(local_path, "wb") as f:
        for chunk in fichier_field.chunks():
            f.write(chunk)

    try:
        # ✅ DEBUG OCR DIRECT
        print("🔍 DEBUG - Test OCR direct:")
        texte_ocr_brut = debug_ocr(local_path)

        # 2) EXTRACTION ROBUSTE AVEC CACHE
        print("🔍 Lancement extraction robuste avec cache...")
        texte_principal = extraire_texte_robuste(local_path)

        if not texte_principal:
            print("❌ Aucun texte extrait, utilisation fallback OCR basique")
            try:
                resultat_simple = call_deepseek_vision(local_path)
                texte_principal = resultat_simple.get("text", "")
            except:
                texte_principal = ""

        # 3) ANALYSE SCIENTIFIQUE POUR LES SCHÉMAS AVEC CACHE
        print("🔍 Analyse scientifique des schémas (avec cache)...")
        analyse_complete = cached_analyser_document_scientifique(local_path)  # ← AVEC CACHE

        # 4) CONSTRUCTION DU TEXTE ENRICHI
        texte_enrichi = []

        # Texte principal
        if texte_principal:
            texte_enrichi.append("## 📝 TEXTE DU DOCUMENT")
            texte_enrichi.append(texte_principal)

        # Éléments visuels (schémas, croquis scientifiques)
        elements_visuels = analyse_complete.get("elements_visuels", [])
        if elements_visuels:
            texte_enrichi.append("\n## 🔬 SCHÉMAS SCIENTIFIQUES IDENTIFIÉS")
            for i, element in enumerate(elements_visuels, 1):
                texte_enrichi.append(f"\n### Schéma {i}: {element.get('type', 'Non spécifié')}")
                texte_enrichi.append(f"**Description:** {element.get('description', '')}")

                donnees = element.get('donnees_extraites', {})
                if donnees:
                    texte_enrichi.append("**Données extraites:**")
                    for key, value in donnees.items():
                        texte_enrichi.append(f"  - {key}: {value}")

                contexte = element.get('contexte_scientifique', '')
                if contexte:
                    texte_enrichi.append(f"**Contexte scientifique:** {contexte}")

        # Formules LaTeX
        formules = analyse_complete.get("formules_latex", [])
        if formules:
            texte_enrichi.append("\n## 📐 FORMULES MATHÉMATIQUES")
            for formule in formules:
                texte_enrichi.append(f"- {formule}")

        # Structure des exercices
        structure = analyse_complete.get("structure_exercices", [])
        if structure:
            texte_enrichi.append("\n## 📚 STRUCTURE DES EXERCICES")
            for element in structure:
                texte_enrichi.append(f"- {element}")

        # 5) Retourner le texte enrichi
        texte_final = "\n".join(texte_enrichi)
        print(f"✅ Extraction terminée: {len(texte_final)} caractères")
        return texte_final.strip()

    except Exception as e:
        print(f"❌ Erreur extraction: {e}")
        return ""
    finally:
        # Nettoyage
        try:
            os.unlink(local_path)
        except:
            pass
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
DEFAULT_SYSTEM_PROMPT = r"""Tu es un professeur expert en Mathématiques, physique, chimie, biologie.

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
def generer_corrige_direct(texte_enonce, contexte, lecons_contenus, exemples_corriges, matiere, donnees_vision=None):
    """
    Traitement direct pour les épreuves courtes avec données vision.
    """
    print("🎯 Traitement DIRECT avec analyse vision")

    # ✅ PASSER les données vision à la fonction de génération
    return generer_corrige_par_exercice(texte_enonce, contexte, matiere, donnees_vision)


def generer_corrige_decoupe(texte_epreuve, contexte, matiere, donnees_vision=None):
    """
    Traitement par découpage pour les épreuves longues avec données vision.
    """
    print("🎯 Traitement AVEC DÉCOUPAGE et analyse vision")

    exercices = separer_exercices(texte_epreuve)
    tous_corriges = []
    tous_graphiques = []

    for i, exercice in enumerate(exercices, 1):
        print(f"📝 Traitement exercice {i}/{len(exercices)}...")

        # ✅ PASSER les données vision à chaque exercice
        corrige, graphiques = generer_corrige_par_exercice(exercice, contexte, matiere, donnees_vision)

        if corrige and not corrige.startswith("Erreur") and not corrige.startswith("Erreur API"):
            titre_exercice = f"\n\n## 📝 Exercice {i}\n\n"
            tous_corriges.append(titre_exercice + corrige)
            if graphiques:
                tous_graphiques.extend(graphiques)
            print(f"✅ Exercice {i} traité avec succès")
        else:
            print(f"❌ Exercice {i} en erreur: {corrige}")
        import time
        time.sleep(1)

    if tous_corriges:
        corrige_final = "".join(tous_corriges)
        print(f"🎉 Découpage terminé: {len(tous_corriges)} exercice(s), {len(tous_graphiques)} graphique(s)")
        return corrige_final, tous_graphiques
    else:
        print("❌ Aucun corrigé généré")
        return "Erreur: Aucun corrigé n'a pu être généré", []



def generer_corrige_ia_et_graphique(texte_enonce, contexte, lecons_contenus=None, exemples_corriges=None, matiere=None,
                                    demande=None, donnees_vision=None):  # ✅ NOUVEAU PARAMÈTRE
    """
    Nouvelle version avec support des données vision
    """
    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    print("\n" + "=" * 60)
    print("🚀 DÉBUT TRAITEMENT INTELLIGENT AVEC VISION")
    print("=" * 60)
    print(f"📏 Longueur texte: {len(texte_enonce)} caractères")

    # ✅ NOUVEAU : Log des données vision
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
                                      donnees_vision)
    else:  # Épreuve longue
        print("🎯 Décision: DÉCOUPAGE (épreuve longue)")
        return generer_corrige_decoupe(texte_enonce, contexte, matiere, donnees_vision)


# ============== TÂCHE ASYNCHRONE ==============

def analyser_document_scientifique(fichier_path: str) -> dict:
    """
    Fonction d'analyse scientifique principale (alias vers la version optimisée)
    Conservée pour la compatibilité avec le code existant
    """
    return analyser_document_scientifique_optimisee(fichier_path)


@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(demande_id, matiere_id=None):
    from correction.models import DemandeCorrection, SoumissionIA
    from resources.models import Matiere
    from abonnement.services import debiter_credit_abonnement

    try:
        # Récupération de la demande et création de la soumission IA
        demande = DemandeCorrection.objects.get(id=demande_id)
        soumission = SoumissionIA.objects.get(demande=demande)

        # Étape 1 : Extraction du texte brut AVEC VISION
        soumission.statut = 'extraction'
        soumission.progression = 20
        soumission.save()

        donnees_vision_complete = None  # ✅ NOUVEAU : Stockage des données vision

        if demande.fichier:
            # ✅ EXTRACTION AVEC VISION SCIENTIFIQUE
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, os.path.basename(demande.fichier.name))

            with open(local_path, "wb") as f:
                for chunk in demande.fichier.chunks():
                    f.write(chunk)

            # Analyse scientifique complète
            donnees_vision_complete = analyser_document_scientifique(local_path)
            texte_brut = extraire_texte_fichier(demande.fichier)  # Utilise la nouvelle fonction

            # Nettoyage
            try:
                os.unlink(local_path)
            except:
                pass
        else:
            texte_brut = demande.enonce_texte or ""

        print("📥 DEBUG – TEXTE BRUT AVEC VISION (premiers 500 chars) :")
        print(texte_brut[:500].replace("\n", "\\n"), "...\n")

        # Étape 2 : Texte final pour l'IA
        texte_enonce = texte_brut

        # Étape 3 : Lancement du traitement IA AVEC DONNÉES VISION
        soumission.statut = 'analyse_ia'
        soumission.progression = 40
        soumission.save()

        matiere = Matiere.objects.get(id=matiere_id) if matiere_id else demande.matiere
        contexte = f"Exercice de {matiere.nom} - {demande.classe.nom if demande.classe else ''}"

        soumission.statut = 'generation_graphiques'
        soumission.progression = 60
        soumission.save()

        # ✅ APPEL AVEC DONNÉES VISION
        corrige_txt, graph_list = generer_corrige_ia_et_graphique(
            texte_enonce,
            contexte,
            matiere=matiere,
            donnees_vision=donnees_vision_complete  # ✅ NOUVEAU
        )

        # [Le reste du code reste identique...]
        soumission.statut = 'formatage_pdf'
        soumission.progression = 80
        soumission.save()

        if not debiter_credit_abonnement(demande.user):
            soumission.statut = 'erreur_credit'
            soumission.save()
            return False

        from .pdf_utils import generer_pdf_corrige
        pdf_path = generer_pdf_corrige(
            {
                "titre_corrige": contexte,
                "corrige_html": corrige_txt,
                "soumission_id": demande_id
            },
            demande_id
        )

        # Étape 5 : Mise à jour du statut et sauvegarde
        soumission.statut = 'termine'
        soumission.progression = 100
        soumission.resultat_json = {
            'corrige_text': corrige_txt,
            'pdf_url': pdf_path,
            'graphiques': graph_list or [],
            'analyse_vision': donnees_vision_complete  # ✅ NOUVEAU : Stocker l'analyse
        }
        soumission.save()

        demande.corrigé = corrige_txt
        demande.save()

        print("🎉 TRAITEMENT AVEC VISION TERMINÉ AVEC SUCCÈS!")
        return True

    except Exception as e:
        print(f"❌ ERREUR dans la tâche IA: {e}")
        try:
            soumission.statut = 'erreur'
            soumission.save()
        except:
            pass
        return False


# ── POINT D'ENTRÉE PRINCIPAL OPTIMISÉ ──────────────────

def generer_corrige_ia_et_graphique_optimise(texte_enonce, contexte, lecons_contenus=None, exemples_corriges=None,
                                             matiere=None, demande=None, donnees_vision=None):
    """
    NOUVELLE VERSION OPTIMISÉE du point d'entrée principal
    """
    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    print("\n" + "=" * 60)
    print("🚀 DÉBUT TRAITEMENT INTELLIGENT OPTIMISÉ")
    print("=" * 60)
    print(f"📏 Longueur texte: {len(texte_enonce)} caractères")

    # Log des données vision
    if donnees_vision:
        print(f"🔬 Données vision disponibles:")
        print(f"   - Éléments visuels: {len(donnees_vision.get('elements_visuels', []))}")
        print(f"   - Formules LaTeX: {len(donnees_vision.get('formules_latex', []))}")

    # Estimation de complexité
    tokens_estimes = estimer_tokens(texte_enonce)

    # Décision optimisée
    if tokens_estimes < 1500:
        print("🎯 Décision: TRAITEMENT DIRECT OPTIMISÉ")
        return generer_corrige_direct_optimise(texte_enonce, contexte, lecons_contenus, exemples_corriges,
                                               matiere, donnees_vision)
    else:
        print("🎯 Décision: DÉCOUPAGE OPTIMISÉ")
        return generer_corrige_decoupe_optimise(texte_enonce, contexte, matiere, donnees_vision)


def generer_corrige_direct_optimise(texte_enonce, contexte, lecons_contenus, exemples_corriges, matiere,
                                    donnees_vision=None):
    """Version optimisée du traitement direct"""
    return generer_corrige_par_exercice_optimise(texte_enonce, contexte, matiere, donnees_vision)


def generer_corrige_decoupe_optimise(texte_epreuve, contexte, matiere, donnees_vision=None):
    """Version optimisée du traitement par découpage"""
    exercices = separer_exercices(texte_epreuve)
    tous_corriges = []
    tous_graphiques = []

    for i, exercice in enumerate(exercices, 1):
        print(f"📝 Traitement exercice {i}/{len(exercices)}...")

        corrige, graphiques = generer_corrige_par_exercice_optimise(exercice, contexte, matiere, donnees_vision)

        if corrige and not corrige.startswith("Erreur"):
            titre_exercice = f"\n\n## 📝 Exercice {i}\n\n"
            tous_corriges.append(titre_exercice + corrige)
            if graphiques:
                tous_graphiques.extend(graphiques)
            print(f"✅ Exercice {i} traité avec succès")
        else:
            print(f"❌ Exercice {i} en erreur: {corrige}")

        time.sleep(0.5)  # Réduction du délai

    if tous_corriges:
        corrige_final = "".join(tous_corriges)
        print(f"🎉 Découpage optimisé terminé: {len(tous_corriges)} exercice(s), {len(tous_graphiques)} graphique(s)")
        return corrige_final, tous_graphiques
    else:
        print("❌ Aucun corrigé généré")
        return "Erreur: Aucun corrigé n'a pu être généré", []


# ── FONCTION POUR TESTER LES PERFORMANCES ──────────────

def tester_performances():
    """
    Fonction utilitaire pour tester les performances des différents modèles
    """
    test_prompt = "Résous : 2x + 5 = 13. Montre toutes les étapes."

    print("🧪 TEST DE PERFORMANCES DES MODÈLES")
    print("=" * 50)

    # Test deepseek-chat
    start = time.time()
    try:
        result_chat = call_deepseek_api_optimise(
            [{"role": "user", "content": test_prompt}],
            MODEL_CHAT
        )
        time_chat = time.time() - start
        print(f"✅ deepseek-chat: {time_chat:.2f}s - {len(result_chat)} caractères")
    except Exception as e:
        print(f"❌ deepseek-chat: Erreur - {e}")

    # Test deepseek-reasoner
    start = time.time()
    try:
        result_reasoner = call_deepseek_api_optimise(
            [{"role": "user", "content": test_prompt}],
            MODEL_REASONER
        )
        time_reasoner = time.time() - start
        print(f"✅ deepseek-reasoner: {time_reasoner:.2f}s - {len(result_reasoner)} caractères")
    except Exception as e:
        print(f"❌ deepseek-reasoner: Erreur - {e}")

