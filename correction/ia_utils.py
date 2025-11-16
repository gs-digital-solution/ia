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
import threading

# Cache mémoire pour réduire les appels API
_analyse_cache: Dict[str, Any] = {}

def cached_analyser_document_scientifique(fichier_path: str) -> Dict[str, Any]:
    """
    Version avec cache de l'analyse scientifique
    """
    import hashlib

    # Créer une clé de cache basée sur le contenu du fichier
    with open(fichier_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    cache_key = f"{file_hash}_{os.path.getsize(fichier_path)}"

    # Vérifier le cache
    if cache_key in _analyse_cache:
        print("✅ Utilisation du cache pour l'analyse scientifique")
        return _analyse_cache[cache_key]

    # Sinon, faire l'analyse et mettre en cache
    print("🔍 Analyse nouvelle (non cachée)")
    resultat = analyser_document_scientifique(fichier_path)
    _analyse_cache[cache_key] = resultat

    # Limiter la taille du cache (éviter memory leak)
    if len(_analyse_cache) > 50:  # Garder seulement 50 analyses
        oldest_key = next(iter(_analyse_cache))
        del _analyse_cache[oldest_key]
        print("🧹 Cache nettoyé (limite atteinte)")

    return resultat

# ── CONFIGURATION DEEPSEEK AVEC VISION ────────────────────
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.api_base = "https://api.deepseek.com"

# ── MODÈLE POUR LA VISION ────────────────────────────────
# deepseek-chat a les capacités vision quand on envoie des images
DEEPSEEK_VISION_MODEL = "deepseek-chat"


# ─── NEW ─── appel multimodal à DeepSeek-V3 pour PDF / images ────
# ── CORRIGÉ : Appel multimodal à DeepSeek pour PDF/images ────
def call_deepseek_vision(path_fichier: str) -> dict:
    """
    Envoie un PDF ou une image à DeepSeek - Version corrigée pour l'API DeepSeek.
    """
    system_prompt = """
    Tu es un analyseur de documents éducatifs.
    Analyse ce document et retourne un JSON valide avec :
    - Le texte intégral
    - Les blocs LaTeX des formules mathématiques  
    - Les légendes des images
    - Les descriptions des graphiques
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

def analyser_document_scientifique(fichier_path: str) -> dict:
    """
    Analyse scientifique avec OCR OPTIMISÉ pour les formules mathématiques
    """
    print("🔍 Analyse scientifique avec OCR optimisé...")

    # 1. OCR AMÉLIORÉ avec configuration scientifique
    texte_ocr = ""
    caracteres_speciaux_detectes = []

    try:
        if fichier_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(fichier_path)

            # ✅ CONFIGURATION OCR AMÉLIORÉE POUR LES SCIENCES
            custom_config = r'--oem 3 --psm 6 -l fra+eng+equ'
            # oem 3 = moteur OCR LSTM le plus avancé
            # psm 6 = bloc de texte uniforme (idéal pour les documents structurés)
            # fra+eng+equ = français + anglais + langage équations mathématiques

            texte_ocr = pytesseract.image_to_string(image, config=custom_config)
            print(f"✅ OCR scientifique extrait: {len(texte_ocr)} caractères")

            # ✅ DÉTECTION DES CARACTÈRES SPÉCIAUX SCIENTIFIQUES
            # Symboles grecs
            symboles_grecs = re.findall(r'[αβγδεζηθικλμνξπρσςτυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ]', texte_ocr)
            # Opérateurs mathématiques
            operateurs_math = re.findall(r'[∑∫∏√∞∠∆∇∂]', texte_ocr)
            # Indices et exposants
            indices_exposants = re.findall(r'[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]', texte_ocr)

            if symboles_grecs:
                caracteres_speciaux_detectes.extend(symboles_grecs)
                print(f"   Symboles grecs détectés: {set(symboles_grecs)}")
            if operateurs_math:
                caracteres_speciaux_detectes.extend(operateurs_math)
                print(f"   Opérateurs mathématiques: {set(operateurs_math)}")
            if indices_exposants:
                caracteres_speciaux_detectes.extend(indices_exposants)
                print(f"   Indices/exposants: {set(indices_exposants)}")

            # ✅ PRÉTRAITEMENT D'IMAGE POUR AMÉLIORER L'OCR
            try:
                # Conversion en niveaux de gris
                image_gris = image.convert('L')
                # Amélioration du contraste
                enhancer = ImageEnhance.Contrast(image_gris)
                image_contrast = enhancer.enhance(2.0)  # Contraste x2
                # Amélioration de la netteté
                enhancer = ImageEnhance.Sharpness(image_contrast)
                image_sharp = enhancer.enhance(2.0)  # Netteté x2

                # OCR sur l'image prétraitée
                texte_ocr_ameliore = pytesseract.image_to_string(image_sharp, config=custom_config)
                if len(texte_ocr_ameliore) > len(texte_ocr):
                    texte_ocr = texte_ocr_ameliore
                    print("✅ OCR amélioré par prétraitement d'image")

            except Exception as e_pretraitement:
                print(f"⚠️ Prétraitement image échoué: {e_pretraitement}")

        elif fichier_path.lower().endswith('.pdf'):
            texte_ocr = extraire_texte_pdf(fichier_path)
            print(f"✅ PDF extrait: {len(texte_ocr)} caractères")

    except Exception as e:
        print(f"❌ Extraction échouée: {e}")
        texte_ocr = ""

    # 2. ANALYSE CONTEXTUELLE AVEC INFORMATION OCR AMÉLIORÉE
    try:
        # ✅ CONSTRUCTION DU PROMPT ENRICHIE AVEC INFO OCR
        info_ocr_ameliore = ""
        if caracteres_speciaux_detectes:
            info_ocr_ameliore = f"""
            INFORMATIONS OCR DÉTECTÉES :
            - Caractères spéciaux scientifiques: {', '.join(set(caracteres_speciaux_detectes))}
            """

        prompt = f"""
        ANALYSE CE DOCUMENT SCIENTIFIQUE :

        TEXTE EXTRAIT PAR OCR :
        {texte_ocr}

        {info_ocr_ameliore}

        TÂCHES IMPORTANTES :
        1. CORRIGE les erreurs d'OCR en priorité (symboles grecs, notations scientifiques)
        2. IDENTIFIE précisément le type d'exercice (physique, maths, chimie, etc.)
        3. EXTRAIT toutes les données numériques avec leurs unités
        4. DÉTECTE les formules mathématiques et notations scientifiques
        5. STRUCTURE l'exercice (parties, questions)

        ATTENTION PARTICULIÈRE :
        - Les symboles grecs doivent être correctement interprétés
        - Les notations scientifiques (exposants, indices) doivent être préservées
        - Les unités de mesure doivent être exactes

        RÉPONDS en JSON :
        {{
            "texte_complet": "texte corrigé et complété avec notations scientifiques exactes",
            "elements_visuels": [
                {{
                    "type": "circuit|pendule|graphique|plan_incline|etc",
                    "description": "description détaillée basée sur le contexte scientifique",
                    "donnees_extraites": {{}},
                    "contexte_scientifique": "explication du concept physique/mathématique"
                }}
            ],
            "formules_latex": ["liste des formules mathématiques détectées"],
            "structure_exercices": ["Exercice X", "Question Y", "Partie Z"],
            "donnees_numeriques": {{
                "valeurs": [liste des valeurs numériques],
                "unites": [liste des unités correspondantes]
            }}
        }}
        """

        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "Tu es un expert en sciences avec une expertise particulière en reconnaissance de notations mathématiques et scientifiques."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=3500  # Légèrement augmenté pour les notations complexes
        )

        resultat = json.loads(response.choices[0].message.content)

        # ✅ VALIDATION ET AMÉLIORATION DU RÉSULTAT
        # S'assurer qu'on a au moins le texte OCR de base
        if not resultat.get("texte_complet") and texte_ocr:
            resultat["texte_complet"] = texte_ocr

        # Ajouter les informations OCR détectées aux métadonnées
        if caracteres_speciaux_detectes and "metadonnees" not in resultat:
            resultat["metadonnees"] = {
                "caracteres_speciaux_detectes": list(set(caracteres_speciaux_detectes)),
                "qualite_ocr": "amelioree" if caracteres_speciaux_detectes else "standard"
            }

        print(f"✅ Analyse terminée: {len(resultat.get('texte_complet', ''))} caractères")
        if caracteres_speciaux_detectes:
            print(f"   Symboles scientifiques traités: {len(set(caracteres_speciaux_detectes))}")

        return resultat

    except Exception as e:
        print(f"❌ Erreur analyse: {e}")
        return {
            "texte_complet": texte_ocr,
            "elements_visuels": [],
            "formules_latex": [],
            "structure_exercices": [],
            "donnees_numeriques": {},
            "metadonnees": {
                "caracteres_speciaux_detectes": list(set(caracteres_speciaux_detectes)),
                "qualite_ocr": "erreur"
            }
        }


def extraire_texte_robuste(fichier_path: str) -> str:
    """
    Extraction simple : OCR direct → Analyse IA AVEC CACHE
    """
    print("🔄 Extraction simple avec cache...")

    # Utiliser l'analyse scientifique AVEC CACHE
    try:
        analyse = cached_analyser_document_scientifique(fichier_path)  # ← AVEC CACHE
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


# ============== BLIP IMAGE CAPTIONING - VERSION SÉCURISÉE ==============
_blip_components = None
_blip_lock = threading.Lock()


def get_blip_components():
    """
    Charge BLIP une seule fois par processus - Évite les conflits entre workers
    """
    global _blip_components

    if _blip_components is None:
        with _blip_lock:
            # Double vérification thread-safe
            if _blip_components is None:
                print(f"🖼️ [BLIP INIT] Initialisation dans le processus {os.getpid()}")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
                    device).eval()
                _blip_components = (processor, model, device)
                print(f"✅ [BLIP INIT] Chargé avec succès dans PID {os.getpid()}")

    return _blip_components


# Fonction utilitaire pour tests (optionnelle)
def test_blip_functionality():
    """
    Teste que BLIP fonctionne dans le processus courant
    """
    try:
        processor, model, device = get_blip_components()
        print(f"✅ [BLIP TEST] Fonctionnel dans PID {os.getpid()}")
        return True
    except Exception as e:
        print(f"❌ [BLIP TEST] Erreur dans PID {os.getpid()}: {e}")
        return False

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
        "model": "deepseek-chat",
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

@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(demande_id, matiere_id=None):
    # ========== DIAGNOSTIC BLIP ==========
    print(f"\n" + "=" * 60)
    print(f"🔍 [TÂCHE DÉBUT] PID: {os.getpid()}, Demande: {demande_id}")

    # Test BLIP
    try:
        processor, model, device = get_blip_components()
        print(f"✅ [BLIP STATUS] Fonctionnel dans PID {os.getpid()}")
        print(f"🖼️ [BLIP INFO] Device: {device}, Processor: {processor is not None}, Model: {model is not None}")
    except Exception as e:
        print(f"❌ [BLIP STATUS] ERREUR dans PID {os.getpid()}: {e}")

    print(f"🔍 [TÂCHE SUITE] Exécution normale...")
    print("=" * 60)
    # ========== FIN DIAGNOSTIC ==========
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

