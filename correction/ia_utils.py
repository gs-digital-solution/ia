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
#from .tasks import generer_un_exercice
#from celery import group
import logging
# Logger dédié
logger = logging.getLogger(__name__)


def debug_table_detection(corrige_text):
    """
    Fonction de debug pour analyser comment les tableaux sont détectés.
    """
    print("\n" + "=" * 60)
    print("🔍 DEBUG DÉTECTION DE TABLEAUX")
    print("=" * 60)

    lines = corrige_text.strip().split('\n')
    table_count = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if '|' in line:
            is_table, end_idx, table_lines = detect_table(lines, i)
            if is_table:
                table_count += 1
                print(f"\n📋 TABLEAU #{table_count} détecté (lignes {i}-{end_idx - 1})")
                print(f"   Lignes: {len(table_lines)}")
                print(f"   Première ligne: {table_lines[0][:80]}...")
                print(f"   Dernière ligne: {table_lines[-1][:80]}...")

                # Tester le formatage
                try:
                    html = format_table_markdown('\n'.join(table_lines))
                    print(f"   ✅ Formatage réussi: {len(html)} caractères HTML")
                except Exception as e:
                    print(f"   ❌ Erreur formatage: {e}")

                i = end_idx
                continue

        i += 1

    print(f"\n✅ Total tableaux détectés: {table_count}")
    print("=" * 60 + "\n")

    return table_count

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

def analyser_document_scientifique(fichier_path: str) -> dict:
    """
    Analyse scientifique avancée avec deepseek-vl2 :
    - OCR (Tesseract) en fallback
    - appel multimodal deepseek-vl2 pour texte + schémas
    Retourne un dict avec :
      - texte_complet (str)
      - elements_visuels (list of captions)
      - formules_latex  (list of LaTeX strings)
      - graphs          (list of dicts graphiques)
      - angles          (list of {"valeur","unité","coord"})
      - numbers         (list of {"valeur","unité","coord"})
      - structure_exercices (list)
    """
    logger.info("🔍 Début analyse scientifique pour %s", fichier_path)

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
        captions     = vision_json.get("captions", [])
        latex_blocks = vision_json.get("latex_blocks", [])
        graphs       = vision_json.get("graphs", [])
        angles       = vision_json.get("angles", [])
        numbers      = vision_json.get("numbers", [])
        struct_exos  = vision_json.get("structure_exercices", [])

        logger.info("✅ deepseek-vl2 OK : texte %d chars, %d schémas, %d formules, %d angles, %d nombres",
                    len(texte_json), len(captions), len(latex_blocks), len(angles), len(numbers))

        return {
            "texte_complet": texte_json,
            "elements_visuels": captions,
            "formules_latex": latex_blocks,
            "graphs": graphs,
            "angles": angles,
            "numbers": numbers,
            "structure_exercices": struct_exos
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
            "structure_exercices": []
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


def format_html_table(table_text):
    """
    Convertit un tableau HTML (même mal formaté) en HTML propre.
    """
    print(f"🌐 Formatage tableau HTML: {len(table_text)} caractères")
    print(f"   Texte HTML brut: {table_text[:200]}...")

    # Nettoyer le HTML
    html_text = table_text.strip()

    # 1. CAS SPÉCIAL: HTML mal formé sans balises fermantes
    # Exemple: "<table>Notes[0,20[[20,40[[40,60[[60,80[[80,100[Effectifs4625510</table>"
    if '<table>' in html_text.lower() and '</table>' in html_text.lower():
        # Extraire le contenu entre <table> et </table>
        start = html_text.lower().find('<table>')
        end = html_text.lower().find('</table>') + len('</table>')

        if start != -1 and end != -1:
            table_html = html_text[start:end]
            print(f"   Tableau HTML extrait: {table_html[:150]}...")

            # Si le HTML est valide, le retourner tel quel
            if '<tr>' in table_html or '<td>' in table_html:
                return f'<div class="table-container">{table_html}</div>'

    # 2. CAS: HTML très mal formaté (comme dans ton exemple)
    # "<table>Notes[0,20[[20,40[[40,60[[60,80[[80,100[Effectifs4625510</table>"
    # On va essayer de le parser manuellement

    # Nettoyer les balises
    html_text = html_text.replace('<table>', '').replace('</table>', '').replace('<TABLE>', '').replace('</TABLE>', '')
    html_text = html_text.strip()

    print(f"   Contenu nettoyé: {html_text[:150]}...")

    # Essayer de détecter la structure
    # Exemple: "Notes[0,20[[20,40[[40,60[[60,80[[80,100[Effectifs4625510"
    # C'est probablement: En-tête: Notes, puis données: Effectifs

    # Chercher des patterns
    # Pattern 1: "[0,20[", "[20,40[", etc.
    intervals = re.findall(r'\[[^]]+\]', html_text)

    # Pattern 2: Chiffres consécutifs (effectifs)
    numbers = re.findall(r'\d+', html_text)

    print(f"   Intervalles détectés: {intervals}")
    print(f"   Nombres détectés: {numbers}")

    # Si on a des intervalles et des nombres, construire un tableau
    if intervals and numbers and len(numbers) >= len(intervals):
        # Construire un tableau HTML
        html = ['<div class="table-container"><table>']

        # En-tête
        html.append('<thead><tr>')
        html.append('<th>Notes</th>')
        for interval in intervals:
            html.append(f'<th>{interval}</th>')
        html.append('</tr></thead>')

        # Corps
        html.append('<tbody><tr>')
        html.append('<td>Effectifs</td>')
        for i in range(len(intervals)):
            if i < len(numbers):
                html.append(f'<td>{numbers[i]}</td>')
        html.append('</tr></tbody></table></div>')

        return ''.join(html)

    # 3. CAS: HTML simple mais mal formaté
    # Essayer d'ajouter des balises manquantes
    if '[' in html_text and ']' in html_text:
        # C'est probablement un tableau de données
        # Exemple: "[0,20[ [20,40[ [40,60[ ..."
        html = ['<div class="table-container"><table><tbody>']

        # Diviser par les doubles crochets
        parts = re.split(r'\]\s*\[', html_text)
        parts = [p + ']' if not p.endswith(']') else p for p in parts]

        for part in parts:
            if part.strip():
                html.append('<tr>')
                # Essayer de séparer les cellules
                cells = re.split(r'[,\[\]]+', part)
                cells = [c for c in cells if c.strip()]
                for cell in cells:
                    html.append(f'<td>{cell.strip()}</td>')
                html.append('</tr>')

        html.append('</tbody></table></div>')
        return ''.join(html)

    # 4. CAS: Texte brut qu'on va mettre dans un tableau simple
    print("⚠️ Impossible de parser le HTML, tableau simple de secours")
    return f'<div class="table-container"><table><tr><td>{html_text}</td></tr></table></div>'



def format_table_markdown(table_text):
    """
    Convertit un tableau markdown en HTML avec support des alignements et séparateurs.
    Version ULTRA-ROBUSTE pour gérer les tableaux mal formatés de l'IA.

    Args:
        table_text (str): Tableau au format markdown

    Returns:
        str: HTML du tableau
    """
    print(f"🔄 Formatage tableau : {len(table_text)} caractères")

    # DEBUG: Afficher le tableau original
    print(f"📋 Tableau original (premiers 200 chars): {table_text[:200].replace(chr(10), '\\n')}...")

    # Nettoyer d'abord le texte du tableau
    table_text = clean_table_text(table_text)

    # DEBUG: Afficher le tableau nettoyé
    print(f"🧹 Tableau nettoyé (premiers 200 chars): {table_text[:200].replace(chr(10), '\\n')}...")

    lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]

    if len(lines) < 1:
        print("⚠️ Tableau vide après nettoyage")
        return f'<div class="table-container"><p class="table-error">Tableau non formatable (vide)</p></div>'

    print(f"   Lignes après nettoyage: {len(lines)}")
    for idx, line in enumerate(lines):
        print(f"   [{idx}] '{line[:80]}'")

    # TENTATIVE DE RÉPARATION SI LE TABLEAU SEMBLE CASSÉ
    if len(lines) >= 2:  # Assez de lignes pour potentiellement être cassé
        print("🔧 Tentative de réparation du tableau...")
        repaired_lines = repair_broken_table(lines)
        if repaired_lines != lines:
            print(f"   ✅ Tableau réparé: {len(repaired_lines)} lignes")
            for idx, line in enumerate(repaired_lines):
                print(f"   [{idx} réparé] '{line[:80]}'")
            lines = repaired_lines
        else:
            print("   ℹ️  Aucune réparation nécessaire")

    # ANALYSE DÉTAILLÉE DE LA STRUCTURE
    print("🔍 Analyse de la structure du tableau:")

    separator_indices = []
    header_candidates = []
    data_lines = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Ligne de séparation
        if re.match(r'^[\|\s:\-]+$', line_stripped) and '|' in line_stripped:
            separator_indices.append(i)
            print(f"   Ligne {i}: SÉPARATEUR - '{line_stripped[:50]}...'")

        # Ligne avec du texte (potentiel en-tête)
        elif '|' in line_stripped and re.search(r'[a-zA-ZÀ-ÿ\d]', line_stripped):
            if re.search(r'[a-zA-ZÀ-ÿ]', line_stripped):  # Contient des lettres
                header_candidates.append(i)
                print(f"   Ligne {i}: EN-TÊTE POTENTIEL - '{line_stripped[:50]}...'")
            else:
                data_lines.append(i)
                print(f"   Ligne {i}: DONNÉES - '{line_stripped[:50]}...'")

        else:
            print(f"   Ligne {i}: AUTRE - '{line_stripped[:50]}...'")

    print(
        f"   Résumé: {len(separator_indices)} séparateurs, {len(header_candidates)} en-têtes potentiels, {len(data_lines)} lignes de données")

    # CAS SPÉCIAL : Tableau avec la structure exacte de l'exemple cassé
    # Format: "| Titre | ... |" suivi de "|---|---|" suivi de "| --- | --- |" suivi de "| Données | ... |"
    if len(lines) >= 3:
        # Recherche du pattern spécifique: ligne1=entête, ligne2=séparateur1, ligne3=séparateur2
        if (len(lines) >= 3 and
                '|' in lines[0] and
                re.match(r'^[\|\s:\-]+$', lines[1]) and
                re.match(r'^[\|\s:\-]+$', lines[2])):
            print("⚠️ Détecté pattern de tableau cassé (2 séparateurs consécutifs)")
            # Fusionner les 2 séparateurs en un
            merged_separator = merge_separator_lines(lines[1], lines[2])
            lines = [lines[0], merged_separator] + lines[3:]
            separator_indices = [1]  # Mettre à jour l'indice du séparateur
            print(f"   Séparateurs fusionnés: '{merged_separator}'")

    # DÉCISION DU FORMATAGE BASÉE SUR LA STRUCTURE ANALYSÉE

    # Cas 1: Structure markdown classique (entête + séparateur + données)
    if (len(header_candidates) >= 1 and
            len(separator_indices) >= 1 and
            min(header_candidates) < min(separator_indices)):

        print("✅ Structure markdown classique détectée")
        separator_idx = min(separator_indices)

        # Vérifier la cohérence des colonnes
        if separator_idx > 0:
            header_line = lines[header_candidates[0]]
            separator_line = lines[separator_idx]

            header_cols = header_line.count('|') - 1
            separator_cols = separator_line.count('|') - 1

            print(f"   Colonnes: en-tête={header_cols}, séparateur={separator_cols}")

            if header_cols > 0 and separator_cols > 0:
                # Standardiser si nécessaire
                if header_cols != separator_cols:
                    print(f"   ⚠️ Incohérence de colonnes, tentative d'ajustement")
                    lines = standardize_table_columns(lines, max(header_cols, separator_cols))

                return format_markdown_table_with_separator(lines, separator_idx)

    # Cas 2: Aucun séparateur explicite mais plusieurs lignes avec pipes
    elif not separator_indices and len(lines) >= 2:
        print("ℹ️  Tableau sans séparateur explicite")
        # Vérifier si toutes les lignes ont à peu près le même nombre de pipes
        pipe_counts = [line.count('|') - 1 for line in lines if '|' in line]
        if pipe_counts and max(pipe_counts) - min(pipe_counts) <= 2:
            print(f"   Structure cohérente: {min(pipe_counts)}-{max(pipe_counts)} colonnes")
            return format_simple_table(lines)

    # Cas 3: Plusieurs séparateurs (tableau complexe avec sous-sections)
    elif len(separator_indices) >= 2:
        print("ℹ️  Tableau complexe avec plusieurs séparateurs")
        return format_complex_table(lines, separator_indices)

    # CAS DE SECOURS : Formatage simple de toute façon
    print("⚠️ Structure non reconnue, formatage simple de secours")

    # Nettoyer et formater toutes les lignes avec pipes
    html_lines = ['<div class="table-container"><table><tbody>']

    for line in lines:
        if '|' not in line:
            continue

        line_clean = re.sub(r'^\|\s*', '', line)
        line_clean = re.sub(r'\s*\|$', '', line_clean)
        cells = [cell.strip() for cell in line_clean.split('|')]

        if cells:
            html_lines.append('<tr>')

            # Déterminer si c'est probablement un en-tête (première ligne ou ligne avec du texte)
            is_header = (lines.index(line) == 0 and
                         any(re.search(r'[a-zA-ZÀ-ÿ]', cell) for cell in cells))

            for cell in cells:
                if is_header:
                    html_lines.append(f'<th>{cell}</th>')
                else:
                    html_lines.append(f'<td>{cell}</td>')

            html_lines.append('</tr>')

    html_lines.append('</tbody></table></div>')

    result = ''.join(html_lines)
    print(f"✅ Formatage de secours terminé: {len(result)} caractères HTML")
    return result


def clean_table_text(table_text):
    """
    Nettoie le texte des tableaux avant traitement.
    Version ULTRA-ROBUSTE pour gérer les tableaux mal formatés de l'IA.
    """
    lines = table_text.strip().split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            # Garder les lignes vides dans les tableaux (peuvent être des séparateurs)
            cleaned_lines.append(line)
            continue

        # CORRECTION CRITIQUE : L'IA génère parfois "|---|---|" sur plusieurs lignes
        # On doit fusionner ces lignes
        if re.match(r'^[\|\s:\-]+$', line) and '|' in line:
            # C'est une ligne de séparation
            cleaned_lines.append(line)
            continue

        # Normaliser les pipes - TOUJOURS avoir | au début et à la fin
        if not line.startswith('|'):
            line = '| ' + line
        if not line.endswith('|'):
            line = line + ' |'

        # Remplacer les séparateurs de cellule multiples
        line = re.sub(r'\|\s*\|\s*\|', '| | |', line)  # Cellules vides consécutives
        line = re.sub(r'\s{2,}', ' ', line)  # Multiples espaces

        # Nettoyer les cellules vides
        line = re.sub(r'\|\s+\|', '| |', line)

        cleaned_lines.append(line)

    # PHASE 2 : Fusionner les séparateurs brisés sur plusieurs lignes
    final_lines = []
    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]

        # Si c'est une ligne de séparation incomplète
        if re.match(r'^[\|\s:\-]+$', line) and '|' in line:
            # Regarder si la ligne suivante est aussi un séparateur
            j = i + 1
            while j < len(cleaned_lines) and re.match(r'^[\|\s:\-]+$', cleaned_lines[j]) and '|' in cleaned_lines[j]:
                # Fusionner les séparateurs
                line = merge_separator_lines(line, cleaned_lines[j])
                j += 1

            if j > i + 1:
                print(f"⚠️ Fusionné {j - i} lignes de séparateur en une seule")
                i = j - 1  # Sauter les lignes fusionnées

        final_lines.append(line)
        i += 1

    return '\n'.join(final_lines)


def merge_separator_lines(line1, line2):
    """
    Fusionne deux lignes de séparateur de tableau.
    Exemple: "|---|---|" + "---|---|---|" → "|---|---|---|---|"
    """
    # Extraire les parties entre pipes
    parts1 = [p.strip() for p in line1.split('|') if p.strip() != '']
    parts2 = [p.strip() for p in line2.split('|') if p.strip() != '']

    # Combiner, en gardant les plus longs séparateurs
    combined_parts = []
    for i in range(max(len(parts1), len(parts2))):
        p1 = parts1[i] if i < len(parts1) else ''
        p2 = parts2[i] if i < len(parts2) else ''

        # Prendre le plus long séparateur
        if len(p2) > len(p1):
            combined_parts.append(p2)
        else:
            combined_parts.append(p1)

    # Reconstruire la ligne
    separator_line = '|' + '|'.join(combined_parts) + '|'
    return separator_line


def repair_broken_table(table_lines):
    """
    Tente de réparer un tableau cassé/mal formaté généré par l'IA.
    Retourne les lignes réparées.
    """
    if not table_lines:
        return table_lines

    print(f"🛠️  Tentative réparation tableau {len(table_lines)} lignes")

    # 1. Identifier les lignes d'en-tête (celles avec du texte, pas juste des séparateurs)
    header_candidates = []
    separator_indices = []

    for idx, line in enumerate(table_lines):
        line = line.strip()
        if not line:
            continue

        # Ligne de séparation
        if re.match(r'^[\|\s:\-]+$', line) and '|' in line:
            separator_indices.append(idx)
        # Ligne avec du texte (potentiel en-tête)
        elif '|' in line and re.search(r'[a-zA-ZÀ-ÿ\d]', line):
            header_candidates.append(idx)

    # 2. Si on a une structure typique: en-tête → séparateur → données
    if len(header_candidates) >= 1 and len(separator_indices) >= 1:
        # Trier pour avoir l'ordre
        header_idx = min(header_candidates)
        separator_idx = min(separator_indices)

        if header_idx < separator_idx:
            print(f"   Structure détectée: en-tête ligne {header_idx}, séparateur ligne {separator_idx}")

            # Vérifier la cohérence du nombre de colonnes
            header_line = table_lines[header_idx]
            separator_line = table_lines[separator_idx]

            header_cols = header_line.count('|') - 1
            separator_cols = separator_line.count('|') - 1

            if header_cols > 0 and separator_cols > 0:
                # Standardiser le nombre de colonnes
                max_cols = max(header_cols, separator_cols)

                repaired_lines = []
                for idx, line in enumerate(table_lines):
                    line = line.strip()
                    if not line:
                        repaired_lines.append(line)
                        continue

                    # Compter les colonnes actuelles
                    current_cols = line.count('|') - 1
                    if current_cols < max_cols:
                        # Ajouter des colonnes manquantes
                        missing = max_cols - current_cols
                        if '|' in line:
                            if line.endswith('|'):
                                line = line + ' |' * missing
                            else:
                                line = line + '|' + ' |' * missing
                        else:
                            line = '| ' + line + ' |' + ' |' * (missing - 1)

                    repaired_lines.append(line)

                print(f"   Réparé: {max_cols} colonnes standardisées")
                return repaired_lines

    # 3. Si réparation échoue, retourner les lignes nettoyées mais garder la structure
    cleaned_lines = []
    for line in table_lines:
        line = line.strip()
        if line:
            # Assurer au moins un format de tableau valide
            if '|' not in line:
                line = '| ' + line + ' |'
            cleaned_lines.append(line)

    return cleaned_lines


def standardize_table_columns(lines, target_cols):
    """
    Standardise toutes les lignes du tableau pour avoir le même nombre de colonnes.
    """
    print(f"📏 Standardisation à {target_cols} colonnes")

    standardized_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            standardized_lines.append(line)
            continue

        # Compter les colonnes actuelles
        current_cols = line.count('|') - 1

        if current_cols < target_cols:
            # Ajouter des colonnes manquantes
            missing = target_cols - current_cols
            if line.endswith('|'):
                line = line + ' |' * missing
            else:
                line = line + '|' + ' |' * missing
            print(f"   Ligne ajustée: {current_cols} → {target_cols} colonnes")

        standardized_lines.append(line)

    return standardized_lines


def format_simple_table(lines):
    """
    Format un tableau simple sans séparateur explicite.
    """
    html = ['<div class="table-container"><table>']

    for i, line in enumerate(lines):
        line_clean = re.sub(r'^\|\s*', '', line)
        line_clean = re.sub(r'\s*\|$', '', line_clean)
        cells = [cell.strip() for cell in line_clean.split('|')]

        if cells:
            # Déterminer si c'est un en-tête (première ligne avec du texte)
            is_header = (i == 0 and any(re.search(r'[a-zA-ZÀ-ÿ]', cell) for cell in cells))

            if is_header:
                html.append('<thead><tr>')
                tag = 'th'
            else:
                if i == (1 if is_header else 0):
                    html.append('<tbody>')
                html.append('<tr>')
                tag = 'td'

            for cell in cells:
                html.append(f'<{tag}>{cell}</{tag}>')

            if is_header:
                html.append('</tr></thead>')
            else:
                html.append('</tr>')

    html.append('</tbody></table></div>')
    return ''.join(html)


def format_markdown_table_with_separator(lines, separator_idx):
    """
    Format un tableau markdown avec un séparateur explicite.
    """
    print(f"📊 Formatage tableau markdown avec séparateur à ligne {separator_idx}")

    # Lignes avant le séparateur = header
    header_lines = lines[:separator_idx]
    separator_line = lines[separator_idx]
    body_lines = lines[separator_idx + 1:] if separator_idx + 1 < len(lines) else []

    # Parser la première ligne d'en-tête
    first_header = header_lines[0] if header_lines else ""
    first_header = re.sub(r'^\|\s*', '', first_header)
    first_header = re.sub(r'\s*\|$', '', first_header)
    header_cells = [cell.strip() for cell in first_header.split('|')]

    # Déterminer les alignements depuis la ligne de séparation
    separator_line = re.sub(r'^\|\s*', '', separator_line)
    separator_line = re.sub(r'\s*\|$', '', separator_line)
    separator_cells = [cell.strip() for cell in separator_line.split('|')]

    alignments = ['left'] * len(header_cells)
    for i, cell in enumerate(separator_cells):
        if i < len(alignments):
            if cell.startswith(':') and cell.endswith(':'):
                alignments[i] = 'center'
            elif cell.endswith(':'):
                alignments[i] = 'right'
            else:
                alignments[i] = 'left'

    # Construire le HTML
    html = ['<div class="table-container"><table>']

    # En-tête
    if header_cells:
        html.append('<thead><tr>')
        for i, cell in enumerate(header_cells):
            align = alignments[i] if i < len(alignments) else 'left'
            html.append(f'<th style="text-align: {align};">{cell}</th>')
        html.append('</tr></thead>')

    # Corps
    if body_lines:
        html.append('<tbody>')
        for line in body_lines:
            line = re.sub(r'^\|\s*', '', line)
            line = re.sub(r'\s*\|$', '', line)
            cells = [cell.strip() for cell in line.split('|')]
            if cells:
                html.append('<tr>')
                for i, cell in enumerate(cells):
                    align = alignments[i] if i < len(alignments) else 'left'
                    html.append(f'<td style="text-align: {align};">{cell}</td>')
                html.append('</tr>')
        html.append('</tbody>')

    html.append('</table></div>')

    result = ''.join(html)
    print(f"✅ Tableau formaté: {len(header_cells)} colonnes, {len(body_lines)} lignes de données")
    return result


def format_complex_table(lines, separator_indices):
    """
    Format un tableau avec plusieurs séparateurs (plusieurs headers).
    """
    print(f"📊 Formatage tableau complexe avec {len(separator_indices)} séparateurs")

    html = ['<div class="table-container"><table>']

    current_section = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Si c'est une ligne de séparation
        if i in separator_indices:
            # Fermer la section précédente si ouverte
            if current_section == 'thead':
                html.append('</tr></thead>')
                current_section = None
            elif current_section == 'tbody':
                html.append('</tbody>')
                current_section = None

            # Déterminer la section suivante
            if i + 1 < len(lines) and i + 1 not in separator_indices:
                next_line = lines[i + 1].strip()
                if '|' in next_line:
                    # Vérifier si la ligne suivante ressemble à un en-tête
                    has_text = bool(re.search(r'[a-zA-ZÀ-ÿ]', next_line))
                    if has_text:
                        html.append('<thead><tr>')
                        current_section = 'thead'
                    else:
                        html.append('<tbody>')
                        current_section = 'tbody'
            i += 1
            continue

        # Traiter la ligne de contenu
        line_clean = re.sub(r'^\|\s*', '', line)
        line_clean = re.sub(r'\s*\|$', '', line_clean)
        cells = [cell.strip() for cell in line_clean.split('|')]

        if cells:
            if current_section == 'thead':
                # C'est une ligne d'en-tête
                for cell in cells:
                    html.append(f'<th>{cell}</th>')
            else:
                # C'est une ligne du corps
                if current_section != 'tbody':
                    html.append('<tbody>')
                    current_section = 'tbody'

                html.append('<tr>')
                for cell in cells:
                    html.append(f'<td>{cell}</td>')
                html.append('</tr>')

        i += 1

    # Fermer les sections ouvertes
    if current_section == 'thead':
        html.append('</tr></thead>')
    elif current_section == 'tbody':
        html.append('</tbody>')

    html.append('</table></div>')

    result = ''.join(html)
    print(f"✅ Tableau complexe formaté: {len(result)} caractères HTML")
    return result


def detect_table(lines, start_idx):
    """
    Détecte si un tableau commence à l'index donné.
    Version ULTRA-TOLÉRANTE pour les tableaux mal formatés de l'IA.
    """
    current_line = lines[start_idx].strip()

    # CRITÈRE ÉLARGI :
    # 1. Tableau markdown (pipes)
    # 2. Tableau HTML (balises <table>)
    # 3. Ligne de séparation
    has_pipes = '|' in current_line
    has_table_tag = '<table>' in current_line.lower() or '</table>' in current_line.lower()
    is_separator = re.match(r'^[\|\s:\-]+$', current_line)

    if not (has_pipes or has_table_tag or is_separator):
        return False, start_idx, []

    # Pour debug
    print(f"🔍 Détection tableau à ligne {start_idx}: '{current_line[:50]}...'")
    if has_table_tag:
        print(f"   ⚡ BALISE HTML DÉTECTÉE: {current_line[:100]}")

    table_lines = []
    i = start_idx
    max_lines = 20  # Limite pour éviter les faux positifs

    while i < len(lines) and i - start_idx < max_lines:
        line = lines[i].strip()

        # CRITÈRE ÉLARGI : Accepter plus de types de lignes comme faisant partie du tableau
        is_table_line = False

        # 1. Ligne avec des pipes (markdown)
        if '|' in line:
            is_table_line = True

        # 2. Ligne avec balise HTML table
        elif '<table>' in line.lower() or '</table>' in line.lower() or '<td>' in line.lower() or '<tr>' in line.lower():
            is_table_line = True
            print(f"   ⚡ Ligne {i}: Balise HTML détectée")

        # 3. Ligne de séparation markdown
        elif re.match(r'^[\|\s:\-]+$', line):
            is_table_line = True

        # 4. Ligne vide ENTRE les lignes de tableau (tolérance)
        elif not line and len(table_lines) > 0:
            # Vérifier si la ligne suivante continue le tableau
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                has_next_pipes = '|' in next_line
                has_next_html = any(tag in next_line.lower() for tag in ['<table>', '</table>', '<td>', '<tr>'])
                is_next_separator = re.match(r'^[\|\s:\-]+$', next_line)

                if has_next_pipes or has_next_html or is_next_separator:
                    is_table_line = True

        if is_table_line:
            table_lines.append(line)
            i += 1
        else:
            # Vérifier si on a assez de lignes pour former un tableau
            if len(table_lines) >= 1:  # Réduit à 1 pour HTML
                # Compter les lignes valides
                valid_table_lines = []
                for l in table_lines:
                    has_p = '|' in l
                    has_html = any(tag in l.lower() for tag in ['<table>', '</table>', '<td>', '<tr>'])
                    is_sep = re.match(r'^[\|\s:\-]+$', l)
                    if has_p or has_html or is_sep:
                        valid_table_lines.append(l)

                if len(valid_table_lines) >= 1:  # Réduit à 1 pour HTML
                    print(f"✅ Tableau détecté: {len(table_lines)} lignes (HTML: {has_table_tag})")
                    return True, i, table_lines
                else:
                    return False, start_idx, []
            else:
                return False, start_idx, []

    # Fin de fichier atteinte
    if len(table_lines) >= 1:
        valid_table_lines = []
        for l in table_lines:
            has_p = '|' in l
            has_html = any(tag in l.lower() for tag in ['<table>', '</table>', '<td>', '<tr>'])
            is_sep = re.match(r'^[\|\s:\-]+$', l)
            if has_p or has_html or is_sep:
                valid_table_lines.append(l)

        if len(valid_table_lines) >= 1:
            print(f"✅ Tableau détecté (fin fichier): {len(table_lines)} lignes")
            return True, i, table_lines

    return False, start_idx, []


def generate_corrige_html(corrige_text):
    """Transforme le corrigé brut en HTML stylisé en PRÉSERVANT les tableaux déjà formatés."""
    if not corrige_text:
        return ""

    print("🔧 Génération HTML - DÉBUT")
    print(f"   Longueur texte: {len(corrige_text)} caractères")

    # DÉTECTION DES TABLEAUX DÉJÀ FORMATÉS EN HTML
    # Chercher les blocs HTML complets <table>...</table>
    import re

    # Pattern pour détecter les tableaux HTML complets
    table_pattern = r'(<table\b[^>]*>.*?</table>)'

    # Diviser le texte en blocs : tableaux HTML vs texte normal
    parts = []
    last_end = 0

    for match in re.finditer(table_pattern, corrige_text, re.DOTALL | re.IGNORECASE):
        # Texte avant le tableau
        if match.start() > last_end:
            text_part = corrige_text[last_end:match.start()]
            parts.append(('text', text_part))

        # Le tableau HTML
        table_html = match.group(1)
        parts.append(('table', table_html))
        last_end = match.end()

    # Dernière partie
    if last_end < len(corrige_text):
        parts.append(('text', corrige_text[last_end:]))

    print(f"   {len(parts)} parties détectées")

    # Traitement séparé
    html_output = []

    # Branding CIS en haut
    html_output.append(
        '<div class="cis-message"><strong>SUJET CORRIGÉ PAR L\'APPLICATION CIS, DISPO SUR PLAYSTORE</strong></div>')

    for part_type, content in parts:
        if part_type == 'table':
            # TABLEAU HTML - NE RIEN FAIRE, juste l'encapsuler
            print(f"   📊 Tableau HTML préservé: {len(content)} caractères")
            html_output.append(f'<div class="table-container">{content}</div>')

        else:
            # TEXTE NORMAL - le traiter comme avant
            html_output.append(process_text_part(content))

    result = "".join(html_output)
    print(f"✅ Génération HTML terminée: {len(result)} caractères")
    return mark_safe(result)


def process_text_part(text):
    """Traite une partie de texte (sans tableaux HTML)."""
    if not text.strip():
        return ""

    lines = text.strip().split('\n')
    html_lines = []

    # Pattern pour détecter les débuts d'exercice/partie
    pattern_exercice = re.compile(r'^(EXERCICE\s*\d+|PARTIE\s*[IVXLCDM]+|Exercice\s*\d+|Partie\s*[IVXLCDM]+)',
                                  re.IGNORECASE)

    in_bloc_exercice = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Début d'un nouvel exercice/partie
        if pattern_exercice.match(line):
            if in_bloc_exercice:
                html_lines.append('</div>')
            html_lines.append(f'<div class="bloc-exercice"><h1 class="titre-exercice">{line}</h1>')
            in_bloc_exercice = True
            i += 1
            continue

        # Détection des tableaux markdown DANS LE TEXTE SEULEMENT
        is_table, table_end_idx, table_lines = detect_table(lines, i)
        if is_table:
            print(f"   📋 Tableau markdown détecté dans texte")
            html_table = format_table_markdown('\n'.join(table_lines))
            html_lines.append(html_table)
            i = table_end_idx
            continue

        # Traitement normal des lignes de texte
        html_lines.append(format_text_line(line))
        i += 1

    if in_bloc_exercice:
        html_lines.append('</div>')

    return "".join(html_lines)


def format_text_line(line):
    """Formate une ligne de texte simple."""
    if not line:
        return ""

    # Sous-titre question principale
    if re.match(r'^Question\s*\d+', line, re.IGNORECASE):
        return f'<h2 class="titre-question">{line}</h2>'

    # Sous-titre secondaire
    if re.match(r'^\d+\.', line):
        return f'<h3 class="titre-question">{line}</h3>'

    # Sous-question
    if re.match(r'^[a-z]\)', line):
        return f'<p><strong>{line}</strong></p>'

    # Formules LaTeX
    if '\\(' in line or '\\[' in line:
        return f'<p class="reponse-question mathjax">{line}</p>'

    # Listes
    if line.startswith('•') or line.startswith('-'):
        return f'<p>{line}</p>'

    # Paragraphe normal
    return f'<p class="reponse-question">{line}</p>'


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

RÈGLES ABSOLUES POUR LES TABLEAUX :

1. ✅ TOUS les tableaux doivent être en HTML COMPLET, pas en markdown !
2. ✅ Format : 
   <table>
   <thead>
   <tr><th>Colonne1</th><th>Colonne2</th></tr>
   </thead>
   <tbody>
   <tr><td>Donnée1</td><td>Donnée2</td></tr>
   </tbody>
   </table>

3. ✅ Pour les tableaux de variation :
   <table>
   <thead>
   <tr>
     <th>x</th>
     <th>-∞</th>
     <th>x₁</th>
     <th>x₂</th>
     <th>+∞</th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <td>f'(x)</td>
     <td>+</td>
     <td>0</td>
     <td>-</td>
     <td>+</td>
   </tr>
   <tr>
     <td>f(x)</td>
     <td>↗</td>
     <td>max</td>
     <td>↘</td>
     <td>↗</td>
   </tr>
   </tbody>
   </table>

4. ✅ Pour les tableaux de signes :
   <table class="sign-table">
   <thead>
   <tr>
     <th>x</th>
     <th>-∞</th>
     <th>racine</th>
     <th>+∞</th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <td>f(x)</td>
     <td>+</td>
     <td>0</td>
     <td>-</td>
   </tr>
   </tbody>
   </table>

EXEMPLES CORRECTS :

--- TABLEAU STATISTIQUE ---
<table>
<thead>
<tr>
<th>Notes</th>
<th>[0,20[</th>
<th>[20,40[</th>
<th>[40,60[</th>
<th>[60,80[</th>
<th>[80,100]</th>
</tr>
</thead>
<tbody>
<tr>
<td>Effectifs</td>
<td>4</td>
<td>6</td>
<td>25</td>
<td>5</td>
<td>10</td>
</tr>
</tbody>
</table>

--- TABLEAU DE VARIATION ---
<table class="variation-table">
<thead>
<tr>
<th>x</th>
<th>-∞</th>
<th>-1</th>
<th>3</th>
<th>+∞</th>
</tr>
</thead>
<tbody>
<tr>
<td>f'(x)</td>
<td>+</td>
<td>0</td>
<td>-</td>
<td>0</td>
</tr>
<tr>
<td>f(x)</td>
<td>↗</td>
<td>4</td>
<td>↘</td>
<td>-2</td>
</tr>
</tbody>
</table>

NE JAMAIS UTILISER :
- ❌ Markdown (| --- | --- |)
- ❌ Pipes simples
- ❌ Séparateurs incomplets

TOUJOURS UTILISER :
- ✅ Balises HTML complètes
- ✅ <thead> pour les en-têtes
- ✅ <tbody> pour les données
- ✅ Classes CSS pour le style


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
            analyse_complete = analyser_document_scientifique(local_path)
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


@shared_task(name='correction.ia_utils.generer_corrige_exercice_async')
def generer_corrige_exercice_async(soumission_id):
    """
    Tâche asynchrone pour corriger UN exercice isolé.
    Version mise à jour avec système unifié.
    """
    try:
        soum = SoumissionIA.objects.get(id=soumission_id)
        dem = soum.demande

        # 1) Préparer le texte complet depuis le fichier d'énoncé
        texte = extraire_texte_fichier(dem.fichier)

        # 2) Séparer et extraire le fragment avec la NOUVELLE fonction
        exercices_data = separer_exercices_avec_titres(texte)
        idx = soum.exercice_index or 0

        # Vérifier l'index
        if idx >= len(exercices_data):
            print(f"⚠️ Index {idx} hors limites, utilisation du dernier exercice")
            idx = len(exercices_data) - 1

        ex_data = exercices_data[idx]
        fragment = ex_data['contenu']

        print(f"✅ Exercice {idx + 1} extrait: {ex_data.get('titre', 'Sans titre')}")
        print(f"   Longueur contenu: {len(fragment)} caractères")

        # 3) Mise à jour statut pour analyse IA
        soum.statut = 'analyse_ia'
        soum.progression = 20
        soum.save()

        # 4) Lancer la génération (IA + graph) sur ce fragment
        mat = dem.matiere if dem.matiere else Matiere.objects.first()
        contexte = f"Exercice de {mat.nom} – {ex_data.get('titre', f'Exercice {idx + 1}')}"

        corrige_txt, _ = generer_corrige_ia_et_graphique(
            texte_enonce=fragment,
            contexte=contexte,
            matiere=mat,
            demande=dem
        )

        # 5) Mise à jour PDF
        soum.statut = 'formatage_pdf'
        soum.progression = 60
        soum.save()

        pdf_url = generer_pdf_corrige(
            {
                "titre_corrige": contexte,
                "corrige_html": corrige_txt,
                "soumission_id": soum.id,
                "titre_exercice": ex_data.get('titre_complet', f"Exercice {idx + 1}")
            },
            soum.id
        )

        # 6) Débit de crédit
        if not debiter_credit_abonnement(dem.user):
            soum.statut = 'erreur_credit'
            soum.save()
            return False

        # 7) CRÉATION DU CorrigePartiel - AVEC TITRE RÉEL
        pdf_relative_path = pdf_url.replace(settings.MEDIA_URL, '')
        pdf_absolute_path = os.path.join(settings.MEDIA_ROOT, pdf_relative_path)

        # Utiliser le titre réel de l'exercice
        titre_reel = ex_data.get('titre_complet', ex_data.get('titre', f"Exercice {idx + 1}"))

        # Nettoyer un peu le titre si trop long
        if len(titre_reel) > 200:
            titre_reel = titre_reel[:197] + "..."

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

        # 8) Finalisation
        soum.statut = 'termine'
        soum.progression = 100
        soum.resultat_json = {
            "exercice_index": idx,
            "exercice_titre": titre_reel,
            "corrige_text": corrige_txt,
            "pdf_url": pdf_url,
            "exercice_data": ex_data  # Stocker toutes les données de l'exercice
        }
        soum.save()

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

