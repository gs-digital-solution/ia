import requests
import os
import tempfile
import json
import re
import numpy as np
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


def generer_corrige_par_exercice(texte_exercice, contexte, matiere=None):
    """
    Génère le corrigé pour un seul exercice et extrait graphiques éventuels.
    """
    print("🎯 Génération corrigé pour exercice individuel...")

    system_prompt = DEFAULT_SYSTEM_PROMPT
    consignes_finales = "Format de réponse strict : LaTeX pour les maths, explications détaillées mais concises"

    if matiere and hasattr(matiere, 'prompt_ia'):
        promptia = matiere.prompt_ia
        system_prompt = promptia.system_prompt or system_prompt
        consignes_finales = promptia.consignes_finales or consignes_finales

    prompt_ia = f"""
{system_prompt}

### CONTEXTE
{contexte}

### EXERCICE À CORRIGER (UNIQUEMENT CELUI-CI)
{texte_exercice.strip()}

### CONSIGNES
{consignes_finales}

**Important : Réponds UNIQUEMENT à cet exercice. Sois complet mais concis.**
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
        "max_tokens": 8000,
        "top_p": 0.9,
        "frequency_penalty": 0.1
    }

    try:
        print("📡 Appel API DeepSeek pour exercice...")
        response = requests.post(api_url, headers=headers, json=data, timeout=90)
        response_data = response.json()

        if response.status_code != 200:
            error_msg = f"Erreur API: {response_data.get('message', 'Pas de détail')}"
            print(f"❌ {error_msg}")
            return error_msg, None

        # 1) On récupère et loggue la réponse brute de l'IA
        output = response_data['choices'][0]['message']['content']
        print("✅ Réponse IA brute (début):")
        print(output[:500].replace("\n", "\\n"))
        print("… (total", len(output), "caractères)\n")
        # 2) Fusion des blocs LaTeX multi-lignes (\[ … \]) en une seule ligne
        output = flatten_multiline_latex_blocks(output)
        print("🛠️ Après flatten_multiline_latex_blocks (début):")
        print(output[:500].replace("\n", "\\n"))
        print("… (total", len(output), "caractères)\n")
        # Nettoyage/structuration dès la réception IA
        output_structured = format_corrige_pdf_structure(output)
        print("🧩 output_structured après format_corrige_pdf_structure:")
        print(output_structured[:500].replace("\n", "\\n"), "\n…\n")

        # Initialisation des variables de retour
        corrige_txt = output_structured
        graph_list = []

        # Extraction graphique: regex robuste !
        json_blocks = extract_json_blocks(output_structured)
        print(f"🔍 JSON blocks détectés : {len(json_blocks)}")

        # 2) Afficher chaque JSON brut et son dict Python
        for i, (graph_dict, start, end) in enumerate(json_blocks, start=1):
            raw_json = output_structured[start:end]
            print(f"   ▶️ Bloc JSON {i} brut:")
            print(raw_json.replace("\n", "\\n"))
            print("   ▶️ Parsed Python dict :", graph_dict)

        # 3) Pour éviter tout décalage, on traite du plus loin au plus près
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
                # remplacement sans offset, indices toujours valables
                corrige_txt = corrige_txt[:start] + img_tag + corrige_txt[end:]
                graph_list.append(graph_dict)
                print(f"✅ Graphique {idx} inséré")
            except Exception as e:
                print(f"❌ Erreur génération graphique {idx}: {e}")
                continue

        # 4) Afficher un extrait du corrigé HTML final
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


def extraire_texte_image(fichier_path):
    """
    OCR amélioré avec pré-traitement PIL uniquement (sans OpenCV)
    """
    try:
        image = Image.open(fichier_path)

        # === PRÉ-TRAITEMENT AVEC PIL SEULEMENT ===

        # 1. Conversion en niveaux de gris
        image = image.convert("L")

        # 2. Redimensionnement adaptatif
        scale_factor = 3 if max(image.size) < 1500 else 2
        new_width = image.width * scale_factor
        new_height = image.height * scale_factor
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # 3. Filtres pour améliorer la netteté
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.MedianFilter(size=3))  # Réduction bruit

        # 4. Amélioration du contraste avec PIL
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.5)

        # 5. Amélioration de la luminosité
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)

        # 6. Binarisation avec PIL (alternative à OpenCV)
        # Méthode 1: Seuil adaptatif manuel
        def binarisation_pil(img):
            # Calcul du seuil basé sur l'histogramme
            histogram = img.histogram()
            total_pixels = img.width * img.height
            cumulative = 0
            threshold = 128  # valeur par défaut

            # Trouver un seuil adaptatif (méthode Otsu simplifiée)
            for i, count in enumerate(histogram):
                cumulative += count
                if cumulative > total_pixels * 0.1:  # Seuil à 10%
                    threshold = i
                    break

            return img.point(lambda x: 0 if x < threshold else 255, '1')

        image = binarisation_pil(image)

        # 7. Configuration Tesseract optimisée pour les maths
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]{}<>+-=*/\\|^_€¥£§%°²³±≤≥≈≠∞∫∑∏√∂∆∇¬∧∨∀∃∈∋⊂⊃∪∩∅αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩℕℤℚℝℂ '

        # === ESSAI AVEC DIFFÉRENTS PARAMÈTRES ===

        text_results = []

        # Essai 1: Tesseract standard
        try:
            texte_std = pytesseract.image_to_string(image, lang="fra+eng", config=custom_config)
            text_results.append(("Standard", texte_std))
        except Exception as e:
            print(f"❌ OCR standard échoué: {e}")

        # Essai 2: Avec image inversée
        try:
            inverted = Image.eval(image, lambda x: 255 - x)
            texte_inv = pytesseract.image_to_string(inverted, lang="fra+eng", config=custom_config)
            text_results.append(("Inversé", texte_inv))
        except Exception as e:
            print(f"❌ OCR inversé échoué: {e}")

        # Essai 3: Avec différents modes PSM
        psm_modes = {
            "PSM6": "6",  # Bloc uniforme de texte
            "PSM8": "8",  # Mot unique
            "PSM11": "11"  # Texte dense
        }

        for psm_name, psm_value in psm_modes.items():
            try:
                psm_config = f'--oem 3 --psm {psm_value} {custom_config}'
                texte_psm = pytesseract.image_to_string(image, lang="fra+eng", config=psm_config)
                if texte_psm.strip():
                    text_results.append((psm_name, texte_psm))
            except Exception as e:
                print(f"❌ OCR {psm_name} échoué: {e}")

        # === SÉLECTION DU MEILLEUR RÉSULTAT ===
        meilleur_texte = ""
        meilleur_score = 0

        for nom, texte in text_results:
            if texte and len(texte.strip()) > 10:  # Ignorer les textes trop courts
                # Score basé sur la longueur et la présence de mots-clés mathématiques
                score = len(texte.strip())

                # Bonus pour les mots-clés mathématiques
                mots_cles_maths = ['lim', 'cos', 'sin', 'tan', 'exp', 'ln', 'log', '∫', '∑', '∞', '∈', '∀', '∃', 'frac']
                for mot in mots_cles_maths:
                    if mot.lower() in texte.lower():
                        score += 15

                # Bonus pour les structures LaTeX
                if '\\' in texte or '^' in texte or '_' in texte:
                    score += 25

                # Malus pour les caractères improbables
                if '$$$' in texte or '@@@' in texte:
                    score -= 50

                print(f"📊 Score OCR {nom}: {score} - Texte: {texte[:80].replace(chr(10), ' ')}...")

                if score > meilleur_score:
                    meilleur_score = score
                    meilleur_texte = texte

        # Nettoyage du texte final
        if meilleur_texte:
            # Correction des erreurs OCR courantes en maths
            corrections = {
                'reos': 'cos', 'c0s': 'cos', 's1n': 'sin', 't an': 'tan',
                'l1m': 'lim', 'ln1': 'lim', '€': '∈', '¥': '∞',
                '--': '→', '++': '∞', 'I1': 'll', 'O': '0', '|': 'l',
                ']1': '[1', ']0': '[0', 'coo': '∞', 'ooo': '∞',
                ']a': '[a', ']b': '[b', ']x': '[x', ']y': '[y'
            }

            for erreur, correction in corrections.items():
                meilleur_texte = meilleur_texte.replace(erreur, correction)

            print(f"🖨️ DEBUG – OCR image améliorée : {len(meilleur_texte)} caractères")
            print(f"📝 Extrait OCR final : {meilleur_texte[:200].replace(chr(10), ' ')}")
            return meilleur_texte.strip()
        else:
            print("❌ Aucun résultat OCR valide")
            # Fallback sur l'ancienne méthode
            return pytesseract.image_to_string(image, lang="fra+eng").strip()

    except Exception as e:
        print(f"❌ Erreur OCR image (améliorée) : {e}")
        # Fallback ultime
        try:
            return pytesseract.image_to_string(Image.open(fichier_path), lang="fra+eng").strip()
        except:
            return ""


# ============== EXTRACTION TEXTE/FICHIER (PDF & IMAGE) ==============
def extraire_texte_fichier(fichier_field):
    """
    - Si PDF    : extraction via pdfminer.
    - Si image  : OCR (pytesseract) + description (BLIP).
    - Sinon     : fallback sur OCR + BLIP.
    """
    if not fichier_field:
        return ""

    temp_dir  = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))

    # 1) Sauvegarde du fichier
    with open(temp_path, "wb") as f:
        for chunk in fichier_field.chunks():
            f.write(chunk)

    ext = os.path.splitext(fichier_field.name)[1].lower()
    resultat = ""

    if ext == ".pdf":
        # extraction textuelle
        texte = extract_text(temp_path)
        print(f"📄 DEBUG – PDF extrait : {len(texte)} caractères")
        resultat = texte.strip() if texte else ""
    else:
        # on considère tout le reste comme une image
        # a) OCR du texte
        try:
            ocr = extraire_texte_image(temp_path)
            print(f"🖨️ DEBUG – OCR image ({ext}) : {len(ocr)} caractères")
        except Exception as e:
            print(f"❌ Erreur OCR image : {e}")
            ocr = ""

        # b) Description visuelle via BLIP
        try:
            caption = decrire_image(temp_path)
            # decrire_image inclut son propre print debug
        except Exception as e:
            print(f"❌ Erreur BLIP captioning : {e}")
            caption = ""

        # c) Assemblage
        morceaux = []
        if ocr.strip():
            morceaux.append("Texte OCR :\n" + ocr.strip())
        if caption.strip():
            morceaux.append(caption.strip())

        resultat = "\n\n".join(morceaux)

    # supprime le temporaire
    try:
        os.remove(temp_path)
    except:
        pass

    if not resultat.strip():
        resultat = "(Impossible d'extraire l'énoncé du fichier envoyé.)"

    print(f"📁 DEBUG – Extraction fichier ({ext}) terminée :")
    print(resultat[:500].replace("\n", "\\n"), "...\n")
    return resultat

# ============== TABLEAUX DE VARIATION (Camelot) ==============

def extraire_tables_pdf(path_pdf: str):
    """
    Détecte et renvoie la liste des tableaux dans le PDF.
    """
    try:
        tables = camelot.read_pdf(path_pdf, pages='all', flavor='stream')
        print(f"==== DEBUG Camelot : {len(tables)} table(s) détectée(s) dans {path_pdf} ====")
        return tables
    except Exception as e:
        print(f"❌ Erreur Camelot.read_pdf sur {path_pdf} : {e}")
        return []

def decrire_table_variation(table):
    """
    Si table.df ressemble à un tableau de variation, renvoie
    une description détaillée (sens de variation, extrema…).
    Sinon, retourne None sans lever d’exception.
    """
    try:
        df = table.df.replace('', np.nan) \
                     .dropna(how='all', axis=1) \
                     .fillna(method='ffill')

        # 1) S’assurer qu’il y a au moins 2 colonnes et 2 lignes (1 en-tête + 1 donnée)
        if df.shape[1] < 2 or df.shape[0] < 2:
            return None

        # 2) Extraction des données (on saute la 1ʳᵉ ligne d’en-tête)
        data = df.iloc[1:].reset_index(drop=True)
        intervalles = data.iloc[:, 0].astype(str).tolist()

        # 3) Conversion sécurisée des valeurs f(x)
        valeurs = []
        for val in data.iloc[:, 1]:
            try:
                valeurs.append(float(str(val).replace(',', '.')))
            except:
                valeurs.append(None)

        # 4) Construction des descriptions de variation
        descs = []
        for i in range(len(valeurs) - 1):
            v1, v2 = valeurs[i], valeurs[i+1]
            a, b = intervalles[i], intervalles[i+1]
            if v1 is None or v2 is None:
                continue
            if v2 > v1:
                descs.append(f"f croissante de {a} à {b}")
            elif v2 < v1:
                descs.append(f"f décroissante de {a} à {b}")
            else:
                descs.append(f"f constante de {a} à {b}")

        # 5) Recherche d’extrema
        extrema = []
        for i in range(1, len(valeurs) - 1):
            v0, v1, v2 = valeurs[i-1], valeurs[i], valeurs[i+1]
            x = intervalles[i]
            if v1 is None or v0 is None or v2 is None:
                continue
            if v1 > v0 and v1 > v2:
                extrema.append(f"maximum en {x} = {v1}")
            elif v1 < v0 and v1 < v2:
                extrema.append(f"minimum en {x} = {v1}")

        # 6) Composition du texte final
        parts = []
        if descs:
            parts.append("Tableau de variation : " + "; ".join(descs) + ".")
        if extrema:
            parts.append("Extrema : " + "; ".join(extrema) + ".")
        return " ".join(parts) if parts else None

    except Exception as e:
        print(f"❌ Erreur decrire_table_variation: {e}")
        return None


def decrire_image(path_image: str) -> str:
    """
    Génère une légende / description de l'image via BLIP.
    """
    try:
        print(f"🖼️ DEBUG – Captioning image : {path_image}")
        img = Image.open(path_image).convert("RGB")
        inputs = _processor(img, return_tensors="pt").to(device)
        # Génération en une passe
        out = _model.generate(**inputs, max_new_tokens=50)
        caption = _processor.decode(out[0], skip_special_tokens=True)
        caption = caption.strip()
        print(f"🖼️ DEBUG – Légende générée : {caption}")
        return "Description image : " + caption
    except Exception as e:
        print(f"❌ Erreur decrire_image pour {path_image} : {e}")
        return "(Erreur description image)"

# ============== NETTOYAGE / REFORMULATION AVEC GPT-3.5 ==============
def nettoyer_pour_deepseek(concat_text: str) -> str:
    """
    Reformule le texte brut + descriptions pour qu'il soit clair et complet
    avant envoi à DeepSeek (GPT-3.5).
    """
    print("🧹 DEBUG – DÉBUT nettoyage GPT-3.5")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    prompt = (
        "Tu es un assistant chargé de reformuler un énoncé scientifique "
        "pour qu'il soit clair et complet pour DeepSeek. Corrige les "
        "imprécisions et structure en paragraphes.\n\n"
        f"{concat_text}"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3000
        )
        cleaned = resp.choices[0].message.content.strip()
        print("🧹 DEBUG – Texte nettoyé (début) :")
        print(cleaned[:500].replace("\n", "\\n"), "...\n")
        return cleaned

    except Exception as e:
        print(f"❌ Erreur nettoyage GPT-3.5: {e}")
        # fallback : on renvoie le texte d’origine
        return concat_text

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
# PROMPT PAR DEFAUT TRES DIRECTIF + EXEMPLES
DEFAULT_SYSTEM_PROMPT = r"""Tu es un professeur expert en sciences (Maths, Physique, SVT, Chimie, Statistique).

Règles :
- Dès qu'un exercice demande un graphique, tu termines la réponse concernée par la balise ---corrigé--- sur une ligne, puis sur la ligne suivante, le JSON du graphique : {"graphique": {...}}

Types supportés : "fonction", "histogramme", "diagramme à bandes", "nuage de points", "effectifs cumulés", "diagramme circulaire"/"camembert", "polygone", "cercle trigo".

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

def generer_corrige_direct(texte_enonce, contexte, lecons_contenus, exemples_corriges, matiere):
    """
    Traitement direct pour les épreuves courtes (un seul exercice).
    Appelle la fonction par exercice pour centraliser l'extraction graphique.
    """
    print("🎯 Traitement DIRECT (épreuve courte)")
    return generer_corrige_par_exercice(texte_enonce, contexte, matiere)


def generer_corrige_decoupe(texte_epreuve, contexte, matiere):
    """
    Traitement par découpage pour les épreuves longues
    """
    print("🎯 Traitement AVEC DÉCOUPAGE (épreuve longue)")

    exercices = separer_exercices(texte_epreuve)
    tous_corriges = []
    tous_graphiques = []

    for i, exercice in enumerate(exercices, 1):
        print(f"📝 Traitement exercice {i}/{len(exercices)}...")

        corrige, graphiques = generer_corrige_par_exercice(exercice, contexte, matiere)

        if corrige and "Erreur" not in corrige:
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
                                    demande=None):
    """
    Nouvelle version avec découpage intelligent des épreuves longues
    """
    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    print("\n" + "=" * 60)
    print("🚀 DÉBUT TRAITEMENT INTELLIGENT")
    print("=" * 60)
    print(f"📏 Longueur texte: {len(texte_enonce)} caractères")

    # 1. ESTIMER LA COMPLEXITÉ
    tokens_estimes = estimer_tokens(texte_enonce)

    # 2. DÉCISION : TRAITEMENT DIRECT OU DÉCOUPÉ
    if tokens_estimes < 1000:  # Épreuve courte
        print("🎯 Décision: TRAITEMENT DIRECT (épreuve courte)")
        return generer_corrige_direct(texte_enonce, contexte, lecons_contenus, exemples_corriges, matiere)
    else:  # Épreuve longue
        print("🎯 Décision: DÉCOUPAGE (épreuve longue)")
        return generer_corrige_decoupe(texte_enonce, contexte, matiere)


# ============== TÂCHE ASYNCHRONE ==============

@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(demande_id, matiere_id=None):
    from correction.models import DemandeCorrection, SoumissionIA
    from resources.models import Matiere

    try:
        demande = DemandeCorrection.objects.get(id=demande_id)
        soumission = SoumissionIA.objects.get(demande=demande)

        soumission.statut = 'extraction'
        soumission.progression = 20
        soumission.save()

        # 1) Extraction initiale
        texte_brut = ""
        if demande.fichier:
            texte_brut = extraire_texte_fichier(demande.fichier)
        else:
            texte_brut = demande.enonce_texte or ""

        print("📥 DEBUG – TEXTE BRUT (premiers 500 chars) :")
        print(texte_brut[:500].replace("\n", "\\n"), "...\n")

        # 2) Extraction & description des tableaux
        descs_tables = []
        if demande.fichier:
            path_pdf = demande.fichier.path
            tables = extraire_tables_pdf(path_pdf)
            for idx, table in enumerate(tables, start=1):
                desc = decrire_table_variation(table)
                if desc:
                    descs_tables.append(desc)
                    print(f"📋 DEBUG – Description table {idx} : {desc}")

        print(f"🔍 DEBUG – Total descriptions tables : {len(descs_tables)}")

        # 3) Assemblage du texte final pour l'IA
        texte_enonce = texte_brut
        if descs_tables:
            texte_enonce += "\n\n" + "\n".join(descs_tables)

        print("📥 DEBUG – TEXTE ENRICHI (après tables) :")
        print(texte_enonce[:500].replace("\n", "\\n"), "...\n")

        soumission.statut = 'analyse_ia'
        soumission.progression = 40
        soumission.save()

        matiere = Matiere.objects.get(id=matiere_id) if matiere_id else demande.matiere
        contexte = f"Exercice de {matiere.nom} - {demande.classe.nom if demande.classe else ''}"

        soumission.statut = 'generation_graphiques'
        soumission.progression = 60
        soumission.save()

        # 3.b) Nettoyage / reformulation avant DeepSeek
        texte_pret = nettoyer_pour_deepseek(texte_enonce)
        print("🧹 DEBUG – TEXTE PRÊT pour DeepSeek (premiers 500 chars) :")
        print(texte_pret[:500].replace("\n", "\\n"), "...\n")

        corrige_txt, graph_list = generer_corrige_ia_et_graphique(
            texte_pret,
            contexte,
            matiere=matiere
        )

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

        soumission.statut = 'termine'
        soumission.progression = 100
        soumission.resultat_json = {
            'corrige_text': corrige_txt,
            'pdf_url': pdf_path,
            'graphiques': graph_list or []
        }
        soumission.save()

        demande.corrigé = corrige_txt
        demande.save()

        print("🎉 TRAITEMENT TERMINÉ AVEC SUCCÈS!")
        return True

    except Exception as e:
        print(f"❌ ERREUR dans la tâche IA: {e}")
        try:
            soumission.statut = 'erreur'
            soumission.save()
        except:
            pass
        return False

