import requests
import os
import tempfile
import json
import re
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from django.conf import settings
from django.utils.safestring import mark_safe
from celery import shared_task
import base64

# ============== CONFIGURATION DES APIS ==============

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# ============== EXTRACTION AVEC GPT-3.5 TURBO VISION ==============

def extraire_avec_gpt35_vision(image_path, contexte, type_exercice=None):
    """
    Utiliser GPT-3.5 Turbo Vision pour l'extraction intelligente
    """
    if not OPENAI_API_KEY:
        print("❌ Clé OpenAI non configurée - Fallback sur OCR standard")
        return extraire_texte_image_optimise(image_path)

    try:
        # Encoder l'image en base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Adapter le prompt selon le type d'exercice
        prompt_type = ""
        if type_exercice:
            prompt_type = f"\nTYPE D'EXERCICE : {type_exercice.nom} - Adaptez l'extraction en conséquence."

        prompt_extraction = f"""
        CONTEXTE : {contexte}
        {prompt_type}

        🎯 VOTRE MISSION : Extraire et décrire complètement cet exercice/scanne.

        ### 1. 📝 EXTRACTION TEXTUELLE COMPLÈTE :
        - Copiez TOUT le texte visible, mot pour mot
        - Conservez la structure originale (titres, questions, numérotation)
        - Gardez les formules mathématiques et notations scientifiques
        - Ne modifiez pas l'ordre ou la hiérarchie

        ### 2. 🖼️ DESCRIPTION DES ÉLÉMENTS VISUELS :
        - Schémas, diagrammes, graphiques
        - Formes géométriques et leurs dimensions
        - Courbes, axes, points remarquables
        - Légendes, annotations, flèches
        - Tableaux et leurs structures

        ### 3. 🎓 CONTEXTE PÉDAGOGIQUE :
        - Difficulté perçue de l'exercice
        - Thèmes ou concepts abordés
        - Type de raisonnement requis

        ### 📋 FORMAT DE RÉPONSE STRICTE :

        [TEXTE COMPLET]
        [Le texte intégral de l'exercice ici...]

        [ÉLÉMENTS VISUELS]
        [Description détaillée des éléments graphiques...]

        [CONTEXTE PÉDAGOGIQUE]
        [Analyse du type d'exercice et difficulté...]
        """

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_extraction},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 3000,
                "temperature": 0.1
            },
            timeout=30
        )

        if response.status_code == 200:
            resultat = response.json()['choices'][0]['message']['content']
            print(f"✅ GPT-3.5 Vision: Extraction réussie ({len(resultat)} caractères)")
            return resultat
        else:
            print(f"❌ Erreur GPT-3.5: {response.status_code}")
            return extraire_texte_image_optimise(image_path)

    except Exception as e:
        print(f"❌ Erreur GPT-3.5 Vision: {e}")
        return extraire_texte_image_optimise(image_path)


# ============== FALLBACK OCR (SI GPT ÉCHoue) ==============

def extraire_avec_ocrspace(image_path):
    """
    OCR.space API - 25 000 requêtes/mois gratuites
    """
    try:
        import requests
        import base64

        with open(image_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode()

        payload = {
            'base64Image': f'data:image/jpeg;base64,{img_base64}',
            'language': 'fre',
            'isOverlayRequired': False,
            'OCREngine': 2
        }

        response = requests.post(
            'https://api.ocr.space/parse/image',
            data=payload,
            headers={'apikey': 'helloworld'}  # Clé gratuite
        )

        if response.status_code == 200:
            result = response.json()
            if not result['IsErroredOnProcessing']:
                texte = result['ParsedResults'][0]['ParsedText']
                print(f"✅ OCR.space: {len(texte)} caractères")
                return texte
        return ""
    except Exception as e:
        print(f"❌ OCR.space error: {e}")
        return ""


def extraire_texte_tesseract_ameliore(image_path):
    """
    Tesseract amélioré avec pré-traitement
    """
    try:
        image = Image.open(image_path)
        image = image.convert('L')
        image = image.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        texte = pytesseract.image_to_string(image, lang='fra+eng')
        print(f"✅ Tesseract amélioré: {len(texte)} caractères")
        return texte.strip()
    except Exception as e:
        print(f"❌ Tesseract error: {e}")
        return ""


def extraire_texte_image_optimise(image_path):
    """
    Pipeline de fallback gratuit
    """
    strategies = [
        ("ocrspace", extraire_avec_ocrspace),
        ("tesseract_ameliore", extraire_texte_tesseract_ameliore),
    ]

    for nom_strategie, fonction_ocr in strategies:
        try:
            texte = fonction_ocr(image_path)
            if texte and len(texte.strip()) > 20:
                print(f"✅ Fallback {nom_strategie} réussi")
                return texte
        except Exception as e:
            print(f"❌ {nom_strategie} a échoué: {e}")
            continue

    return "(Extraction texte limitée - éléments visuels non décrits)"


# ============== CORRECTION AVEC DEEPSEEK ==============

def generer_corrige_avec_deepseek(texte_exercice, contexte, matiere=None, type_exercice=None, lecons_contenus=None):
    """
    Nouvelle fonction utilisant DeepSeek pour la correction uniquement
    """
    if not DEEPSEEK_API_KEY:
        return "Erreur: Clé DeepSeek non configurée", None

    # Préparer le contexte enrichi
    contexte_enrichi = preparer_contexte_correction(contexte, matiere, type_exercice, lecons_contenus)

    # CORRECTION : Formule LaTeX corrigée (échappement des backslashes)
    prompt_correction = f"""
    {contexte_enrichi}

    ### 📝 EXERCICE À CORRIGER (extrait par vision IA) :
    {texte_exercice}

    ### 🎯 CONSIGNES DE CORRECTION :
    1. **Corrigez complètement** l'exercice en expliquant chaque étape
    2. **Adaptez votre approche** au type d'exercice et niveau
    3. **Utilisez LaTeX** pour toutes les formules mathématiques : \\(...\\) pour inline et \\[...\\] pour display
    4. **Incluez des graphiques** si nécessaire avec le format JSON standard
    5. **Structurez clairement** avec titres et sous-titres
    6. **Soyez pédagogique** mais concis

    ### 📋 FORMAT DE RÉPONSE :
    # Correction de l'exercice

    ## 1. [Première question/partie]
    [Correction détaillée...]
    [Formules: \\(x = \\frac{{-b \\pm \\sqrt{{b^2-4ac}}}}{{2a}}\\)]

    ## 2. [Deuxième question/partie]
    [Correction détaillée...]

    [Graphiques si nécessaire...]
    """

    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": contexte_enrichi},
            {"role": "user", "content": prompt_correction}
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.9,
        "frequency_penalty": 0.1
    }

    try:
        print("📡 Appel DeepSeek pour correction...")
        response = requests.post(api_url, headers=headers, json=data, timeout=90)
        response_data = response.json()

        if response.status_code != 200:
            error_msg = f"Erreur DeepSeek: {response_data.get('message', 'Pas de détail')}"
            print(f"❌ {error_msg}")
            return error_msg, None

        output = response_data['choices'][0]['message']['content']
        print(f"✅ DeepSeek: Correction générée ({len(output)} caractères)")

        # Traitement des graphiques
        corrige_final, graphiques = extract_and_process_graphs(output)
        return corrige_final, graphiques

    except Exception as e:
        error_msg = f"Erreur API DeepSeek: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, None


def preparer_contexte_correction(contexte, matiere, type_exercice, lecons_contenus):
    """
    Préparer le contexte enrichi pour DeepSeek avec tous les paramètres
    """
    contexte_base = contexte

    # Ajouter le type d'exercice
    if type_exercice:
        contexte_base += f"\nType d'exercice: {type_exercice.nom}"

    # Ajouter les prompts spécifiques de la matière
    if matiere and hasattr(matiere, 'prompt_ia'):
        promptia = matiere.prompt_ia
        system_prompt = promptia.system_prompt or DEFAULT_SYSTEM_PROMPT
        consignes_finales = promptia.consignes_finales or "Format de réponse strict : LaTeX pour les maths, explications détaillées mais concises"
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        consignes_finales = "Format de réponse strict : LaTeX pour les maths, explications détaillées mais concises"

    # Ajouter les leçons si disponibles
    lecons_text = ""
    if lecons_contenus:
        lecons_text = "\n### LEÇONS CONNEXES :\n"
        for titre, contenu in lecons_contenus[:2]:  # Limiter à 2 leçons max
            lecons_text += f"**{titre}** : {contenu[:200]}...\n"

    contexte_final = f"""
    {system_prompt}

    ### CONTEXTE :
    {contexte_base}
    {lecons_text}

    ### CONSIGNES FINALES :
    {consignes_finales}
    """

    return contexte_final


# ============== FONCTION PRINCIPALE HYBRIDE ==============

def generer_corrige_hybride(texte_enonce, contexte, lecons_contenus=None, exemples_corriges=None,
                            matiere=None, type_exercice=None, demande=None):
    """
    NOUVELLE FONCTION PRINCIPALE : GPT-3.5 (extraction) + DeepSeek (correction)
    """
    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    print("\n" + "=" * 60)
    print("🚀 DÉBUT TRAITEMENT HYBRIDE GPT-3.5 + DEEPSEEK")
    print("=" * 60)

    # Si c'est un fichier image, utiliser GPT-3.5 Vision pour l'extraction
    if demande and demande.fichier and demande.fichier.name.lower().endswith(('.jpg', '.jpeg', '.png')):
        print("🖼️ Fichier image détecté - Extraction avec GPT-3.5 Vision")
        temp_path = sauvegarder_fichier_temporaire(demande.fichier)
        texte_enonce = extraire_avec_gpt35_vision(temp_path, contexte, type_exercice)

        # Nettoyer le fichier temporaire
        try:
            os.remove(temp_path)
        except:
            pass

    # Utiliser DeepSeek pour la correction
    print("🎓 Correction avec DeepSeek...")
    corrige_txt, graph_list = generer_corrige_avec_deepseek(
        texte_enonce, contexte, matiere, type_exercice, lecons_contenus
    )

    print(f"✅ Traitement hybride terminé: {len(corrige_txt)} caractères, {len(graph_list or [])} graphiques")
    return corrige_txt, graph_list


# ============== FONCTIONS EXISTANTES (MAINTENUES) ==============

def separer_exercices(texte_epreuve):
    """Détecte et sépare automatiquement les exercices"""
    if not texte_epreuve:
        return []

    print("🔍 Détection des exercices...")
    patterns_separation = [
        r'Exercice\s+\d+[:.]',
        r'EXERCICE\s+\d+[:.]',
        r'Partie\s+[IVXLCDM]+[:.]',
        r'\n\d+[-.)]\s',
        r'\n[a-z]\)\s',
        r'Question\s+\d+',
    ]

    exercices = []
    lignes = texte_epreuve.split('\n')
    exercice_courant = []
    dans_exercice = False

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue

        nouvel_exercice = False
        for pattern in patterns_separation:
            if re.search(pattern, ligne, re.IGNORECASE):
                nouvel_exercice = True
                break

        if nouvel_exercice and exercice_courant:
            exercices.append('\n'.join(exercice_courant))
            exercice_courant = []
            dans_exercice = True

        exercice_courant.append(ligne)

    if exercice_courant:
        exercices.append('\n'.join(exercice_courant))

    if not exercices:
        exercices = [texte_epreuve]

    print(f"✅ {len(exercices)} exercice(s) détecté(s)")
    return exercices


def estimer_tokens(texte):
    """Estimation simple du nombre de tokens"""
    mots = len(texte.split())
    tokens = int(mots / 0.75)
    print(f"📊 Estimation tokens: {mots} mots → {tokens} tokens")
    return tokens


def extract_and_process_graphs(output):
    """Extrait et traite les graphiques d'un corrigé"""
    print("🖼️ Extraction des graphiques...")
    graphs_data = []
    final_text = output

    pattern = r'---corrigé---\s*\n*\s*(\{[\s\S]*?\})(?=\s*$|\s*---|\s*\n\s*\w)'
    matches = re.finditer(pattern, output)

    for match_idx, match in enumerate(matches):
        json_str = match.group(1).strip()
        print(f"📦 JSON brut {match_idx + 1}: {json_str[:100]}...")

        try:
            json_str = re.sub(r"'", '"', json_str)
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            json_str = re.sub(r'\s+', ' ', json_str)

            graph_data = json.loads(json_str)
            graphs_data.append(graph_data)

            output_name = f"graphique_{match_idx + 1}_{np.random.randint(1000)}.png"
            print(f"🎨 Génération graphique {match_idx + 1}...")
            img_path = tracer_graphique(graph_data, output_name)

            if img_path:
                img_tag = f'<div class="graphique-container"><img src="/media/{img_path}" alt="Graphique {match_idx + 1}" style="max-width:100%;margin:10px 0;" />'
                final_text = final_text.replace(match.group(0), img_tag)
                print(f"✅ Graphique {match_idx + 1} généré: {img_path}")
            else:
                final_text = final_text.replace(match.group(0),
                                                '<div class="graphique-error">Erreur génération graphique</div>')

        except Exception as e:
            print(f"❌ Erreur parsing JSON graphique {match_idx + 1}: {e}")
            final_text = final_text.replace(match.group(0), f'<div class="graph-error">Erreur: {str(e)}</div>')

    print(f"🎯 Extraction terminée: {len(graphs_data)} graphique(s) traité(s)")
    return final_text, graphs_data


def flatten_multiline_latex_blocks(text):
    if not text:
        return ""

    def block_replacer(match):
        contents = match.group(1).replace('\n', ' ').replace('\r', ' ')
        contents = re.sub(r' {2,}', ' ', contents)
        return r'\[' + contents.strip() + r'\]'

    def inline_replacer(match):
        contents = match.group(1).replace('\n', ' ').replace('\r', ' ')
        contents = re.sub(r' {2,}', ' ', contents)
        return r'\(' + contents.strip() + r'\)'

    text = re.sub(r'\\\[\s*([\s\S]?)\s\\\]', block_replacer, text)
    text = re.sub(r'\\\(\s*([\s\S]?)\s\\\)', inline_replacer, text)

    return text


def detect_and_format_math_expressions(text):
    if not text:
        return ""

    text = re.sub(
        r'\$\$\s*([\s\S]+?)\s*\$\$',
        lambda m: r'\[' + m.group(1).replace('\n', ' ').strip() + r'\]',
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r'\$\s*([^$]+?)\s*\$',
        lambda m: r'\(' + m.group(1).replace('\n', ' ').strip() + r'\)',
        text
    )

    text = re.sub(
        r'(?<!\\)\[\s*([\s\S]+?)\s*\]',
        lambda m: r'\[' + ' '.join(m.group(1).splitlines()).strip() + r'\]',
        text
    )

    text = flatten_multiline_latex_blocks(text)
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
    if not corrige_text:
        return ""

    formatted = detect_and_format_math_expressions(corrige_text)
    lines = formatted.strip().split('\n')
    html_output = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
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

        if re.search(r'\\\[.?\\\]', line):
            line = re.sub(r'\\\[(\s)(.?)(\s)\\\]', r'\[\2\]', line)
            html_output.append(f'<p>{line}</p>')
            i += 1
        elif re.match(r'^\d+\.', line):
            html_output.append(f'<h2>{line}</h2>')
            i += 1
        elif re.match(r'^[a-z]\)', line):
            html_output.append(f'<p><strong>{line}</strong></p>')
            i += 1
        elif line.startswith('•') or line.startswith('-') or line.startswith('•'):
            html_output.append(f'<p>{line}</p>')
            i += 1
        elif '\\(' in line or '\\[' in line:
            line = re.sub(r'\\\(\s*([^)]?)\s\\\)', r'\\(\1\\)', line)
            line = re.sub(r'\\\[\s*([^]]?)\s\\\]', r'\[\1\]', line)
            html_output.append(f'<p>{line}</p>')
            i += 1
        else:
            html_output.append(f'<p>{line}</p>')
            i += 1

    return mark_safe("".join(html_output))


def extraire_texte_pdf(fichier_path):
    try:
        texte = extract_text(fichier_path)
        print(f"📄 PDF extrait: {len(texte)} caractères")
        return texte.strip() if texte else ""
    except Exception as e:
        print(f"❌ Erreur extraction PDF: {e}")
        return ""


def extraire_texte_image(fichier_path):
    try:
        image = Image.open(fichier_path)
        image = image.convert("L").filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.2)
        image = image.point(lambda x: 0 if x < 150 else 255, '1')
        texte = pytesseract.image_to_string(image, lang="fra+eng")
        print(f"🖼️ Image extraite: {len(texte)} caractères")
        return texte.strip()
    except Exception as e:
        print(f"❌ Erreur extraction image: {e}")
        return ""


def extraire_texte_fichier(fichier_field):
    if not fichier_field:
        return ""

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))

    with open(temp_path, "wb") as f:
        for chunk in fichier_field.chunks():
            f.write(chunk)

    ext = os.path.splitext(fichier_field.name)[1].lower()
    texte = ""

    if ext == ".pdf":
        texte = extraire_texte_pdf(temp_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        # MAINTENANT UTILISE GPT-3.5 VISION EN PREMIER
        texte = extraire_avec_gpt35_vision(temp_path, "Extraction d'exercice")

    try:
        os.remove(temp_path)
    except Exception:
        pass

    resultat = texte if texte.strip() else "(Impossible d'extraire l'énoncé du fichier envoyé.)"
    print(f"📁 Extraction fichier terminée: {len(resultat)} caractères")
    return resultat


def tracer_graphique(graphique_dict, output_name):
    if 'graphique' in graphique_dict:
        graphique_dict = graphique_dict['graphique']

    gtype = graphique_dict.get("type", "fonction").lower().strip()
    titre = graphique_dict.get("titre", "Graphique généré")

    def safe_float(expr):
        try:
            return float(eval(str(expr), {"__builtins__": None, "pi": np.pi, "np": np, "sqrt": np.sqrt}))
        except Exception:
            try:
                return float(expr)
            except Exception:
                return None

    try:
        dossier = os.path.join(settings.MEDIA_ROOT, "graphes")
        os.makedirs(dossier, exist_ok=True)
        chemin_png = os.path.join(dossier, output_name)

        print(f"🎨 Traçage graphique type: {gtype}")

        if "fonction" in gtype:
            x_min = safe_float(graphique_dict.get("x_min", -2)) or -2
            x_max = safe_float(graphique_dict.get("x_max", 4)) or 4
            expression = graphique_dict.get("expression", "x")
            x = np.linspace(x_min, x_max, 400)
            expr_patch = expression.replace('^', '*')

            for func in ["sin", "cos", "tan", "exp", "log", "log10", "arcsin", "arccos", "arctan", "sinh", "cosh",
                         "tanh", "sqrt", "abs"]:
                expr_patch = re.sub(r'(?<![\w.])' + func + r'\s\(', f'np.{func}(', expr_patch)

            expr_patch = expr_patch.replace('ln(', 'np.log(')

            try:
                y = eval(expr_patch, {'x': x, 'np': np, '__builtins__': None, "pi": np.pi})
                if np.isscalar(y):
                    y = np.full_like(x, y)
            except Exception as e:
                print(f"❌ Erreur tracé expression: {e}")
                return None

            plt.figure(figsize=(6, 4))
            plt.plot(x, y, color="#008060")
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

        elif "histogramme" in gtype:
            intervalles = graphique_dict.get("intervalles") or graphique_dict.get("classes") or []
            eff = graphique_dict.get("effectifs", [])
            labels = [str(ival) for ival in intervalles]
            x_pos = np.arange(len(labels))
            eff = [float(e) for e in eff]

            plt.figure(figsize=(7, 4.5))
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
                print("❌ Erreur polygone : aucun point")
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
            print(f"❌ Type graphique non supporté : {gtype}")
            return None

        plt.tight_layout()
        plt.savefig(chemin_png)
        plt.close()
        print(f"✅ Graphique sauvegardé: {chemin_png}")
        return "graphes/" + output_name

    except Exception as ee:
        print(f"❌ Erreur générale sauvegarde PNG: {ee}")
        return None


def sauvegarder_fichier_temporaire(fichier_field):
    """Sauvegarde un fichier dans un emplacement temporaire"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))

    with open(temp_path, "wb") as f:
        for chunk in fichier_field.chunks():
            f.write(chunk)

    return temp_path


# ============== PROMPT PAR DEFAUT ==============

DEFAULT_SYSTEM_PROMPT = r"""
Tu es un professeur expert en sciences (Maths, Physique, SVT, Chimie, Statistique).
- **Dès qu'un exercice demande un graphique ou un tracé, finis le paragraphe avec la balise ---corrigé--- sur une ligne, et sur la ligne qui suit, le JSON du graphique au format ci-dessous.**
- **N'utilise que des doubles guillemets dans ton JSON, jamais de simples guillemets.**

---

Types de graphiques supportés :  
- "fonction", "histogramme", "diagramme à bandes", "nuage de points", "effectifs cumulés", "diagramme circulaire"/"camembert", "polygone", "cercle trigo".

---

EXEMPLES OBLIGATOIRES DE JSON :

--- EX 1 : Fonction ---
Corrigé détaillé...
---corrigé---
{"graphique": {
   "type": "fonction",
   "expression": "x**2 - 2*x + 1",
   "x_min": -1,
   "x_max": 3,
   "titre": "Courbe parabole"
}}

--- EX 2 : Cercle trigo (solutions équation trigo sur le cercle) ---
Corrigé explicatif...
---corrigé---
{"graphique": {
   "type": "cercle trigo",
   "angles": ["-pi/4", "pi/4", "7*pi/4", "9*pi/4"],
   "labels": ["S1", "S2", "S3", "S4"],
   "titre": "Solutions trigonométriques"
}}

--- EX 3 : Histogramme ---
Tracé...
---corrigé---
{"graphique": {
   "type": "histogramme",
   "intervalles": ["0-5", "5-10", "10-15"],
   "effectifs": [3, 6, 7],
   "titre": "Histogramme des effectifs"
}}

--- EX 4 : Diagramme à bandes ---
Tracé...
---corrigé---
{"graphique": {
   "type": "diagramme à bandes",
   "categories": ["A", "B", "C"],
   "effectifs": [10, 7, 12],
   "titre": "Comparaison"
}}

--- EX 5 : Nuage de points ---
---corrigé---
{"graphique": {
   "type": "nuage de points",
   "x": [1,2,3,4],
   "y": [2,5,7,3],
   "titre": "Nuage"
}}

--- EX 6 : Polygone des effectifs cumulés croissants (ECC) ---
Corrigé...
---corrigé---
{"graphique": {
   "type": "polygone",
   "points": [[0,0],[5,3],[10,9],[15,16],[20,20]],
   "titre": "Polygone ECC",
   "x_label": "Borne supérieure",
   "y_label": "Effectifs cumulés"
}}

--- EX 7 : Polygone des effectifs cumulés décroissants (ECD) ---
Corrigé...
---corrigé---
{"graphique": {
   "type": "polygone",
   "points": [[0,20],[5,17],[10,11],[15,4],[20,0]],
   "titre": "Polygone ECD",
   "x_label": "Borne supérieure",
   "y_label": "Effectifs cumulés décroissants"
}}

--- EX 8 : Effectifs cumulés sur courbe (autre notation) ---
Corrigé...
---corrigé---
{"graphique": {
   "type": "effectifs cumulés",
   "x": [0,5,10,15,20],
   "y": [3,9,16,20,24],
   "titre": "Courbe ECC classique"
}}

--- EX 9 : Camembert / diagramme circulaire ---
Corrigé...
---corrigé---
{"graphique": {
   "type": "camembert",
   "categories": ["L1", "L2", "L3"],
   "effectifs": [4, 6, 5],
   "titre": "Répartition"
}}

---

Rappels :
- Toujours respecter ces formats JSON. Les exemples ci-dessus sont obligatoires à suivre si l'exercice appelle ce type de tracé.
- N'inclue jamais de simples guillemets ou de commentaires dans le JSON.
- Structure le corrigé strictement: titres, espacements, numérotation, épure.

"""


# ============== ALIAS POUR COMPATIBILITÉ ==============

def generer_corrige_ia_et_graphique(texte_enonce, contexte, lecons_contenus=None, exemples_corriges=None,
                                    matiere=None, demande=None):
    """
    Alias pour compatibilité avec l'ancien code - utilisé par views.py
    """
    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    print("🔁 Utilisation de l'alias de compatibilité")

    # Appeler la nouvelle fonction hybride
    return generer_corrige_hybride(
        texte_enonce=texte_enonce,
        contexte=contexte,
        lecons_contenus=lecons_contenus,
        exemples_corriges=exemples_corriges,
        matiere=matiere,
        type_exercice=None,  # Non disponible dans l'ancienne signature
        demande=demande
    )

# ============== TÂCHE ASYNCHRONE MISE À JOUR ==============

@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(demande_id, matiere_id=None):
    from correction.models import DemandeCorrection, SoumissionIA
    from resources.models import Matiere, TypeExercice

    try:
        demande = DemandeCorrection.objects.get(id=demande_id)
        soumission = SoumissionIA.objects.get(demande=demande)

        soumission.statut = 'extraction'
        soumission.progression = 20
        soumission.save()

        # Récupérer le type d'exercice (NOUVEAU)
        type_exercice = demande.type_exercice

        texte_enonce = ""
        if demande.fichier:
            texte_enonce = extraire_texte_fichier(demande.fichier)
        if not texte_enonce and hasattr(demande, 'enonce_texte'):
            texte_enonce = demande.enonce_texte or ""

        print(f"📥 Texte à traiter: {len(texte_enonce)} caractères")
        print(f"🎯 Type d'exercice: {type_exercice.nom if type_exercice else 'Non spécifié'}")

        soumission.statut = 'analyse_ia'
        soumission.progression = 40
        soumission.save()

        matiere = Matiere.objects.get(id=matiere_id) if matiere_id else demande.matiere
        contexte = f"Exercice de {matiere.nom} - {demande.classe.nom if demande.classe else ''}"

        # Préparer les leçons
        lecons_contenus = []
        if demande.lecons.exists():
            for lecon in demande.lecons.all()[:3]:  # Limiter à 3 leçons
                lecons_contenus.append((lecon.titre, lecon.contenu or ""))

        soumission.statut = 'generation_graphiques'
        soumission.progression = 60
        soumission.save()

        # UTILISER LA NOUVELLE FONCTION HYBRIDE
        corrige_txt, graph_list = generer_corrige_hybride(
            texte_enonce=texte_enonce,
            contexte=contexte,
            lecons_contenus=lecons_contenus,
            matiere=matiere,
            type_exercice=type_exercice,
            demande=demande
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
            'graphiques': graph_list or [],
            'type_exercice': type_exercice.nom if type_exercice else None
        }
        soumission.save()

        demande.corrigé = corrige_txt
        demande.save()

        print("🎉 TRAITEMENT HYBRIDE TERMINÉ AVEC SUCCÈS!")
        return True

    except Exception as e:
        print(f"❌ ERREUR dans la tâche IA: {e}")
        try:
            soumission.statut = 'erreur'
            soumission.save()
        except:
            pass
        return False

