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
    Génère le corrigé pour un seul exercice
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
        "max_tokens": 4000,  # Suffisant pour un exercice individuel
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

        output = response_data['choices'][0]['message']['content']
        print(f"✅ Réponse API reçue: {len(output)} caractères")

        # Traitement des graphiques pour cet exercice
        corrige_final, graphiques = extract_and_process_graphs(output)

        print(f"📊 Exercice traité: {len(graphiques)} graphique(s) généré(s)")
        return corrige_final, graphiques

    except Exception as e:
        error_msg = f"Erreur: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, None


def extract_and_process_graphs(output):
    """
    Extrait et traite les graphiques d'un corrigé
    """
    print("🖼️ Extraction des graphiques...")

    graphs_data = []
    final_text = output

    pattern = r'---corrigé---\s*\n*\s*(\{[\s\S]*?\})(?=\s*$|\s*---|\s*\n\s*\w)'
    matches = re.finditer(pattern, output)

    print(f"🔍 Recherche de JSON graphique: {len(list(matches))} correspondance(s) trouvée(s)")

    # Réinitialiser l'itérateur
    matches = re.finditer(pattern, output)

    for match_idx, match in enumerate(matches):
        json_str = match.group(1).strip()
        print(f"📦 JSON brut {match_idx + 1}: {json_str[:100]}...")

        try:
            # Nettoyage du JSON
            json_str = re.sub(r"'", '"', json_str)
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            json_str = re.sub(r'\s+', ' ', json_str)

            print(f"🧹 JSON nettoyé {match_idx + 1}: {json_str[:100]}...")

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
                print(f"❌ Erreur génération graphique {match_idx + 1}")

        except Exception as e:
            print(f"❌ Erreur parsing JSON graphique {match_idx + 1}: {e}")
            final_text = final_text.replace(match.group(0), f'<div class="graph-error">Erreur: {str(e)}</div>')

    print(f"🎯 Extraction terminée: {len(graphs_data)} graphique(s) traité(s)")
    return final_text, graphs_data


# ============== UTILITAIRES TEXTE / LATEX / TABLEAU ==============

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
    # Convertir les dollars en balises LaTeX pour MathJax
    # inline : $...$ → \(...\)
    text = re.sub(r'\$(.+?)\$', r'\\(\1\\)', text)
    # block : $$...$$ → \[...\]
    text = re.sub(r'\$\$([\s\S]+?)\$\$', r'\\[\1\\]', text)
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
    Amélioration OCR : grossissement, filtre et contraste puis Tesseract.
    """
    try:
        image = Image.open(fichier_path)
        # niveaux de gris
        image = image.convert("L")
        # redimensionnement pour doubler la résolution
        image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
        # filtre médian pour réduire le bruit
        image = image.filter(ImageFilter.MedianFilter())
        # augmenter le contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        # binarisation manuelle (seuil à ajuster si besoin)
        image = image.point(lambda x: 0 if x < 140 else 255, '1')

        texte = pytesseract.image_to_string(image, lang="fra+eng")
        print(f"🖨️ DEBUG – OCR image améliorée : {len(texte)} caractères")
        return texte.strip()
    except Exception as e:
        print(f"❌ Erreur OCR image (améliorée) : {e}")
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
    df = table.df.replace('', np.nan).ffill().dropna(how='all', axis=1)
    # Identifie les colonnes : x vs f(x) ou dérivée f'
    cols = [c.lower() for c in df.iloc[0].tolist()]
    data = df.iloc[1:].reset_index(drop=True)
    # Extrait intervalles et valeurs
    interv = data.iloc[:,0].astype(str).tolist()
    valeurs = data.iloc[:,1].apply(lambda v: float(v.replace(',','.')) if re.match(r'^[0-9]', str(v)) else None).tolist()
    descs = []
    extrema = []
    for i in range(len(valeurs)-1):
        a,b = interv[i], interv[i+1]
        v1,v2 = valeurs[i], valeurs[i+1]
        if v1 is None or v2 is None: continue
        sens = "croissante" if v2>v1 else "décroissante" if v2<v1 else "constante"
        descs.append(f"f {sens} de {a} à {b}")
    # Cherche un extremum (v2<v1>v3 ou v2>v1<v3)
    for i in range(1, len(valeurs)-1):
        if valeurs[i] is not None:
            if valeurs[i]>valeurs[i-1] and valeurs[i]>valeurs[i+1]:
                extrema.append(f"maximum en {interv[i]} = {valeurs[i]}")
            if valeurs[i]<valeurs[i-1] and valeurs[i]<valeurs[i+1]:
                extrema.append(f"minimum en {interv[i]} = {valeurs[i]}")
    texte = ""
    if descs:
        texte += "Tableau de variation : " + "; ".join(descs) + "."
    if extrema:
        texte += " Extrema : " + "; ".join(extrema) + "."
    return texte if texte else None


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
    try:
        # --- Gestion du multigraph ---
        if "multigraph" in graphique_dict:
            print("🎨 Tracé MULTIGRAPHE détecté")
            graf_dir = os.path.join(settings.MEDIA_ROOT, "graphes")
            os.makedirs(graf_dir, exist_ok=True)
            chemin_png = os.path.join(graf_dir, output_name)

            fig, ax = plt.subplots(figsize=(6, 4))
            style_axes(ax, graphique_dict)

            for g in graphique_dict["multigraph"]:
                expr   = g.get("expression", "x")
                xmin   = float(g.get("x_min", -2))
                xmax   = float(g.get("x_max", 4))
                x_vals = np.linspace(xmin, xmax, 400)
                y_vals = eval(expr.replace('^','**'),
                              {"x": x_vals, "np": np, "__builtins__": None})
                style  = g.get("style", "solid")
                label  = g.get("label", expr)
                ax.plot(x_vals, y_vals, linestyle=style, label=label)

            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(chemin_png)
            plt.close()
            print(f"✅ Multigraph sauvegardé: {chemin_png}")
            return "graphes/" + output_name

        # sinon, on continue la logique existante...
        if 'graphique' in graphique_dict:
            graphique_dict = graphique_dict['graphique']

        gtype = graphique_dict.get("type", "fonction").lower().strip()
        titre = graphique_dict.get("titre", "Graphique généré")
        graf_dir = os.path.join(settings.MEDIA_ROOT, "graphes")
        os.makedirs(graf_dir, exist_ok=True)
        chemin_png = os.path.join(graf_dir, output_name)

        print(f"🎨 Traçage graphique type: {gtype}")

        # Fonction utilitaire safe_float...
        def safe_float(expr):
            try:
                return float(eval(str(expr),
                                  {"__builtins__": None,
                                   "pi": np.pi, "np": np, "sqrt": np.sqrt}))
            except:
                try:
                    return float(expr)
                except:
                    return None

        # Branche "fonction"
        if "fonction" in gtype:
            x_min = safe_float(graphique_dict.get("x_min", -2)) or -2
            x_max = safe_float(graphique_dict.get("x_max", 4)) or 4
            expr  = graphique_dict.get("expression", "x").replace('^','**')

            x = np.linspace(x_min, x_max, 400)
            y = eval(expr, {'x': x, 'np': np, '__builtins__': None, "pi": np.pi})
            if np.isscalar(y):
                y = np.full_like(x, y)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y, color="#008060", label=titre)
            style_axes(ax, graphique_dict)
            ax.set_title(titre)
            ax.grid(True)
            ax.legend()

        # Branche "histogramme"
        elif "histogramme" in gtype:
            intervalles = graphique_dict.get("intervalles") or graphique_dict.get("classes") or []
            eff = [float(e) for e in graphique_dict.get("effectifs", [])]
            labels = [str(i) for i in intervalles]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(labels, eff, color="#208060", edgecolor='black', width=0.9)
            style_axes(ax, graphique_dict)
            ax.set_title(titre)
            ax.grid(axis='y')

        # Branche "diagramme à bandes / bâtons"
        elif any(k in gtype for k in ["diagramme à bandes", "bâtons", "batons"]):
            cat = graphique_dict.get("categories", [])
            eff = [float(e) for e in graphique_dict.get("effectifs", [])]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(cat, eff, color="#208060", edgecolor='black', width=0.7)
            style_axes(ax, graphique_dict)
            ax.set_title(titre)
            ax.grid(axis='y')

        # Branche "nuage de points"
        elif "nuage de points" in gtype or "scatter" in gtype:
            x_pts = graphique_dict.get("x", [])
            y_pts = graphique_dict.get("y", [])

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(x_pts, y_pts, color="#006080")
            style_axes(ax, graphique_dict)
            ax.set_title(titre)
            ax.grid(True)

        # Branche "effectifs cumulés"
        elif "effectifs cumulés" in gtype:
            x_pts = graphique_dict.get("x", [])
            y_pts = graphique_dict.get("y", [])

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_pts, y_pts, marker="o", color="#b65d2f")
            style_axes(ax, graphique_dict)
            ax.set_title(titre)
            ax.grid(True)

        # Branche "camembert / pie"
        elif any(k in gtype for k in ["diagramme circulaire", "camembert", "pie"]):
            cat = graphique_dict.get("categories", [])
            eff = graphique_dict.get("effectifs", [])

            fig, ax = plt.subplots(figsize=(5.3, 5.3))
            ax.pie(eff, labels=cat, autopct='%1.1f%%',
                   colors=plt.cm.Paired.colors, startangle=90,
                   wedgeprops={"edgecolor": "k"})
            ax.set_title(titre)

        # Branche "polygone"
        elif "polygone" in gtype or "polygon" in gtype:
            # extraction des points
            pts = graphique_dict.get("points") or []
            if pts:
                x = [float(p[0]) for p in pts]
                y = [float(p[1]) for p in pts]
            else:
                x = [float(xx) for xx in graphique_dict.get("points_x", [])]
                y = [float(yy) for yy in graphique_dict.get("points_y", [])]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(x, y, marker="o", color="#003355")
            style_axes(ax, graphique_dict)
            ax.set_title(titre)
            ax.grid(True)

        # Branche "cercle trigo"
        elif "cercle trigo" in gtype:
            angles = graphique_dict.get("angles", [])
            labels = graphique_dict.get("labels", [])

            fig, ax = plt.subplots(figsize=(5, 5))
            circle = plt.Circle((0,0), 1, fill=False,
                                edgecolor='black', linestyle='--')
            ax.add_artist(circle)
            for i, ang in enumerate(angles):
                try:
                    a = float(eval(ang, {"pi": np.pi}))
                except:
                    a = 0
                x, y = np.cos(a), np.sin(a)
                ax.plot([0, x], [0, y], color='#992020')
                ax.text(1.1*x, 1.1*y,
                        labels[i] if i < len(labels) else f"S{i+1}")
            ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
            ax.axis('off')
            ax.set_title(titre)

        else:
            print(f"❌ Type graphique non supporté : {gtype}")
            return None

        # Sauvegarde des graphiques "classiques"
        plt.tight_layout()
        plt.savefig(chemin_png)
        plt.close()
        print(f"✅ Graphique sauvegardé: {chemin_png}")
        return "graphes/" + output_name

    except Exception as ee:
        print(f"❌ Erreur générale sauvegarde PNG: {ee}")
        return None

# ============== PROMPT PAR DEFAUT ==============

DEFAULT_SYSTEM_PROMPT = r"""
Tu es un professeur expert en sciences (Maths, Physique, SVT, Chimie, Statistique).
- **Dès qu'une question dans un exercie demande un graphique ou un tracé, finis la question avec la balise ---corrigé--- sur une ligne, et sur la ligne qui suit, le JSON du graphique au format ci-dessous.**
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


# ============== FONCTIONS PRINCIPALES AVEC DÉCOUPAGE ==============

def generer_corrige_direct(texte_enonce, contexte, lecons_contenus, exemples_corriges, matiere):
    """
    Traitement direct pour les épreuves courtes (ANCIENNE MÉTHODE)
    """
    print("🎯 Traitement DIRECT (épreuve courte)")

    system_prompt = DEFAULT_SYSTEM_PROMPT
    exemple_prompt = ""
    consignes_finales = "Format de réponse strict : LaTeX pour les maths, Markdown pour les tableaux"

    if matiere and hasattr(matiere, 'prompt_ia'):
        promptia = matiere.prompt_ia
        system_prompt = promptia.system_prompt or system_prompt
        exemple_prompt = promptia.exemple_prompt or exemple_prompt
        consignes_finales = promptia.consignes_finales or consignes_finales

    lecons = [f"### {t}\n{c}" for t, c in lecons_contenus[:3]]
    exemples = exemples_corriges[:2]

    prompt_ia = f"""### CONTEXTE DU COURS
{contexte}

### LEÇONS UTILES
{"".join(lecons) if lecons else 'Aucune leçon supplémentaire'}

### EXEMPLES DE CORRIGÉS
{exemple_prompt if exemple_prompt else ("".join(exemples) if exemples else 'Aucun exemple fourni')}

### EXERCICE À CORRIGER
{texte_enonce.strip()}

### CONSIGNES FINALES
{consignes_finales}
"""

    api_key = os.getenv('DEEPSEEK_API_KEY')
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
        "temperature": 0.12,
        "max_tokens": 6000,
        "top_p": 0.3,
        "frequency_penalty": 0.2
    }

    try:
        print("📡 Appel API DeepSeek (traitement direct)...")
        response = requests.post(api_url, headers=headers, json=data, timeout=120)
        response_data = response.json()

        if response.status_code != 200:
            error_msg = f"Erreur API DeepSeek: {response_data.get('message', 'Pas de détail')}"
            print(f"❌ {error_msg}")
            return error_msg, None

        output = response_data['choices'][0]['message']['content']
        print(f"✅ Réponse API reçue: {len(output)} caractères")

        print("\n" + "=" * 50)
        print("DEBUG: OUTPUT BRUT DE L'IA")
        print("=" * 50)
        print(output)
        print("=" * 50)

        # Traitement des graphiques
        regex_all_json = re.findall(r'---corrigé---\s\n*({[\s\S]+?})', output)
        graph_list = []

        print(f"🔍 JSONs détectés: {len(regex_all_json)}")

        if regex_all_json:
            corrige_txt = output
            for idx, found_json in enumerate(regex_all_json, 1):
                try:
                    sjson = found_json.replace("'", '"').replace('\n', '').replace('\r', '').strip()
                    sjson = re.sub(r'},\s*$', '}', sjson)
                    sjson = re.sub(r',\s*}', '}', sjson)
                    sjson = re.sub(r',\s*\]', ']', sjson)

                    nb_open = sjson.count("{")
                    nb_close = sjson.count("}")
                    if nb_close < nb_open:
                        sjson = sjson + "}" * (nb_open - nb_close)

                    nb_open = sjson.count("[")
                    nb_close = sjson.count("]")
                    if nb_close < nb_open:
                        sjson = sjson + "]" * (nb_open - nb_close)

                    print(f'DEBUG PATCHED sjson {idx}: {sjson[:200]}...')
                    graph_dict = json.loads(sjson)
                    output_name = f"graphique{idx}{int(1000 * np.random.rand())}.png"
                    img_path = tracer_graphique(graph_dict, output_name)

                    if img_path:
                        abs_path = os.path.join(settings.MEDIA_ROOT, img_path)
                        img_tag = f'<img src="file://{abs_path}" alt="Graphique {idx}" style="max-width:100%;margin:10px 0;" />'

                        for tag in [
                            f"---corrigé---\n{found_json}",
                            f"---corrigé---\r\n{found_json}",
                            f"---corrigé--- {found_json}",
                            f"---corrigé---{found_json}",
                            found_json
                        ]:
                            corrige_txt = corrige_txt.replace(tag, img_tag, 1)
                    else:
                        for tag in [
                            f"---corrigé---\n{found_json}",
                            f"---corrigé---\r\n{found_json}",
                            f"---corrigé--- {found_json}",
                            f"---corrigé---{found_json}",
                            found_json
                        ]:
                            corrige_txt = corrige_txt.replace(tag, "[Erreur génération graphique]", 1)

                    graph_list.append(graph_dict)

                except Exception as e:
                    print(f"❌ Erreur parsing JSON graphique {idx}: {e}")
                    continue

            print(f"✅ Traitement direct terminé: {len(graph_list)} graphique(s)")
            return corrige_txt.strip(), graph_list

        print("✅ Traitement direct terminé (sans graphiques)")
        return output.strip(), None

    except Exception as e:
        error_msg = f"Erreur API: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, None


def generer_corrige_decoupe(texte_epreuve, contexte, matiere):
    """
    Traitement par découpage pour les épreuves longues
    """
    print("🎯 Traitement AVEC DÉCOUPAGE (épreuve longue)")

    # 1. SÉPARER LES EXERCICES
    exercices = separer_exercices(texte_epreuve)

    # 2. TRAITER CHAQUE EXERCICE
    tous_corriges = []
    tous_graphiques = []

    for i, exercice in enumerate(exercices, 1):
        print(f"📝 Traitement exercice {i}/{len(exercices)}...")

        # Générer le corrigé pour cet exercice
        corrige, graphiques = generer_corrige_par_exercice(exercice, contexte, matiere)

        if corrige and "Erreur" not in corrige:
            # Ajouter un titre pour cet exercice
            titre_exercice = f"\n\n## 📝 Exercice {i}\n\n"
            tous_corriges.append(titre_exercice + corrige)

            if graphiques:
                tous_graphiques.extend(graphiques)
            print(f"✅ Exercice {i} traité avec succès")
        else:
            print(f"❌ Exercice {i} en erreur: {corrige}")

        # Petite pause pour éviter la surcharge API
        import time
        time.sleep(1)

    # 3. COMBINER TOUS LES CORRIGÉS
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
