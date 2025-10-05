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


# ==============
# UTILITAIRES TEXTE / LATEX / TABLEAU (inchangés)
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
        elif line.startswith('•') or line.startswith('-') or line.startswith('·'):
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


# ==============
# EXTRACTION TEXTE/FICHIER
def extraire_texte_pdf(fichier_path):
    try:
        texte = extract_text(fichier_path)
        return texte.strip() if texte else ""
    except Exception as e:
        print("Erreur extraction PDF:", e)
        return ""


def extraire_texte_image(fichier_path):
    try:
        image = Image.open(fichier_path)
        image = image.convert("L").filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.2)
        image = image.point(lambda x: 0 if x < 150 else 255, '1')
        texte = pytesseract.image_to_string(image, lang="fra+eng")
        return texte.strip()
    except Exception as e:
        print("Erreur extraction image:", e)
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
        texte = extraire_texte_image(temp_path)

    try:
        os.remove(temp_path)
    except Exception:
        pass

    return texte if texte.strip() else "(Impossible d'extraire l'énoncé du fichier envoyé.)"


# ==============
# DESSIN DE GRAPHIQUES
def tracer_graphique(graphique_dict, output_name):
    if 'graphique' in graphique_dict:
        graphique_dict = graphique_dict['graphique']

    gtype = graphique_dict.get("type", "fonction").lower().strip()
    titre = graphique_dict.get("titre", "Graphique généré")

    def safe_float(expr):
        try:
            return float(eval(str(expr), {"__builtins__": None, "pi": np.pi, "np": np, "sqrt": np.sqrt}))
        except Exception as e:
            try:
                return float(expr)
            except Exception:
                return None

    try:
        dossier = os.path.join(settings.MEDIA_ROOT, "graphes")
        os.makedirs(dossier, exist_ok=True)
        chemin_png = os.path.join(dossier, output_name)

        # === MULTI-CURVE SUPPORT ===
        if "multi" in gtype or "curves" in graphique_dict:
            curves = graphique_dict.get("curves", [])
            plt.figure(figsize=(6, 4))
            plt.axhline(y=0, color='#000000', linewidth=1.8)
            plt.axvline(x=0, color='#000000', linewidth=1.8)
            color_cycle = iter(['#008060', '#b65d2f', '#2080C0', '#DB1919', '#003355', '#C08800', '#591e63'])
            for courbe in curves:
                ctype = courbe.get("type", "")
                label = courbe.get("label", None)
                plot_color = courbe.get("color", next(color_cycle))
                style = courbe.get("style", "-")
                # Tracé de fonction
                if ctype == "fonction":
                    x_min = float(courbe.get("x_min", -2))
                    x_max = float(courbe.get("x_max", 4))
                    expression = courbe.get("expression", "x")
                    x = np.linspace(x_min, x_max, 400)
                    expr_patch = expression.replace('^', '**')
                    for func in [
                        "sin", "cos", "tan", "exp", "log", "log10", "arcsin",
                        "arccos", "arctan", "sinh", "cosh", "tanh", "sqrt", "abs"]:
                        expr_patch = re.sub(r'(?<![\w.])'+func+r'\s*\(', f'np.{func}(', expr_patch)
                    expr_patch = expr_patch.replace('ln(', 'np.log(')
                    try:
                        y = eval(expr_patch, {'x': x, 'np': np, '__builtins__': None, "pi": np.pi})
                        if np.isscalar(y): y = np.full_like(x, y)
                    except Exception as e:
                        continue
                    plt.plot(x, y, style, label=label, color=plot_color)
                # Asymptotes
                elif ctype == "asymptote":
                    if courbe.get("sens") == "verticale" and "x" in courbe:
                        plt.axvline(x=float(courbe["x"]), color=plot_color, linestyle='--', linewidth=1.5, label=label)
                    if courbe.get("sens") == "horizontale" and "y" in courbe:
                        plt.axhline(y=float(courbe["y"]), color=plot_color, linestyle='--', linewidth=1.5, label=label)
                # Nuage de points
                elif ctype in ("nuage de points", "points"):
                    x = courbe.get("x", [])
                    y = courbe.get("y", [])
                    plt.scatter(x, y, label=label, color=plot_color)
                # Polygone, ECC/ECD, etc.
                elif ctype in ("polygone", "ecd", "ecc"):
                    points = courbe.get("points")
                    points_x = courbe.get("points_x")
                    points_y = courbe.get("points_y")
                    absc = courbe.get("abscisses")
                    ords = courbe.get("ordonnees")
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
                        continue
                    plt.plot(x, y, marker="o", linestyle=style, color=plot_color, label=label)
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
        # === Cas standard : une seule courbe/fonction ===
        elif "fonction" in gtype:
            x_min = safe_float(graphique_dict.get("x_min", -2)) or -2
            x_max = safe_float(graphique_dict.get("x_max", 4)) or 4
            expression = graphique_dict.get("expression", "x")
            x = np.linspace(x_min, x_max, 400)
            expr_patch = expression.replace('^', '**')
            for func in [
                "sin", "cos", "tan", "exp", "log", "log10", "arcsin",
                "arccos", "arctan", "sinh", "cosh", "tanh", "sqrt", "abs"]:
                expr_patch = re.sub(r'(?<![\w.])'+func+r'\s*\(', f'np.{func}(', expr_patch)
            expr_patch = expr_patch.replace('ln(', 'np.log(')
            try:
                y = eval(expr_patch, {'x': x, 'np': np, '__builtins__': None, "pi": np.pi})
                if np.isscalar(y): y = np.full_like(x, y)
            except Exception as e:
                print(f"Erreur tracé expression: {e}")
                return None
            plt.figure(figsize=(6, 4))
            plt.axhline(y=0, color='#000000', linewidth=1.8)
            plt.axvline(x=0, color='#000000', linewidth=1.8)
            plt.plot(x, y, color="#008060")
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            # Tracé éventuel asymptotes (standard ou en liste)
            asymptotes = graphique_dict.get("asymptotes")
            if asymptotes:
                if isinstance(asymptotes, dict):
                    for xval in asymptotes.get("verticales", []):
                        plt.axvline(x=float(xval), linestyle="--", color="#DB1919")
                    for yval in asymptotes.get("horizontales", []):
                        plt.axhline(y=float(yval), linestyle="--", color="#0070e0")
                elif isinstance(asymptotes, list):
                    for item in asymptotes:
                        if "x=" in item:
                            xval = float(item.split('=')[1])
                            plt.axvline(x=xval, linestyle="--", color="#DB1919")
                        if "y=" in item:
                            yval = float(item.split('=')[1])
                            plt.axhline(y=yval, linestyle="--", color="#0070e0")
        elif "histogramme" in gtype:
            intervalles = graphique_dict.get("intervalles") or graphique_dict.get("classes") or []
            eff = graphique_dict.get("effectifs", [])
            labels = [str(ival) for ival in intervalles]
            x_pos = np.arange(len(labels))
            eff = [float(e) for e in eff]
            plt.figure(figsize=(7, 4.5))
            plt.axhline(y=0, color='#000000', linewidth=1.8)
            plt.axvline(x=0, color='#000000', linewidth=1.8)
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
            plt.axhline(y=0, color='#000000', linewidth=1.8)
            plt.axvline(x=0, color='#000000', linewidth=1.8)
            plt.bar(x_pos, eff, color="#208060", edgecolor='black', width=0.7)
            plt.xticks(x_pos, cat, rotation=15)
            plt.title(titre)
            plt.xlabel("Catégories")
            plt.ylabel("Effectif")
        elif "nuage de points" in gtype or "scatter" in gtype:
            x_points = graphique_dict.get("x", [])
            y_points = graphique_dict.get("y", [])
            plt.figure(figsize=(6, 4))
            plt.axhline(y=0, color='#000000', linewidth=1.8)
            plt.axvline(x=0, color='#000000', linewidth=1.8)
            plt.scatter(x_points, y_points, color="#006080")
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
        elif "effectifs cumulés" in gtype or "courbe des effectifs cumulés" in gtype:
            x_points = graphique_dict.get("x", [])
            y_points = graphique_dict.get("y", [])
            plt.figure(figsize=(6, 4))
            plt.axhline(y=0, color='#000000', linewidth=1.8)
            plt.axvline(x=0, color='#000000', linewidth=1.8)
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
            plt.axhline(y=0, color='#000000', linewidth=1.8)
            plt.axvline(x=0, color='#000000', linewidth=1.8)
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

d'autre exemples:
Exemple 1 : Fonction, Asymptotes, Réciproque

Soit ( f(x) = (x+1)/(x-2) ).
Représente dans le même repère :

La courbe de f
Ses asymptotes
Sa fonction réciproque f⁻¹(x)

Corrigé attendu : (détail…)
---corrigé---

{"graphique": { "type": "multi", "curves": [ {"type": "fonction", "expression": "(x+1)/(x-2)", "x_min": -5, "x_max": 7, "label": "f(x)"}, {"type": "asymptote", "sens": "verticale", "x": 2, "label": "x=2"}, {"type": "asymptote", "sens": "horizontale", "y": 1, "label": "y=1"}, {"type": "fonction", "expression": "(x-2)/(x+1)", "x_min": -5, "x_max": 7, "style":"--", "label": "f⁻¹(x)"} ], "titre": "Courbe f, asymptotes, réciproque" }}

Exemple 2 : ECC, ECD (statistique)

Les effectifs cumulés croissants (ECC) et décroissants (ECD) d’une série donnée sont donnés par… Trace dans le même repère les diagrammes ECC (points et segments) et ECD (segments pointillés).

---corrigé---

{"graphique": { "type": "multi", "curves": [ {"type": "ecc", "points_x": [0,5,10,15], "points_y": [3,8,17,20], "label": "ECC"}, {"type": "ecd", "points_x": [0,5,10,15], "points_y": [20,17,12,3], "label": "ECD", "style":"--"} ], "titre": "Diagrammes ECC et ECD sur la même figure" }}
Exemple 3 : Polygones plusieurs séries

On considère deux séries statistiques…

---corrigé---

{"graphique": { "type": "multi", "curves": [ {"type":"polygone","points":[[0,1],[5,3],[10,4],[15,8]],"label":"Série A"}, {"type":"polygone","points":[[0,2],[5,4],[10,5],[15,11]],"label":"Série B", "style": "--"} ], "titre": "Polygone séries A et B" }}

Rappels :
- Si plusieurs graphiques, recommence cette structure à chaque question concernée.
- Pas de texte entre ---corrigé--- et le JSON.
- Le JSON est obligatoire dès qu'un tracé est demandé.

"Rends TOUJOURS le JSON avec des guillemets doubles, jamais de dict Python. Pour les listes/types, toujours notation JSON [ ... ] et jamais { ... } sauf pour des objets. N’insère JAMAIS de virgule en trop."
"""


# ===========================
def generer_corrige_ia_et_graphique(texte_enonce, contexte, lecons_contenus=None, exemples_corriges=None, matiere=None,
                                    demande=None):
    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    # Prompt spécifique, ou général par défaut
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
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.9,
        "frequency_penalty": 0.1
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=120)
        response_data = response.json()

        if response.status_code != 200:
            error_msg = f"Erreur API DeepSeek: {response_data.get('message', 'Pas de détail')}"
            return error_msg, None

        output = response_data['choices'][0]['message']['content']
        print("\n=========== DEBUT OUTPUT HTML ===========\n")
        print(output)
        print("\n=========== FIN OUTPUT HTML =============\n")

        # Analyse et génération graphique dans le corrigé
        # Nouvelle regex robuste : capture le JSON qui suit ---corrigé--- (prend en compte espaces/retours à la ligne)
        regex_all_json = re.findall(r'---corrigé---\s*\n*({[\s\S]+?})', output)
        graph_list = []
        print("DEBUG JSONs détectés:", regex_all_json)  # Optionnel pour debug

        if regex_all_json:
            corrige_txt = output

            for idx, found_json in enumerate(regex_all_json, 1):
                try:
                    sjson = found_json.replace("'", '"').replace('\n', '').replace('\r', '').strip()
                    # Nettoyage du JSON généré par l'IA (supprime virgules parasites, espaces, caractères spéciaux)
                    sjson = re.sub(r'},\s*$', '}', sjson)
                    sjson = re.sub(r',\s*}', '}', sjson)
                    sjson = re.sub(r',\s*\]', ']', sjson)

                    # PATCH : Ajoute les accolades/crochets fermants manquants à la fin si besoin
                    nb_open = sjson.count("{")
                    nb_close = sjson.count("}")
                    if nb_close < nb_open:
                        sjson = sjson + "}" * (nb_open - nb_close)
                    nb_open = sjson.count("[")
                    nb_close = sjson.count("]")
                    if nb_close < nb_open:
                        sjson = sjson + "]" * (nb_open - nb_close)
                    # Debug : affiche après patch, avant le parsing JSON
                    print('DEBUG PATCHED sjson:', sjson)
                    graph_dict = json.loads(sjson)
                    output_name = f"graphique{idx}_{int(1000 * np.random.rand())}.png"
                    img_path = tracer_graphique(graph_dict, output_name)

                    if img_path:
                        abs_path = os.path.join(settings.MEDIA_ROOT, img_path)
                        img_tag = f'<img src="file://{abs_path}" alt="Graphique {idx}" style="max-width:100%;margin:10px 0;" />'

                        # Remplace aussi le tag ---corrigé--- juste avant ce JSON (tous formats possibles)
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
                    print("Erreur parsing JSON graphique:", e)
                    continue

            return corrige_txt.strip(), graph_list

        return output.strip(), None

    except Exception as e:
        return f"Erreur API: {str(e)}", None


# ===========================
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

        texte_enonce = ""
        if demande.fichier:
            texte_enonce = extraire_texte_fichier(demande.fichier)
        if not texte_enonce and hasattr(demande, 'enonce_texte'):
            texte_enonce = demande.enonce_texte or ""

        soumission.statut = 'analyse_ia'
        soumission.progression = 40
        soumission.save()

        matiere = Matiere.objects.get(id=matiere_id) if matiere_id else demande.matiere
        contexte = f"Exercice de {matiere.nom} - {demande.classe.nom if demande.classe else ''}"

        soumission.statut = 'generation_graphiques'
        soumission.progression = 60
        soumission.save()

        corrige_txt, graph_list = generer_corrige_ia_et_graphique(
            texte_enonce,
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

        return True

    except Exception as e:
        print("DEBUG ❌ Erreur dans la tâche IA :", e)
        try:
            soumission.statut = 'erreur'
            soumission.save()
        except:
            pass
        return False