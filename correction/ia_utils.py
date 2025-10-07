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
# UTILITAIRES TEXTE / LATEX / TABLEAU
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


def extract_and_process_graphs(output):
    """Nouvelle fonction robuste pour extraction des graphiques"""
    graphs_data = []
    final_text = output

    # Pattern amélioré pour capturer le JSON après ---corrigé---
    pattern = r'---corrigé---\s*\n*\s*(\{[\s\S]*?\})(?=\s*$|\s*---|\s*\n\s*\w)'
    matches = re.finditer(pattern, output)

    for match_idx, match in enumerate(matches):
        json_str = match.group(1).strip()

        try:
            # Nettoyage en plusieurs étapes
            json_str = re.sub(r"'", '"', json_str)  # Quotes simples → doubles
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Clés sans quotes
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Virgules traînantes
            json_str = re.sub(r'\s+', ' ', json_str)  # Espaces multiples

            print(f"DEBUG JSON {match_idx + 1} nettoyé: {json_str[:200]}...")

            # Validation et parsing
            graph_data = json.loads(json_str)
            graphs_data.append(graph_data)

            # Génération de l'image
            output_name = f"graphique_{match_idx + 1}_{np.random.randint(1000)}.png"
            img_path = tracer_graphique(graph_data, output_name)

            if img_path:
                # Remplacement dans le texte
                img_tag = f'<div class="graphique-container"><img src="/media/{img_path}" alt="Graphique {match_idx + 1}" style="max-width: 100%; border: 1px solid #ccc; margin: 20px 0;" /></div>'
                final_text = final_text.replace(match.group(0), img_tag)
            else:
                final_text = final_text.replace(match.group(0),
                                                '<div class="graphique-error">Erreur de génération du graphique</div>')

        except json.JSONDecodeError as e:
            print(f"ERREUR JSON: {e}")
            print(f"JSON problématique: {json_str}")
            final_text = final_text.replace(match.group(0), f'<div class="json-error">Erreur JSON: {str(e)}</div>')
        except Exception as e:
            print(f"ERREUR graphique: {e}")
            final_text = final_text.replace(match.group(0), f'<div class="graph-error">Erreur: {str(e)}</div>')

    return final_text, graphs_data


def generate_corrige_html(corrige_text):
    """Version corrigée - préserve le LaTeX"""
    if not corrige_text:
        return ""

    # Étape 1: Extraire et traiter les graphiques
    texte_avec_graphiques, graphs_list = extract_and_process_graphs(corrige_text)

    # Étape 2: Formater le reste du texte SANS toucher au LaTeX
    lines = texte_avec_graphiques.strip().split('\n')
    html_output = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Si la ligne est déjà du HTML (des graphiques)
        if line.startswith('<div class="graphique'):
            html_output.append(line)
            continue

        # Détection des blocs mathématiques
        if re.match(r'^\\\[.*\\\]$', line) or re.match(r'^\$\$.*\$\$$', line):
            html_output.append(f'<div class="math-display">{line}</div>')
        # Détection des math inline
        elif re.search(r'\\\(.*\\\)', line) or re.search(r'\$[^$]+\$', line):
            # Laisser le LaTeX intact pour le frontend
            html_output.append(f'<p>{line}</p>')
        # Tableaux
        elif line.startswith('|') and '|' in line:
            # Chercher les lignes suivantes pour former le tableau complet
            table_lines = [line]
            current_idx = lines.index(line) + 1
            while current_idx < len(lines) and lines[current_idx].strip().startswith('|'):
                table_lines.append(lines[current_idx].strip())
                current_idx += 1
            html_table = format_table_markdown('\n'.join(table_lines))
            html_output.append(html_table)
            # Sauter les lignes déjà traitées
            continue
        # Titres
        elif re.match(r'^\d+\.', line):
            html_output.append(f'<h3>{line}</h3>')
        # Sous-titres
        elif re.match(r'^[a-z]\)', line):
            html_output.append(f'<p><strong>{line}</strong></p>')
        # Listes
        elif line.startswith(('•', '-', '·')):
            html_output.append(f'<p>{line}</p>')
        # Par défaut
        else:
            html_output.append(f'<p>{line}</p>')

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
            plt.figure(figsize=(8, 6))
            plt.axhline(y=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.axvline(x=0, color='#000000', linewidth=1.2, alpha=0.7)
            color_cycle = iter(['#008060', '#b65d2f', '#2080C0', '#DB1919', '#003355', '#C08800', '#591e63'])

            for courbe in curves:
                ctype = courbe.get("type", "")
                label = courbe.get("label", "")
                plot_color = courbe.get("color", next(color_cycle))
                style = courbe.get("style", "-")
                linewidth = courbe.get("linewidth", 2)

                # Tracé de fonction
                if ctype == "fonction":
                    x_min = safe_float(courbe.get("x_min", -2)) or -2
                    x_max = safe_float(courbe.get("x_max", 4)) or 4
                    expression = courbe.get("expression", "x")
                    x = np.linspace(x_min, x_max, 400)
                    expr_patch = expression.replace('^', '**')

                    for func in [
                        "sin", "cos", "tan", "exp", "log", "log10", "arcsin",
                        "arccos", "arctan", "sinh", "cosh", "tanh", "sqrt", "abs"]:
                        expr_patch = re.sub(r'(?<![\w.])' + func + r'\s*\(', f'np.{func}(', expr_patch)
                    expr_patch = expr_patch.replace('ln(', 'np.log(')

                    try:
                        y = eval(expr_patch, {'x': x, 'np': np, '__builtins__': None, "pi": np.pi})
                        if np.isscalar(y):
                            y = np.full_like(x, y)
                        plt.plot(x, y, style, label=label, color=plot_color, linewidth=linewidth)
                    except Exception as e:
                        print(f"Erreur tracé fonction {expression}: {e}")
                        continue

                # Asymptotes
                elif ctype == "asymptote":
                    if courbe.get("sens") == "verticale" and "x" in courbe:
                        x_val = safe_float(courbe["x"])
                        if x_val is not None:
                            plt.axvline(x=x_val, color=plot_color, linestyle='--', linewidth=1.5, label=label)
                    if courbe.get("sens") == "horizontale" and "y" in courbe:
                        y_val = safe_float(courbe["y"])
                        if y_val is not None:
                            plt.axhline(y=y_val, color=plot_color, linestyle='--', linewidth=1.5, label=label)

                # Nuage de points
                elif ctype in ("nuage de points", "points", "scatter"):
                    x_points = courbe.get("x", [])
                    y_points = courbe.get("y", [])
                    if x_points and y_points and len(x_points) == len(y_points):
                        plt.scatter(x_points, y_points, label=label, color=plot_color, s=50)

                # Polygone, ECC/ECD, etc.
                elif ctype in ("polygone", "ecd", "ecc"):
                    points = courbe.get("points", [])
                    points_x = courbe.get("points_x", [])
                    points_y = courbe.get("points_y", [])
                    absc = courbe.get("abscisses", [])
                    ords = courbe.get("ordonnees", [])

                    if points:
                        x = [float(p[0]) for p in points if len(p) >= 2]
                        y = [float(p[1]) for p in points if len(p) >= 2]
                    elif points_x and points_y and len(points_x) == len(points_y):
                        x = [float(xx) for xx in points_x]
                        y = [float(yy) for yy in points_y]
                    elif absc and ords and len(absc) == len(ords):
                        x = [float(xx) for xx in absc]
                        y = [float(yy) for yy in ords]
                    else:
                        continue

                    if x and y:
                        plt.plot(x, y, marker="o" if ctype != "asymptote" else "",
                                 linestyle=style, color=plot_color, label=label, linewidth=linewidth)

            plt.title(titre, fontsize=14, pad=20)
            plt.xlabel("x", fontsize=12)
            plt.ylabel("y", fontsize=12)
            plt.grid(True, alpha=0.3)
            if any(courbe.get('label') for courbe in curves):
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
                expr_patch = re.sub(r'(?<![\w.])' + func + r'\s*\(', f'np.{func}(', expr_patch)
            expr_patch = expr_patch.replace('ln(', 'np.log(')

            try:
                y = eval(expr_patch, {'x': x, 'np': np, '__builtins__': None, "pi": np.pi})
                if np.isscalar(y):
                    y = np.full_like(x, y)
            except Exception as e:
                print(f"Erreur tracé expression: {e}")
                return None

            plt.figure(figsize=(8, 6))
            plt.axhline(y=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.axvline(x=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.plot(x, y, color="#008060", linewidth=2.5)
            plt.title(titre, fontsize=14, pad=20)
            plt.xlabel("x", fontsize=12)
            plt.ylabel("y", fontsize=12)
            plt.grid(True, alpha=0.3)

            # Tracé éventuel asymptotes
            asymptotes = graphique_dict.get("asymptotes")
            if asymptotes:
                if isinstance(asymptotes, dict):
                    for xval in asymptotes.get("verticales", []):
                        x_val = safe_float(xval)
                        if x_val is not None:
                            plt.axvline(x=x_val, linestyle="--", color="#DB1919", alpha=0.7)
                    for yval in asymptotes.get("horizontales", []):
                        y_val = safe_float(yval)
                        if y_val is not None:
                            plt.axhline(y=y_val, linestyle="--", color="#0070e0", alpha=0.7)
                elif isinstance(asymptotes, list):
                    for item in asymptotes:
                        if isinstance(item, dict):
                            if item.get("type") == "verticale" and "x" in item:
                                x_val = safe_float(item["x"])
                                if x_val is not None:
                                    plt.axvline(x=x_val, linestyle="--", color="#DB1919", alpha=0.7)
                            elif item.get("type") == "horizontale" and "y" in item:
                                y_val = safe_float(item["y"])
                                if y_val is not None:
                                    plt.axhline(y=y_val, linestyle="--", color="#0070e0", alpha=0.7)

        elif "histogramme" in gtype:
            intervalles = graphique_dict.get("intervalles") or graphique_dict.get("classes") or []
            eff = graphique_dict.get("effectifs", [])
            labels = [str(ival) for ival in intervalles]
            x_pos = np.arange(len(labels))
            eff = [float(e) for e in eff]

            plt.figure(figsize=(8, 6))
            plt.axhline(y=0, color='#000000', linewidth=1.2, alpha=0.7)
            bars = plt.bar(x_pos, eff, color="#208060", edgecolor='black', width=0.8, alpha=0.8)
            plt.xticks(x_pos, labels, rotation=45, ha='right')
            plt.title(titre, fontsize=14, pad=20)
            plt.xlabel(graphique_dict.get("xlabel", "Classes / Intervalles"), fontsize=12)
            plt.ylabel(graphique_dict.get("ylabel", "Effectif"), fontsize=12)
            plt.grid(axis='y', alpha=0.3)

            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                         f'{height:.0f}', ha='center', va='bottom', fontsize=10)

        elif "diagramme à bandes" in gtype or "diagramme en bâtons" in gtype or "bâtons" in gtype or "batons" in gtype:
            cat = graphique_dict.get("categories", [])
            eff = graphique_dict.get("effectifs", [])
            x_pos = np.arange(len(cat))

            plt.figure(figsize=(8, 6))
            plt.axhline(y=0, color='#000000', linewidth=1.2, alpha=0.7)
            bars = plt.bar(x_pos, eff, color="#208060", edgecolor='black', width=0.7, alpha=0.8)
            plt.xticks(x_pos, cat, rotation=45, ha='right')
            plt.title(titre, fontsize=14, pad=20)
            plt.xlabel("Catégories", fontsize=12)
            plt.ylabel("Effectif", fontsize=12)
            plt.grid(axis='y', alpha=0.3)

            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                         f'{height:.0f}', ha='center', va='bottom', fontsize=10)

        elif "nuage de points" in gtype or "scatter" in gtype:
            x_points = graphique_dict.get("x", [])
            y_points = graphique_dict.get("y", [])

            plt.figure(figsize=(8, 6))
            plt.axhline(y=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.axvline(x=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.scatter(x_points, y_points, color="#006080", s=60, alpha=0.7)
            plt.title(titre, fontsize=14, pad=20)
            plt.xlabel("x", fontsize=12)
            plt.ylabel("y", fontsize=12)
            plt.grid(True, alpha=0.3)

        elif "effectifs cumulés" in gtype or "courbe des effectifs cumulés" in gtype:
            x_points = graphique_dict.get("x", [])
            y_points = graphique_dict.get("y", [])

            plt.figure(figsize=(8, 6))
            plt.axhline(y=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.axvline(x=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.plot(x_points, y_points, marker="o", color="#b65d2f", linewidth=2.5, markersize=6)
            plt.title(titre, fontsize=14, pad=20)
            plt.xlabel("x", fontsize=12)
            plt.ylabel("Effectifs cumulés", fontsize=12)
            plt.grid(True, alpha=0.3)

        elif "diagramme circulaire" in gtype or "camembert" in gtype or "pie" in gtype:
            cat = graphique_dict.get("categories", [])
            eff = graphique_dict.get("effectifs", [])

            plt.figure(figsize=(8, 8))
            wedges, texts, autotexts = plt.pie(
                eff,
                labels=cat,
                autopct='%1.1f%%',
                colors=plt.cm.Set3.colors,
                startangle=90,
                wedgeprops={"edgecolor": "k", "linewidth": 1}
            )
            plt.title(titre, fontsize=14, pad=20)

            # Améliorer la lisibilité
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')

        elif "polygone" in gtype or "polygon" in gtype:
            points = graphique_dict.get("points", [])
            points_x = graphique_dict.get("points_x", [])
            points_y = graphique_dict.get("points_y", [])
            absc = graphique_dict.get("abscisses", [])
            ords = graphique_dict.get("ordonnees", [])

            if points:
                x = [float(p[0]) for p in points if len(p) >= 2]
                y = [float(p[1]) for p in points if len(p) >= 2]
            elif points_x and points_y:
                x = [float(xx) for xx in points_x]
                y = [float(yy) for yy in points_y]
            elif absc and ords:
                x = [float(xx) for xx in absc]
                y = [float(yy) for yy in ords]
            else:
                print("Erreur polygone : aucun point valide")
                return None

            plt.figure(figsize=(8, 6))
            plt.axhline(y=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.axvline(x=0, color='#000000', linewidth=1.2, alpha=0.7)
            plt.plot(x, y, marker="o", color="#003355", linewidth=2.5, markersize=6)
            plt.title(graphique_dict.get("titre", "Polygone"), fontsize=14, pad=20)
            plt.xlabel(graphique_dict.get("x_label", "Abscisse"), fontsize=12)
            plt.ylabel(graphique_dict.get("y_label", "Ordonnée"), fontsize=12)
            plt.grid(True, alpha=0.3)

        elif "cercle trigo" in gtype:
            angles = graphique_dict.get("angles", [])
            labels = graphique_dict.get("labels", [])

            plt.figure(figsize=(8, 8))
            circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linestyle='-', linewidth=2)
            ax = plt.gca()
            ax.add_artist(circle)

            # Axes
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

            for i, angle_txt in enumerate(angles):
                try:
                    a = float(eval(angle_txt, {"pi": np.pi, "__builtins__": None}))
                except Exception:
                    a = 0
                x, y = np.cos(a), np.sin(a)
                ax.plot([0, x], [0, y], color='#992020', linewidth=2)
                label = labels[i] if i < len(labels) else f"S{i + 1}"
                ax.text(1.15 * x, 1.15 * y, label, fontsize=12, ha='center', va='center')

            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            plt.title(titre, fontsize=14, pad=20)

        else:
            print("Type graphique non supporté :", gtype)
            return None

        plt.tight_layout()
        plt.savefig(chemin_png, dpi=150, bbox_inches='tight')
        plt.close()

        return "graphes/" + output_name

    except Exception as ee:
        print(f"Erreur générale sauvegarde PNG: {ee}")
        return None


# ===========================
# PROMPT PAR DEFAUT AMÉLIORÉ
IMPROVED_SYSTEM_PROMPT = r"""Tu es un professeur expert en sciences (Maths, Physique, SVT, Chimie, Statistique).

Règles STRICTES :

1. **POUR LES RÉPONSES** :
- Explique étape par étape
- Utilise \(...\) pour les formules inline et \[...\] pour les formules centrées
- Pour les tableaux : format Markdown avec |

2. **POUR LES GRAPHIQUES** :
- Quand un exercice demande un graphique, termine la réponse par :
---corrigé---
{"graphique": {...}}

- Le JSON DOIT ÊTRE VALIDE :
  - Toujours des guillemets doubles "
  - Pas de virgules traînantes
  - Toutes les clés entre guillemets
  - Pas de simples quotes '

3. **EXEMPLES CORRECTS** :

Exemple fonction :
---corrigé---
{"graphique": {"type": "fonction", "expression": "x**2 - 2*x + 1", "x_min": -1, "x_max": 3, "titre": "Courbe parabole"}}

Exemple multi-courbes :
---corrigé---
{"graphique": {"type": "multi", "curves": [{"type": "fonction", "expression": "sin(x)", "label": "sin(x)"}, {"type": "fonction", "expression": "cos(x)", "label": "cos(x)"}], "titre": "Fonctions trigo"}}

Exemple histogramme :
---corrigé---
{"graphique": {"type": "histogramme", "intervalles": ["0-5","5-10","10-15"], "effectifs":[3,5,7], "titre":"Histogramme"}}

4. **TYPES SUPPORTÉS** :
- "fonction", "histogramme", "diagramme à bandes" 
- "nuage de points", "effectifs cumulés", "diagramme circulaire"
- "polygone", "cercle trigo", "multi" (plusieurs courbes)

5. **INTERDICTIONS** :
- JAMAIS de simples quotes '
- JAMAIS de virgules après le dernier élément
- JAMAIS de clés sans guillemets
- JAMAIS de texte entre ---corrigé--- et le JSON

Format de réponse strict : LaTeX pour les maths, explications détaillées, JSON valide pour les graphiques.
"""


# ===========================
def generer_corrige_ia_et_graphique(texte_enonce, contexte, lecons_contenus=None, exemples_corriges=None, matiere=None,
                                    demande=None):
    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    # Utiliser le prompt amélioré
    system_prompt = IMPROVED_SYSTEM_PROMPT
    exemple_prompt = ""
    consignes_finales = "Format de réponse strict : LaTeX pour les maths, Markdown pour les tableaux, JSON valide pour les graphiques"

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
    if not api_key:
        return "Erreur: Clé API DeepSeek non configurée", None

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

        print("\n" + "=" * 50)
        print("DEBUG: OUTPUT BRUT DE L'IA")
        print("=" * 50)
        print(output)
        print("=" * 50)

        # Traitement avec la nouvelle fonction
        corrige_final, graphiques = extract_and_process_graphs(output)

        return corrige_final, graphiques

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


