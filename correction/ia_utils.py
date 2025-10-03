import requests
import os
import tempfile
import json
from pdfminer.high_level import extract_text
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import numpy as np
import matplotlib
from celery import shared_task
from django.conf import settings
import re
from django.utils.safestring import mark_safe

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .pdf_generator import pdf_generator

# --- Conversion et sanitation latex/markdown vers HTML+LaTeX ---
def flatten_multiline_latex_blocks(text):
    """
    Remplace tout bloc latex (\[...\] ou \(...\)) multi-ligne par la même balise sur UNE SEULE ligne.
    """
    if not text: return ""
    def block_replacer(match):
        contents = match.group(1).replace('\n', ' ').replace('\r', ' ')
        contents = re.sub(r' {2,}', ' ', contents)
        return r'\[' + contents.strip() + r'\]'
    def inline_replacer(match):
        contents = match.group(1).replace('\n', ' ').replace('\r', ' ')
        contents = re.sub(r' {2,}', ' ', contents)
        return r'\(' + contents.strip() + r'\)'
    # Flatten les display latex
    text = re.sub(r'\\\[\s*([\s\S]*?)\s*\\\]', block_replacer, text)
    # Inline aussi
    text = re.sub(r'\\\(\s*([\s\S]*?)\s*\\\)', inline_replacer, text)
    return text

def detect_and_format_math_expressions(text):
    """
    Sanitize latex : met tous les blocs latex (display / inline) sur une seule ligne
    Évite de masquer/protéger plus que nécessaire — traite tout ce qui traîne.
    """
    if not text: return ""

    # Correction de balises erronées issues de l'IA ou Markdown :
    text = re.sub(r'\$\$\s*([\s\S]+?)\s*\$\$', lambda m: r'\[' + m.group(1).replace('\n', ' ').strip() + r'\]', text, flags=re.DOTALL)
    text = re.sub(r'\$\s*([^$]+?)\s*\$', lambda m: r'\(' + m.group(1).replace('\n', ' ').strip() + r'\)', text)
    text = re.sub(r'(?<!\\)\[\s*([\s\S]+?)\s*\]', lambda m: r'\[' + ' '.join(m.group(1).splitlines()).strip() + r'\]', text)

    # Mets tous les blocs latex (display/inline) sur UNE SEULE ligne
    text = flatten_multiline_latex_blocks(text)
    # Clean backslash parasite, caractères invisibles
    text = text.replace('\\backslash', '\\').replace('\xa0', ' ')
    return text

def format_table_markdown(table_text):
    """
    Transforme un tableau markdown (type |...|...|) en vrai tableau HTML
    """
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
    """
    Transforme latex + texte en HTML+latex prêt pour flutter_tex.
    Gère tableaux, titres, blocs latex display sur une ligne, etc.
    """
    if not corrige_text:
        return ""
    formatted_text = detect_and_format_math_expressions(corrige_text)
    lines = formatted_text.strip().split('\n')
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
        if re.search(r'\\\[.*?\\\]', line):
            line = re.sub(r'\\\[(\s*)(.*?)(\s*)\\\]', r'\[\2\]', line)
            html_output.append(f'<p>{line}</p>'); i += 1
        elif re.match(r'^\d+\.', line):
            html_output.append(f'<h2>{line}</h2>'); i += 1
        elif re.match(r'^[a-z]\)', line):
            html_output.append(f'<p><strong>{line}</strong></p>'); i += 1
        elif line.startswith('•') or line.startswith('-') or line.startswith('·'):
            html_output.append(f'<p>{line}</p>'); i += 1
        elif '\\(' in line or '\\[' in line:
            line = re.sub(r'\\\(\s*([^)]*?)\s*\\\)', r'\\(\1\\)', line)
            line = re.sub(r'\\\[\s*([^]]*?)\s*\\\]', r'\[\1\]', line)
            html_output.append(f'<p>{line}</p>'); i += 1
        else:
            html_output.append(f'<p>{line}</p>'); i += 1
    return mark_safe("".join(html_output))
# --- Fin sanitation/converter ---

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

        # Prétraitement amélioré de l'image
        # Conversion en niveaux de gris
        image = image.convert("L")

        # Amélioration du contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # Réduction du bruit
        image = image.filter(ImageFilter.MedianFilter(size=3))

        # Seuillage adaptatif
        image = image.point(lambda x: 0 if x < 180 else 255, '1')

        # Configuration Tesseract améliorée
        custom_config = r'--oem 3 --psm 6 -l fra+eng'

        texte = pytesseract.image_to_string(image, config=custom_config)

        if not texte.strip():
            # Essayer avec différents modes PSM
            for psm_mode in [6, 7, 8, 11]:
                custom_config = f'--oem 3 --psm {psm_mode} -l fra+eng'
                texte = pytesseract.image_to_string(image, config=custom_config)
                if texte.strip():
                    break

        return texte.strip() if texte else "(Texte non extrait de l'image)"

    except Exception as e:
        print("Erreur extraction image:", e)
        return f"(Erreur lors de l'extraction: {str(e)})"


def extraire_texte_fichier(fichier_field):
    if not fichier_field:
        return ""

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, os.path.basename(fichier_field.name))

    try:
        with open(temp_path, "wb") as f:
            for chunk in fichier_field.chunks():
                f.write(chunk)

        ext = os.path.splitext(fichier_field.name)[1].lower()
        texte = ""

        print(f"Extraction fichier: {fichier_field.name}, type: {ext}")

        if ext == ".pdf":
            texte = extraire_texte_pdf(temp_path)
            print(f"PDF extrait: {len(texte)} caractères")
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            texte = extraire_texte_image(temp_path)
            print(f"Image extraite: {len(texte)} caractères")
        else:
            texte = f"Format non supporté: {ext}"

        return texte if texte.strip() else "(Impossible d'extraire l'énoncé du fichier envoyé.)"

    except Exception as e:
        print(f"Erreur extraction fichier {fichier_field.name}: {e}")
        return f"(Erreur lors de l'extraction: {str(e)})"
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass



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

        if "fonction" in gtype:
            x_min_val = safe_float(graphique_dict.get("x_min", -2)) or -2
            x_max_val = safe_float(graphique_dict.get("x_max", 4)) or 4
            expression = graphique_dict.get("expression", "x")

            x = np.linspace(x_min_val, x_max_val, 400)
            expression_patch = expression.replace('^', '**')

            funcs = ["sin", "cos", "tan", "exp", "log", "log10", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
                     "sqrt", "abs"]
            for fct in funcs:
                expression_patch = re.sub(r'(?<![\w.])' + fct + r'\s*\(', f'np.{fct}(', expression_patch)
            expression_patch = expression_patch.replace('ln(', 'np.log(')

            try:
                y = eval(expression_patch, {'x': x, 'np': np, '__builtins__': None, "pi": np.pi, "sqrt": np.sqrt})
                if np.isscalar(y) or (isinstance(y, np.ndarray) and y.shape == ()):
                    y = np.full_like(x, y)
            except Exception as e:
                print(f"Erreur tracé expression: {e}")
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
            effectifs = graphique_dict.get("effectifs", [])

            try:
                labels = [str(ival) for ival in intervalles]
                x_pos = np.arange(len(labels))
                effectifs = [float(e) for e in effectifs]

                plt.figure(figsize=(7, 4.5))
                plt.bar(x_pos, effectifs, color="#208060", edgecolor='black', width=0.9)
                plt.xticks(x_pos, labels, rotation=35)
                plt.title(titre)
                plt.xlabel(graphique_dict.get("xlabel", "Classes / Intervalles"))
                plt.ylabel(graphique_dict.get("ylabel", "Effectif"))
                plt.grid(axis='y')
                plt.tight_layout()
            except Exception as e:
                print("Erreur histogramme:", e)
                return None

        # [Autres types de graphiques conservés...]
        # ... (le reste de votre code pour les autres types de graphiques)

        plt.savefig(chemin_png)
        plt.close()
        return f"graphes/{output_name}"

    except Exception as e:
        print(f"Erreur génération graphique: {e}")
        return None


def generer_corrige_ia_et_graphique(texte_enonce, contexte, lecons_contenus=None, exemples_corriges=None, matiere=None, demande=None):
    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    # PROMPT IA personnalisable par matière
    DEFAULT_SYSTEM_PROMPT =r"""
Tu es un professeur expert chargé de corriger des exercices de façon structurée, claire et rigoureuse.
Règles incontournables :
- Structure chaque corrigé sans sauter d'étapes
- Toutes les formules mathématiques doivent être en LaTeX avec \( \) pour inline et \[ \] pour display,
- voici un exemple d'équation attendue:  \[\lim_{x \to \pm \infty} f(x) = \lim_{x \to \pm \infty} \frac{x}{x} = 1\]
- voici un autre exemple: \( f(x) = 0 \Rightarrow x = 1 \) 
- NE PAS répéter l'énoncé des questions avant chaque réponse
- répondre clairement à la question sans préciser la méthode
-évite de mettre entre crochets quoique ce soit, seuls le latex aura les balises décrites ci-dessus
- Pour les équations, utiliser \(\implies\) ou \(\iff\) à chaque étape
- Les tableaux en Markdown avec alignement correct
- Toutes les formules doivent être en LaTeX avec \( ... \) (inline) et \[ ... \] (display), SANS aucuns retours à la ligne dans ou autour des balises.
- Pour les équations block : TOUJOURS une seule ligne \[ ... \]
- Ne produis jamais de formule math, ni d'expression dans une simple parenthèse ou crochets. Toute équation doit être sous les balises latex décrites ci-dessus, sur une ligne stricte.
- Utiliser un langage clair et pédagogique

IMPORTANT : 
- A chaque fois qu'un exercice demande de TRACER ou REPRÉSENTER une courbe/un graphique :
— Ne donne JAMAIS une simple description.
— Tu dois OBLIGATOIREMENT insérer précisément, à la fin du paragraphe concerné, un bloc JSON du type suivant (toujours sur une seule ligne, jamais entouré de texte, jamais modifié) :
 {"graphique":{"type":"fonction","expression":"sin(x)","titre":"Courbe de g","x_min":"-5","x_max":"5"}}
- Sans ce bloc JSON, ton corrigé sera invalidé !
- La description de la courbe/ne doit jamais remplacer ce bloc JSON obligatoire.
“Après CHAQUE question qui nécessite un graphique, tu termines TOUTES tes phrases, puis sur une LIGNE À PART, tu écris :\n\n{JSON}\n\n. Tu ne dois JAMAIS oublier ni placer ce JSON ailleurs.”
"""

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
        "max_tokens": 4000,
        "top_p": 0.3,
        "frequency_penalty": 0.2
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=120)
        response_data = response.json()

        if response.status_code != 200:
            error_msg = f"Erreur API DeepSeek: {response_data.get('message', 'Pas de détail')}"
            return error_msg, None

        output = response_data['choices'][0]['message']['content']

        # Traitement des graphiques
        regex_all_json = re.findall(r'(\{\s*"graphique"\s*:\s*\{[\s\S]+?\}\s*\})', output)
        graph_list = []

        if regex_all_json:
            corrige_txt = output
            for idx, found_json in enumerate(regex_all_json, 1):
                try:
                    # Décodage JSON
                    sjson = found_json.replace("'", '"').replace('\n', '').replace('\r', '').strip()
                    graph_dict = json.loads(sjson)
                    # Génère le graphique ici !
                    output_name = f"graphique_{idx}_{int(1000 * np.random.rand())}.png"
                    img_path = tracer_graphique(graph_dict, output_name)
                    # Compose le tag d’image HTML ou markdown
                    if img_path:
                        img_tag = f'<img src="/media/{img_path}" alt="Graphique {idx}" style="max-width:100%;margin:10px 0;" />'
                        # REMPLACE le tag dans le texte
                        corrige_txt = corrige_txt.replace(found_json, img_tag, 1)
                    else:
                        # Si génération échoue, laisse le tag ou mets un message d'erreur
                        corrige_txt = corrige_txt.replace(found_json, f"[Erreur génération graphique]", 1)
                    graph_list.append(graph_dict)
                except Exception as e:
                    print("Erreur parsing JSON graphique:", e)
                    continue
            return corrige_txt.strip(), graph_list

        return output.strip(), None

    except Exception as e:
        return f"Erreur API: {str(e)}", None

from celery import shared_task
# … tes imports existants …

@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(demande_id, matiere_id=None):
    print(f"DEBUG ▶️ Tâche IA démarrée pour demande_id={demande_id}, matiere_id={matiere_id}")
    from correction.models import DemandeCorrection, SoumissionIA
    from resources.models   import Matiere

    try:
        # Récupération des objets
        demande = DemandeCorrection.objects.get(id=demande_id)
        soumission = SoumissionIA.objects.get(demande=demande)
        print("DEBUG    → DemandeCorrection et SoumissionIA récupérés")

        # Étape 1 : extraction
        soumission.statut = 'extraction'
        soumission.progression = 20
        soumission.save()
        print("DEBUG    • Statut « extraction » (20%)")

        # Extraction du texte
        texte_enonce = ""
        if demande.fichier:
            texte_enonce = extraire_texte_fichier(demande.fichier)
        if not texte_enonce and hasattr(demande, 'enonce_texte'):
            texte_enonce = demande.enonce_texte or ""
        print(f"DEBUG    • Texte énoncé extrait ({len(texte_enonce)} caractères)")

        # Étape 2 : analyse IA
        soumission.statut = 'analyse_ia'
        soumission.progression = 40
        soumission.save()
        print("DEBUG    • Statut « analyse_ia » (40%)")

        # Contexte IA
        matiere = Matiere.objects.get(id=matiere_id) if matiere_id else demande.matiere
        contexte = f"Exercice de {matiere.nom} - {demande.classe.nom if demande.classe else ''}"

        # Étape 3 : génération graphiques
        soumission.statut = 'generation_graphiques'
        soumission.progression = 60
        soumission.save()
        print("DEBUG    • Statut « generation_graphiques » (60%)")

        # Appel de l’IA
        corrige_txt, graph_list = generer_corrige_ia_et_graphique(
            texte_enonce, contexte, matiere=matiere
        )
        print("DEBUG    • IA renvoyé texte et éventuels graphiques")

        # Étape 4 : formatage PDF
        soumission.statut = 'formatage_pdf'
        soumission.progression = 80
        soumission.save()
        print("DEBUG    • Statut « formatage_pdf » (80%)")

        # Génération du PDF
        from .pdf_utils import generer_pdf_corrige
        pdf_path = generer_pdf_corrige(
            {"titre_corrige": contexte, "corrige_html": corrige_txt, "soumission_id": demande_id},
            demande_id
        )
        print(f"DEBUG    • PDF généré à l’emplacement : {pdf_path}")

        # Étape finale : terminé
        soumission.statut = 'termine'
        soumission.progression = 100
        soumission.resultat_json = {
            'corrige_text': corrige_txt,
            'pdf_url': pdf_path,
            'graphiques': graph_list or []
        }
        soumission.save()
        print("DEBUG ✔️ Tâche IA terminée (100%)")

        # On met à jour la demande
        demande.corrigé = corrige_txt
        demande.save()

        return True

    except Exception as e:
        print("DEBUG ❌ Erreur dans la tâche IA :", e)
        # En cas d’erreur on met à jour le statut
        try:
            soumission.statut = 'erreur'
            soumission.save()
        except:
            pass
        return False