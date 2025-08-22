import requests  # Ajout pour les appels API à DeepSeek
import re as _re
import os
import tempfile
import json
from pdfminer.high_level import extract_text
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from celery import shared_task


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
        # Prétraitement : niveaux de gris, contraste, binarisation
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
            try: return float(expr)
            except Exception as e2: print("Erreur safe_float cast direct:", expr, e2); return None
    try:
        from django.conf import settings
        dossier = os.path.join(settings.MEDIA_ROOT, "graphes")
        os.makedirs(dossier, exist_ok=True)
        chemin_png = os.path.join(dossier, output_name)
        if "fonction" in gtype:
            x_min = graphique_dict.get("x_min", -2)
            x_max = graphique_dict.get("x_max", 4)
            expression = graphique_dict.get("expression", "x")
            x_min_val = safe_float(x_min)
            x_max_val = safe_float(x_max)
            if x_min_val is None: x_min_val = -2
            if x_max_val is None: x_max_val = 4
            x = np.linspace(x_min_val, x_max_val, 400)
            expression_patch = expression.replace('^', '**')
            # PATCH pour ln et autres fonctions/math
            funcs = [
                "sin", "cos", "tan", "exp", "log", "log10",
                "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "sqrt", "abs"
            ]
            for fct in funcs:
                expression_patch = re.sub(r'(?<![\w.])' + fct + r'\s*\(', f'np.{fct}(', expression_patch)
            expression_patch = expression_patch.replace('ln(', 'np.log(')
            print(f">>> final expression to eval = {expression_patch}")
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
            effectifs = graphique_dict.get("effectifs", [])
            print(f">>> HISTO : intervalles={intervalles}, effectifs={effectifs}")
            try:
                labels = [str(ival) for ival in intervalles]
                x_pos = np.arange(len(labels))
                effectifs = [float(e) for e in effectifs]
                print(f">>> x_pos={x_pos}, labels={labels}, effectifs(san)={effectifs}")
                plt.figure(figsize=(7, 4.5))
                plt.bar(x_pos, effectifs, color="#208060", edgecolor='black', width=0.9)
                plt.xticks(x_pos, labels, rotation=35)
                plt.title(titre)
                plt.xlabel(graphique_dict.get("xlabel", "Classes / Intervalles"))
                plt.ylabel(graphique_dict.get("ylabel", "Effectif"))
                plt.grid(axis='y')
                plt.tight_layout()
            except Exception as e:
                print("Erreur dans le bloc HISTO :", e)
                return None
        elif "diagramme à bandes" in gtype or "diagramme en bâtons" in gtype or "diagramme à batons" in gtype:
            categories = graphique_dict.get("categories", [])
            effectifs = graphique_dict.get("effectifs", [])
            print(f">>> BANDS : categories={categories}, effectifs={effectifs}")
            x_pos = np.arange(len(categories))
            plt.figure(figsize=(7, 4.5))
            plt.bar(x_pos, effectifs, color="#208060", edgecolor='black', width=0.7)
            plt.xticks(x_pos, categories, rotation=15)
            plt.title(titre)
            plt.xlabel("Catégories")
            plt.ylabel("Effectif")
            plt.tight_layout()
        elif "nuage de points" in gtype:
            x_points = graphique_dict.get("x", [])
            y_points = graphique_dict.get("y", [])
            print(f">>> SCATTER : x={x_points}, y={y_points}")
            plt.figure(figsize=(6, 4))
            plt.scatter(x_points, y_points, color="#006080")
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.tight_layout()
        elif "effectifs cumulés" in gtype or "courbe des effectifs cumulés" in gtype:
            x_points = graphique_dict.get("x", [])
            y_points = graphique_dict.get("y", [])
            print(f">>> EC : x={x_points}, y={y_points}")
            plt.figure(figsize=(6, 4))
            plt.plot(x_points, y_points, marker="o", color="#b65d2f")
            plt.title(titre)
            plt.xlabel("x")
            plt.ylabel("Effectifs cumulés")
            plt.grid(True)
            plt.tight_layout()
        elif "diagramme circulaire" in gtype or "camembert" in gtype or "pie" in gtype:
            categories = graphique_dict.get("categories", [])
            effectifs = graphique_dict.get("effectifs", [])
            print(f">>> PIE : categories={categories}, effectifs={effectifs}")
            plt.figure(figsize=(5.3, 5.3))
            plt.pie(effectifs, labels=categories, autopct='%1.1f%%', colors=plt.cm.Paired.colors, startangle=90, wedgeprops={"edgecolor":"k"})
            plt.title(titre)
            plt.tight_layout()
        elif "polygone" in gtype:
            # Prendre en compte tous les formats possibles pour la polygone statistique
            points = graphique_dict.get("points")
            points_x = graphique_dict.get("points_x")
            points_y = graphique_dict.get("points_y")
            abscisses = graphique_dict.get("abscisses")
            ordonnees = graphique_dict.get("ordonnees")
            # 1. Format [[x, y], ...]
            if points:
                x = [float(p[0]) for p in points]
                y = [float(p[1]) for p in points]
            # 2. Format points_x + points_y (listes coordonnées)
            elif points_x and points_y:
                x = [float(xx) for xx in points_x]
                y = [float(yy) for yy in points_y]
            # 3. Format abscisses/ordonnees
            elif abscisses and ordonnees:
                x = [float(xx) for xx in abscisses]
                y = [float(yy) for yy in ordonnees]
            else:
                # Cas d'erreur : rien à tracer
                print("Erreur : aucun point trouvé pour le polygone")
                x = []
                y = []
            print(f">>> POLYGONE : x={x}, y={y}")
            plt.figure(figsize=(7, 4.5))
            plt.plot(x, y, marker="o", color="#003355")
            plt.title(graphique_dict.get("titre", "Polygone des effectifs cumulés"))
            plt.xlabel(graphique_dict.get("x_label", "Abscisse"))
            plt.ylabel(graphique_dict.get("y_label", "Effectifs cumulés ou ordonnée"))
            plt.grid(True)
            plt.tight_layout()
        else:
            print("Type de graphique non supporté dans le dict reçu :", gtype)
            return None
        plt.savefig(chemin_png)
        plt.close()
        print(f"IMAGE PNG GENERE : {chemin_png} -- accessible à /media/graphes/{output_name}")
        return "graphes/" + output_name
    except Exception as ee:
        print(f"Erreur générale sauvegarde PNG {chemin_png if 'chemin_png' in locals() else output_name} :", ee)
        return None




def generer_corrige_ia_et_graphique(
        texte_enonce,
        contexte,
        lecons_contenus=None,
        exemples_corriges=None,
        matiere=None  # NOUVEAU : on passe maintenant la Matiere
):

    if lecons_contenus is None:
        lecons_contenus = []
    if exemples_corriges is None:
        exemples_corriges = []

    # ---- PROMPT IA personnalisable par matière -----
    system_prompt = ""
    exemple_prompt = ""
    consignes_finales = ""

    DEFAULT_SYSTEM_PROMPT = """
Corrige cet exercice comme un expert discipliné, étape par étape ...
(mets par défaut ta consigne la plus générique ici)
"""
    DEFAULT_EXEMPLE_PROMPT = """
(mets un exemple très générique ici, qui s'affichera si prompt IA n'est pas custom)
"""
    DEFAULT_CONSIGNES_FINALES = """
Structure : Méthode → Calcul → [Réponse]. Tex Latex...
"""

    # On cherche s'il y a un prompt custom matière
    if matiere and hasattr(matiere, 'prompt_ia'):
        promptia = matiere.prompt_ia
        system_prompt = promptia.system_prompt or DEFAULT_SYSTEM_PROMPT
        exemple_prompt = promptia.exemple_prompt or DEFAULT_EXEMPLE_PROMPT
        consignes_finales = promptia.consignes_finales or DEFAULT_CONSIGNES_FINALES
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        exemple_prompt = DEFAULT_EXEMPLE_PROMPT
        consignes_finales = DEFAULT_CONSIGNES_FINALES

    lecons = [f"### {t}\n{c}" for t, c in lecons_contenus[:3]]
    exemples = [e for e in exemples_corriges[:2]]

    # -- PROMPT CONSTRUIT SUR-MESURE --
    prompt_ia = (
        "### CONTEXTE DU COURS\n"
        f"{contexte}\n\n"
        "### LEÇONS UTILES\n"
        f"{chr(10).join(lecons) if lecons else 'Aucune leçon supplémentaire'}\n\n"
        "### EXEMPLES DE CORRIGÉS\n"
        f"{exemple_prompt if exemple_prompt else (chr(10).join(exemples) if exemples else 'Aucun exemple fourni')}\n\n"
        "### EXERCICE À CORRIGER\n"
        f"{texte_enonce.strip()}\n\n"
        "### CONSIGNES FINALES\n"
        f"{consignes_finales}"
    )

    # Appel à l'API DeepSeek-R1...
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_ia}
        ],
        "temperature": 0.12,
        "max_tokens": 3000,
        "top_p": 0.3,
        "frequency_penalty": 0.2
    }
    # Configuration DeepSeek (API clé/config)
    api_key = os.getenv('DEEPSEEK_API_KEY')  # Vérifie que ta variable est bien définie dans .env ou settings
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=100)
        response_data = response.json()

        if response.status_code != 200:
            error_msg = f"Erreur API DeepSeek [{response.status_code}]: {response_data.get('message', 'Pas de détail')}"
            print(error_msg)
            return error_msg, None

        output = response_data['choices'][0]['message']['content']
        print("=== DEEPSEEK OUTPUT ===\n", output)

        # Traitement des graphiques (identique à avant)
        regex_all_json = _re.findall(r'(\{\s*"graphique"\s*:\s*\{[\s\S]+?\}\s*\})', output)

        if regex_all_json:
            graph_list = []
            corrige_txt = output
            for idx, found_json in enumerate(regex_all_json, 1):
                try:
                    sjson = found_json.replace("'", '"').replace('\n', '').replace('\r', '').strip()
                    graph_dict = json.loads(sjson)
                    corrige_txt = corrige_txt.replace(found_json, f"\n[[GRAPHIC_{idx}]]\n", 1)
                    graph_list.append(graph_dict)
                except Exception as e:
                    print("Erreur parsing JSON:", e)
                    continue
            return corrige_txt.strip(), graph_list

        return output.strip(), None

    except Exception as e:
        return f"Erreur API: {str(e)}", None


@shared_task(name='correction.ia_utils.generer_corrige_ia_et_graphique_async')
def generer_corrige_ia_et_graphique_async(
    texte_enonce,
    contexte,
    lecons_contenus=None,
    exemples_corriges=None,
    matiere_id=None,
    demande_id=None
):
    print("--- Task celery appelée ! ---")
    print("texte_enonce:", texte_enonce)
    print("contexte:", contexte)
    print("lecons_contenus:", lecons_contenus)
    print("exemples_corriges:", exemples_corriges)
    print("matiere_id:", matiere_id)
    print("demande_id:", demande_id)
    from resources.models import Matiere
    from correction.models import DemandeCorrection
    matiere = Matiere.objects.get(id=matiere_id) if matiere_id else None
    corrige_txt, graph_list = generer_corrige_ia_et_graphique(
        texte_enonce,
        contexte,
        lecons_contenus=lecons_contenus,
        exemples_corriges=exemples_corriges,
        matiere=matiere
    )
    demande = DemandeCorrection.objects.get(id=demande_id)
    demande.corrigé = corrige_txt
    demande.save()