import tempfile
from pdfminer.high_level import extract_text
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import openai
import json
import os

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
    x_min = float(graphique_dict.get("x_min", -2))
    x_max = float(graphique_dict.get("x_max", 4))
    expression = graphique_dict.get("expression", "x")
    titre = graphique_dict.get("titre", "Graphique à tracer")
    x = np.linspace(x_min, x_max, 400)
    try:
        y = eval(expression, {'x': x, 'np': np, '__builtins__': None})
    except Exception as e:
        print("Erreur tracé:", e)
        return None
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color="#008060")
    plt.title(titre)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    from django.conf import settings
    dossier = os.path.join(settings.MEDIA_ROOT, "graphes")
    os.makedirs(dossier, exist_ok=True)
    chemin_png = os.path.join(dossier, output_name)
    plt.savefig(chemin_png)
    plt.close()
    return "graphes/" + output_name

def generer_corrige_ia_et_graphique(
        texte_enonce,
        contexte,
        lecons_contenus=[],  # Listes de tuples (titre, contenu)
        exemples_corriges=[]  # Listes de corrigés du même type
):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    prompt_ia = (
        "Voici le CONTEXTE pédagogique et local de l'utilisateur :\n"
        f"{contexte}\n\n"
        "OBJECTIF : Donne un corrigé détaillé, étape par étape, STRICTEMENT adapté à ce contexte (niveau, pays, sous-système, matière, type d'exercice, ...).\n"
        "---\n"
        "EXEMPLES DE CORRIGÉS SIMILAIRES :\n"
    )
    if exemples_corriges:
        for ex_ind, ex_corr in enumerate(exemples_corriges, 1):
            prompt_ia += f"\n— Exemple corrigé {ex_ind} :\n{ex_corr}\n"
    else:
        prompt_ia += "\nAucun exemple similaire disponible dans la base.\n"

    if lecons_contenus:
        prompt_ia += "\nLeçon(s) pédagogique(s) à exploiter/insister si pertinent :\n"
        for titre, contenu in lecons_contenus:
            prompt_ia += f"\n>> {titre} : {contenu}\n"
        prompt_ia += "\nIntègre ces leçons dans ta solution si elles sont nécessaires."

    prompt_ia += (
        "\n---\nÉNONCÉ À CORRIGER :\n" + texte_enonce +
        "\n---\n"
        "Consignes de RÉDACTION :\n"
        "- Correction détaillée, pédagogique, structurée étape par étape, adaptée au pays et au niveau, claire et sans phrases inutiles.\n"
        "- Si un graphique (courbe, histogramme, etc.) doit être produit, ajoute à la fin de la réponse le JSON suivant sur UNE SEULE LIGNE (sinon mets juste {}):\n"
        '{"graphique":{"type":"fonction","expression":"...","titre":"...","x_min":"...","x_max":"..."}}\n'
        "- Pour toutes les formules, utilises impérativement \\( ... \\) (en ligne) ou \\[ ... \\] (affichées). N'utilises jamais $...$.\n"
        "- SEPARATEUR OBLIGATOIRE : après ta correction, écris sur une NOUVELLE ligne\n"
        "---corrigé---\n"
        "- Puis écris le JSON EXACTEMENT sur une ligne. Ne mets jamais de phrase ou de commentaire en dehors de la correction ou du JSON."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Tu es un super correcteur pédagogique expert, qui répond TOUJOURS en français, strictement selon le contexte, les leçons et les exemples, et qui respecte toutes les consignes."
            },
            {
                "role": "user",
                "content": prompt_ia
            }
        ],
        temperature=0.18,
        max_tokens=1900
    )

    output = response.choices[0].message.content

    # Séparation corrigé/JSON (retourne TOUJOURS DEUX VALEURS)
    if "---corrigé---" in output:
        corrigé, part_json = output.rsplit("---corrigé---", 1)
        try:
            graphique_dict = json.loads(part_json.strip())
        except Exception:
            graphique_dict = {}
    else:
        corrigé = output
        graphique_dict = {}

    return corrigé.strip(), graphique_dict.get("graphique", None)