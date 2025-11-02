import requests
import os
import tempfile
import json
import re
import numpy as np
import matplotlib
import openai
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings
from django.utils.safestring import mark_safe
from celery import shared_task
import base64

# Récupérer ta clé OpenAI et initialiser le client
openai.api_key = os.getenv("OPENAI_API_KEY")

# ========== EXTRACTION DE L'ÉNONCE AVEC GPT-4 AVANT DE PASSER A DEEPSEEK ==========
def extraire_texte_gpt4(fichier_field):
    """
    Envoie le fichier (image ou PDF) à GPT-4 Vision pour en extraire
    l'énoncé, les formules (en LaTeX) et les tableaux (en markdown).
    """
    # 1) Récupération du chemin local du fichier (Django)
    try:
        fichier_local = fichier_field.path
    except AttributeError:
        # Si .path n’existe pas, on recrée un fichier temporaire
        fichier_local = os.path.join(
            tempfile.gettempdir(),
            os.path.basename(fichier_field.name)
        )
        with open(fichier_local, "wb") as f:
            for chunk in fichier_field.chunks():
                f.write(chunk)

    # 2) Encodage en base64 + détection du MIME
    ext = os.path.splitext(fichier_local)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf"
    }.get(ext, "application/octet-stream")
    with open(fichier_local, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # 3) Construction du prompt multimodal
    prompt = (
        "Tu es un outil d'extraction. Je t'envoie le document encodé "
        "en base64. Rends-moi tout le texte, les formules en LaTeX "
        "et les tableaux en markdown.\n\n"
        f"data:{mime};base64,{b64}"
    )

    # 4) Appel à l'API GPT-4 Vision (avec logs précis)
    try:
        # ←– AJOUT #1 : avant l’appel, indique que l’on démarre GPT-4 Vision
        print("⚙️ Appel à OpenAI GPT-4 Vision…")

        resp = openai.ChatCompletion.create(
            model="gpt-4",   # ou le modèle GPT-4 auquel vous avez accès
            messages=[
                {"role": "system", "content": "Extrait le contenu du document."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
        )

        # ←– AJOUT #2 : juste après l’appel, on logge le nom du modèle utilisé
        print("✅ Réponse OpenAI reçue. Modèle utilisé :", getattr(resp, "model", "inconnu"))

        extrait = resp.choices[0].message.content.strip()

    except Exception as e:
        # ←– EXISTANT / AMÉLIORÉ : on logge l’erreur détaillée
        print("❌ Erreur GPT-4 Vision lors de l'extraction :", e)
        # pour ne pas masquer l’erreur en amont
        raise

    # 5) (Optionnel) suppression du fichier temporaire si on l’a créé
    # try: os.remove(fichier_local)
    # except: pass

    return extrait

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
        # Nouveaux genres d'épreuves (langues, lettres, geo, etc.)
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

    ### CONSIGNES STRICTES - À RESPECTER IMPÉRATIVEMENT
    {consignes_finales}

    **EXIGENCES ABSOLUES :**
    1. Sois EXTRÊMEMENT RIGOUREUX dans tous les calculs
    2. Vérifie systématiquement chaque résultat intermédiaire  
    3. Donne TOUTES les étapes de calcul détaillées
    4. Les réponses doivent être NUMÉRIQUEMENT EXACTES
    5. Ne laisse AUCUNE question sans réponse complète
    6. Si l'énoncé semble ambigu, prends l'interprétation mathématique standard

    **FORMAT DE RÉPONSE :**
    - Réponses complètes avec justification
    - Calculs intermédiaires détaillés
    - Solutions numériques exactes
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
        print("📡 Appel API DeepSeek pour exercice...")

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
                    "content"] += "\n\n⚠️ ATTENTION : Sois plus rigoureux ! Vérifie tous tes calculs. Donne des réponses complètes et exactes. Ne laisse aucune question sans réponse."

                if tentative == 0:  # Attendre un peu avant la 2ème tentative
                    import time
                    time.sleep(2)
        else:
            print("❌ Échec après 2 tentatives - qualité insuffisante")
            return "Erreur: Qualité du corrigé insuffisante après plusieurs tentatives", None

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
                # Mettre à jour l'offset en fonction de la différence de longueur
                offset += len(img_tag) - (end - start)

                graphs_data.append(graph_dict)
                print(f"✅ Graphique {idx} inséré.")
            else:
                # En cas d'échec de tracé, on remplace par un message
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

    # noms d'axes
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
DEFAULT_SYSTEM_PROMPT = r""" résous ce sujet
si une question demande un graphite alors trace-le
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
    if tokens_estimes < 1500:  # Épreuve courte
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

        # 1) Extraction initiale PAR GPT-4 Vision
        if demande.fichier:
            print("🔍 Extraction via GPT-4 Vision…")
            texte_brut = extraire_texte_gpt4(demande.fichier)
        else:
            texte_brut = demande.enonce_texte or ""

        print("📥 DEBUG – TEXTE BRUT (premiers 500 chars) :")
        print(texte_brut[:500].replace("\n", "\\n"), "...\n")

        # 2) Assemblage du texte final pour l'IA
        texte_enonce = texte_brut

        print("📥 DEBUG – TEXTE ENRICHI :")
        print(texte_enonce[:500].replace("\n", "\\n"), "...\n")

        soumission.statut = 'analyse_ia'
        soumission.progression = 40
        soumission.save()

        matiere = Matiere.objects.get(id=matiere_id) if matiere_id else demande.matiere
        contexte = f"Exercice de {matiere.nom} - {demande.classe.nom if demande.classe else ''}"

        soumission.statut = 'generation_graphiques'
        soumission.progression = 60
        soumission.save()

        # plus de nettoyage GPT : on passe directement l'extraction à Deepseek
        texte_pret = texte_enonce
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