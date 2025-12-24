import os
import subprocess
import tempfile

import pdfkit
from django.conf import settings
from django.template.loader import render_to_string
try:
    from PyPDF2 import PdfMerger
except ImportError:
    from pypdf import PdfMerger

import logging
logger = logging.getLogger(__name__)
from urllib.parse import urlparse



def prerender_mathjax(html: str) -> str:
    """
    Passe le HTML dans mjpage (via npx) pour exécuter MathJax et produire du HTML dont les formules sont rendues en SVG.
    """
    # 1) Temp file entrée
    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_in.write(html.encode("utf-8"))
    tf_in.close()

    # 2) Temp file sortie
    tf_out = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_out.close()

    # 3) Appel en pipe : cat input | mjpage > output
    print("DEBUG ▶️ Lancement de npx mjpage pour pré-rendu…")
    cmd = f"npx mjpage < {tf_in.name} > {tf_out.name}"
    subprocess.check_call(cmd, shell=True)
    print("DEBUG ✔️ Pré-rendu MathJax terminé.")

    # 4) Lecture du HTML rendu
    with open(tf_out.name, encoding="utf-8") as f:
        rendered = f.read()

    return rendered


def generer_pdf_corrige(context: dict, soumission_id: int) -> str:
    """
    1) Render du template
    2) Prérendu MathJax
    3) Génération PDF via pdfkit/wkhtmltopdf
    4) Sauvegarde et URL
    """
    # 1) HTML initial
    html = render_to_string("correction/corrige_view.html", context)

    # 2) Prérendu MathJax
    html_prerender = prerender_mathjax(html)

    # DEBUG : Sauvegarde du HTML intermédiaire dans un fichier temporaire à examiner
    debug_html_path = f"/tmp/dernier_corrige_{soumission_id}.html"
    with open(debug_html_path, "w", encoding="utf-8") as f:
        f.write(html_prerender)
    print(f"DEBUG : HTML intermédiaire sauvegardé à {debug_html_path}")

    # 3) Conversion en PDF
    options = {
        "enable-local-file-access": None,
        "print-media-type": None,
        "no-stop-slow-scripts": None,
        "javascript-delay": "8000",  # le temps
        "zoom": "3",  # Zoom pour agrandir le contenu
        "disable-smart-shrinking": "",  # IMPORTANT : Évite le rétrécissement
        "dpi": 300,  # Meilleure résolution
        "page-size": "A4",
        "margin-top": "15mm",
        "margin-bottom": "15mm",
        "margin-left": "10mm",
        "margin-right": "10mm",
        "quiet": "",
    }

    pdf_bytes = pdfkit.from_string(html_prerender, False, options=options)

    # 4) Sauvegarde et URL
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    fname = f"corrige_{soumission_id}.pdf"
    path = os.path.join(pdf_dir, fname)

    with open(path, "wb") as f:
        f.write(pdf_bytes)

    return settings.MEDIA_URL + "pdfs/" + fname



def merge_pdfs(pdf_urls: list, output_name: str) -> str:
    merger = PdfMerger()
    logger.debug(f"[merge_pdfs] pdf_urls reçues : {pdf_urls}")

    chemins_trouves = []
    for url in pdf_urls:
        # 1) Extraire le chemin relatif depuis MEDIA_URL
        if url.startswith('http'):
            # si url absolue : ne garder que le path
            path = urlparse(url).path
        else:
            path = url
        rel = path.replace(settings.MEDIA_URL, '').lstrip('/')
        fpath = os.path.join(settings.MEDIA_ROOT, rel)

        # 2) Vérifier l’existence
        if not os.path.exists(fpath):
            logger.warning(f"[merge_pdfs] Fichier introuvable : {fpath}")
            continue

        # 3) Ajouter au merger
        chemins_trouves.append(fpath)
        merger.append(fpath)

    # 4) Cas sans aucun PDF à fusionner
    if not chemins_trouves:
        raise Exception("merge_pdfs: aucun fichier valide trouvé pour fusion")

    # 5) Écriture du PDF global
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, output_name)

    merger.write(out_path)
    merger.close()

    logger.debug(f"[merge_pdfs] Fusion OK, fichier généré : {out_path}")
    return settings.MEDIA_URL + "pdfs/" + output_name

