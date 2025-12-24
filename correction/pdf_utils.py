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
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def prerender_mathjax(html: str) -> str:
    """
    Passe le HTML dans mjpage (via npx) pour exécuter MathJax
    et produire du HTML dont les formules sont rendues en SVG.
    """
    logger.info("[prerender_mathjax] Début du pré-rendu MathJax")
    # 1) Temp file entrée
    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_in.write(html.encode("utf-8"))
    tf_in.close()
    logger.info(f"[prerender_mathjax] Fichier d'entrée : {tf_in.name}")

    # 2) Temp file sortie
    tf_out = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_out.close()
    logger.info(f"[prerender_mathjax] Fichier de sortie : {tf_out.name}")

    # 3) Appel en pipe : cat input | mjpage > output
    cmd = f"npx mjpage < {tf_in.name} > {tf_out.name}"
    logger.info(f"[prerender_mathjax] Exécution de la commande : {cmd}")
    subprocess.check_call(cmd, shell=True)
    logger.info("[prerender_mathjax] Pré-rendu MathJax terminé")

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
    logger.info(f"[generer_pdf_corrige] Début génération du PDF corrigé pour la soumission #{soumission_id}")

    # 1) HTML initial
    html = render_to_string("correction/corrige_view.html", context)
    logger.info(f"[generer_pdf_corrige] Template rendu ({len(html)} caractères)")

    # 2) Prérendu MathJax
    html_prerender = prerender_mathjax(html)

    # DEBUG : Sauvegarde du HTML intermédiaire dans un fichier temporaire à examiner
    debug_html_path = f"/tmp/dernier_corrige_{soumission_id}.html"
    with open(debug_html_path, "w", encoding="utf-8") as f:
        f.write(html_prerender)
    logger.info(f"[generer_pdf_corrige] HTML intermédiaire sauvegardé à {debug_html_path}")

    # 3) Conversion en PDF
    options = {
        "enable-local-file-access": None,
        "print-media-type": None,
        "no-stop-slow-scripts": None,
        "javascript-delay": "8000",
        "zoom": "3",
        "disable-smart-shrinking": "",
        "dpi": 300,
        "page-size": "A4",
        "margin-top": "15mm",
        "margin-bottom": "15mm",
        "margin-left": "10mm",
        "margin-right": "10mm",
        "quiet": "",
    }
    logger.info("[generer_pdf_corrige] Conversion du HTML en PDF avec pdfkit")
    pdf_bytes = pdfkit.from_string(html_prerender, False, options=options)
    logger.info("[generer_pdf_corrige] Conversion terminée")

    # 4) Sauvegarde et URL
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    fname = f"corrige_{soumission_id}.pdf"
    path = os.path.join(pdf_dir, fname)

    with open(path, "wb") as f:
        f.write(pdf_bytes)
    logger.info(f"[generer_pdf_corrige] PDF enregistré à {path}")

    return settings.MEDIA_URL + "pdfs/" + fname


def merge_pdfs(pdf_urls: list, output_name: str) -> str:
    """
    Fusionne plusieurs fichiers PDF en un seul.
    pdf_urls : liste d'URLs (ou chemins relatifs) pointant vers les PDF à fusionner.
    output_name : nom du fichier PDF de sortie (ex : "global_42.pdf").
    """
    logger.info(f"[merge_pdfs] Début fusion de {len(pdf_urls)} PDF(s)")
    merger = PdfMerger()

    chemins_trouves = []
    for url in pdf_urls:
        # 1) Extraire le chemin relatif depuis MEDIA_URL
        if url.startswith('http'):
            path = urlparse(url).path
        else:
            path = url
        rel = path.replace(settings.MEDIA_URL, '').lstrip('/')
        fpath = os.path.join(settings.MEDIA_ROOT, rel)
        logger.info(f"[merge_pdfs] URL reçue : {url} → chemin local : {fpath}")

        # 2) Vérifier l’existence
        if not os.path.exists(fpath):
            logger.warning(f"[merge_pdfs] Fichier introuvable : {fpath}")
            continue

        # 3) Ajouter au merger
        chemins_trouves.append(fpath)
        merger.append(fpath)

    # 4) Cas sans aucun PDF à fusionner
    if not chemins_trouves:
        logger.error("[merge_pdfs] Aucun fichier valide trouvé pour la fusion – interruption")
        raise Exception("merge_pdfs: aucun fichier valide trouvé pour fusion")

    # 5) Écriture du PDF global
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, output_name)

    merger.write(out_path)
    merger.close()
    logger.info(f"[merge_pdfs] Fusion OK, fichier généré : {out_path}")

    return settings.MEDIA_URL + "pdfs/" + output_name