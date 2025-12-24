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
    logger.info("[prerender_mathjax] Début du pré-rendu MathJax")
    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_in.write(html.encode("utf-8"))
    tf_in.close()
    logger.info(f"[prerender_mathjax] Fichier d'entrée : {tf_in.name}")

    tf_out = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_out.close()
    logger.info(f"[prerender_mathjax] Fichier de sortie : {tf_out.name}")

    cmd = f"npx mjpage < {tf_in.name} > {tf_out.name}"
    logger.info(f"[prerender_mathjax] Exécution de la commande : {cmd}")
    subprocess.check_call(cmd, shell=True)
    logger.info("[prerender_mathjax] Pré-rendu terminé")

    with open(tf_out.name, encoding="utf-8") as f:
        return f.read()


def generer_pdf_corrige(context: dict, soumission_id: int) -> str:
    logger.info(f"[generer_pdf_corrige] Début génération du PDF pour la soumission #{soumission_id}")

    html = render_to_string("correction/corrige_view.html", context)
    logger.info(f"[generer_pdf_corrige] Template rendu ({len(html)} caractères)")

    html_prerender = prerender_mathjax(html)
    debug_html_path = f"/tmp/dernier_corrige_{soumission_id}.html"
    with open(debug_html_path, "w", encoding="utf-8") as f:
        f.write(html_prerender)
    logger.info(f"[generer_pdf_corrige] HTML intermédiaire sauvé : {debug_html_path}")

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
    logger.info("[generer_pdf_corrige] Conversion du HTML en PDF")
    pdf_bytes = pdfkit.from_string(html_prerender, False, options=options)
    logger.info("[generer_pdf_corrige] Conversion terminée")

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
    """
    logger.info(f"[merge_pdfs] Début fusion de {len(pdf_urls)} PDF(s) : {pdf_urls}")
    merger = PdfMerger()
    chemins_trouves = []

    for url in pdf_urls:
        # Extraction du chemin local
        if url.startswith('http'):
            path = urlparse(url).path
        else:
            path = url
        rel = path.replace(settings.MEDIA_URL, '').lstrip('/')
        fpath = os.path.join(settings.MEDIA_ROOT, rel)
        logger.info(f"[merge_pdfs] URL reçue : {url} → chemin local : {fpath}")

        # Vérification d’existence
        if not os.path.exists(fpath):
            logger.warning(f"[merge_pdfs] Fichier introuvable : {fpath}")
            continue

        # Ajout au merger
        try:
            merger.append(fpath)
            chemins_trouves.append(fpath)
            logger.info(f"[merge_pdfs] Ajouté au merger : {fpath}")
        except Exception:
            logger.exception(f"[merge_pdfs] Échec ajout du PDF : {fpath}")
            raise

    if not chemins_trouves:
        logger.error("[merge_pdfs] Aucun fichier valide trouvé pour la fusion – arrêt")
        raise FileNotFoundError("merge_pdfs: aucun fichier valide trouvé pour fusion")

    # Écriture du PDF global
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, output_name)
    try:
        merger.write(out_path)
        merger.close()
        logger.info(f"[merge_pdfs] Fusion OK, fichier généré : {out_path}")
    except Exception:
        logger.exception(f"[merge_pdfs] Échec écriture du PDF global : {out_path}")
        raise

    return settings.MEDIA_URL + "pdfs/" + output_name