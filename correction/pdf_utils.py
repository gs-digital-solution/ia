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

from urllib.parse import urlparse


def prerender_mathjax(html: str) -> str:
    print("▶▶ prerender_mathjax: début")
    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_in.write(html.encode("utf-8"))
    tf_in.close()
    print(f"    fichier entrée  : {tf_in.name}")

    tf_out = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_out.close()
    print(f"    fichier sortie : {tf_out.name}")

    cmd = f"npx mjpage < {tf_in.name} > {tf_out.name}"
    print(f"    lancement : {cmd}")
    subprocess.check_call(cmd, shell=True)
    print("✔ prerender_mathjax terminé")

    with open(tf_out.name, encoding="utf-8") as f:
        return f.read()


def generer_pdf_corrige(context: dict, soumission_id: int) -> str:
    print(f"▶▶ generer_pdf_corrige: soumission #{soumission_id}")
    html = render_to_string("correction/corrige_view.html", context)
    print(f"    template rendu, {len(html)} caractères")

    html_prerender = prerender_mathjax(html)
    debug_html = f"/tmp/dernier_corrige_{soumission_id}.html"
    with open(debug_html, "w", encoding="utf-8") as f:
        f.write(html_prerender)
    print(f"    HTML intermédiaire sauvegardé : {debug_html}")

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
    print("    conversion HTML → PDF")
    pdf_bytes = pdfkit.from_string(html_prerender, False, options=options)
    print("✔ PDF généré en mémoire")

    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    fname = f"corrige_{soumission_id}.pdf"
    path = os.path.join(pdf_dir, fname)
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    print(f"✔ PDF sauvé : {path}")

    return settings.MEDIA_URL + "pdfs/" + fname


def merge_pdfs(pdf_urls: list, output_name: str) -> str:
    print("▶▶ merge_pdfs appelé")
    print("    MEDIA_URL   =", settings.MEDIA_URL)
    print("    MEDIA_ROOT  =", settings.MEDIA_ROOT)
    print("    pdf_urls    =", pdf_urls)

    try:
        merger = PdfMerger()
        print("    PdfMerger instancié")
    except Exception as e:
        print("‼ Erreur instanciation PdfMerger:", e)
        raise

    chemins_trouves = []
    for url in pdf_urls:
        print("    traiter url:", url)
        if url.startswith("http"):
            path = urlparse(url).path
        else:
            path = url
        print("        path :", path)

        rel = path.replace(settings.MEDIA_URL, "").lstrip("/")
        print("        rel  :", rel)

        fpath = os.path.join(settings.MEDIA_ROOT, rel)
        print("        fpath:", fpath)

        exists = os.path.exists(fpath)
        print("        existe ?", exists)
        if not exists:
            print("    ⚠ fichier introuvable, skip.")
            continue

        try:
            merger.append(fpath)
            chemins_trouves.append(fpath)
            print("    + ajouté au merger")
        except Exception as e:
            print("‼ Erreur ajout PDF:", fpath, e)
            raise

    print("    fichiers valides:", chemins_trouves)
    if not chemins_trouves:
        print("‼ Aucun fichier valide à fusionner, on lève FileNotFoundError")
        raise FileNotFoundError("merge_pdfs: aucun fichier valide trouvé pour fusion")

    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, output_name)
    try:
        merger.write(out_path)
        merger.close()
        print("✔ fusion OK, fichier généré :", out_path)
    except Exception as e:
        print("‼ Erreur écriture PDF global :", e)
        raise

    return settings.MEDIA_URL + "pdfs/" + output_name