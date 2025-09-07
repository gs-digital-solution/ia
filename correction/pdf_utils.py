import os
import subprocess
import tempfile

import pdfkit
from django.conf import settings
from django.template.loader import render_to_string


def prerender_mathjax(html: str) -> str:
    """
    Passe le HTML dans mathjax-node-page via npx pour obtenir un HTML où les formules LaTeX sont remplacées par du SVG.
    """
    # 1) Écrire le HTML brut dans un fichier temporaire
    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_in.write(html.encode("utf-8"))
    tf_in.close()

    # 2) Créer le fichier temporaire de sortie
    tf_out = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_out.close()

    # 3) Appeler mathjax-node-page via npx
    subprocess.check_call([
        "npx", "mathjax-node-page", tf_in.name, tf_out.name,
        "--svg", "--output-format", "html"
    ])

    # 4) Lire et retourner le HTML rendu
    with open(tf_out.name, encoding="utf-8") as f:
        rendered = f.read()

    return rendered


def generer_pdf_corrige(context: dict, soumission_id: int) -> str:
    """
    Génère le PDF à partir du template + prerender MathJax + pdfkit.
    Renvoie l'URL publique du fichier PDF.
    """
    # 1) Rend le template initial
    html = render_to_string("correction/corrige_view.html", context)

    # 2) Prérend le HTML (formules en SVG)
    html_prerender = prerender_mathjax(html)

    # 3) Convertit en PDF avec wkhtmltopdf via pdfkit
    options = {
        "enable-local-file-access": None,
        "margin-top": "10mm",
        "margin-bottom": "10mm",
        "margin-left": "8mm",
        "margin-right": "8mm",
    }
    pdf_bytes = pdfkit.from_string(html_prerender, False, options=options)

    # 4) Sauve le PDF dans MEDIA_ROOT/pdfs/
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    fname = f"corrige_{soumission_id}.pdf"
    path = os.path.join(pdf_dir, fname)

    with open(path, "wb") as f:
        f.write(pdf_bytes)

    # 5) Retourne l'URL publique
    return settings.MEDIA_URL + "pdfs/" + fname