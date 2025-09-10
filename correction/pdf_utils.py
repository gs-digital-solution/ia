import os
import subprocess
import tempfile

import pdfkit
from django.conf import settings
from django.template.loader import render_to_string


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

    # 3) Conversion en PDF
    options = {
        "enable-local-file-access": None,
        "print-media-type": None,
        "no-stop-slow-scripts": None,
        "javascript-delay": "3000",  # <-- plus de temps (3s)
        "margin-top": "10mm",
        "margin-bottom": "10mm",
        "margin-left": "8mm",
        "margin-right": "8mm",
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