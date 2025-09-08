import os
import subprocess
import tempfile
import pdfkit
from django.conf import settings
from django.template.loader import render_to_string

def prerender_mathjax(html: str) -> str:
    """
    Passe le HTML dans mathjax-node-page (via npx) pour exécuter MathJax
    et produire du HTML dont les formules sont rendues en SVG.
    """
    # Fichier temporaire d'entrée
    tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_in.write(html.encode("utf-8"))
    tf_in.close()

    # Fichier temporaire de sortie
    tf_out = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tf_out.close()

    # DEBUG avant l’appel
    print("DEBUG ▶️ Lancement de npx mathjax-node-page pour pré-rendu…")
    subprocess.check_call([
        "npx", "mathjax-node-page",
        tf_in.name,
        tf_out.name,
        "--svg",
        "--output-format", "html"
    ])
    # DEBUG après l’appel
    print("DEBUG ✔️ Pré-rendu MathJax terminé.")

    # Lecture du HTML rendu
    with open(tf_out.name, encoding="utf-8") as f:
        rendered = f.read()
    return rendered


def generer_pdf_corrige(context: dict, soumission_id: int) -> str:
    """
    Génère un PDF fidèle à partir du template + prerendu MathJax.
    Retourne l'URL publique du PDF.
    """
    # 1) Rendu HTML initial
    html = render_to_string("correction/corrige_view.html", context)

    # 2) Prérendu MathJax
    html_prerender = prerender_mathjax(html)

    # 3) Conversion en PDF via wkhtmltopdf/pdfkit
    options = {
        "enable-local-file-access": None,
        "margin-top":    "10mm",
        "margin-bottom": "10mm",
        "margin-left":   "8mm",
        "margin-right":  "8mm",
    }
    pdf_bytes = pdfkit.from_string(html_prerender, False, options=options)

    # 4) Sauvegarde sur disque
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    fname = f"corrige_{soumission_id}.pdf"
    path = os.path.join(pdf_dir, fname)
    with open(path, "wb") as f:
        f.write(pdf_bytes)

    # 5) Renvoi de l'URL publique
    return settings.MEDIA_URL + "pdfs/" + fname