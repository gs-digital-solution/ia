import os
from datetime import datetime
from django.conf import settings
from weasyprint import HTML
from .ia_utils import tracer_graphique, convertir_latex_vers_html


def generer_pdf_corrige(corrige_text, graphiques, demande):
    # Convertir LaTeX en HTML avec MathJax
    corrige_html = convertir_latex_vers_html(corrige_text)

    # Générer les graphiques
    graphiques_html = ""
    if graphiques:
        for i, graphique in enumerate(graphiques):
            nom_fichier = f"graph_{demande.id}_{i}_{datetime.now().strftime('%H%M%S')}.png"
            chemin_graphique = tracer_graphique(graphique, nom_fichier)
            if chemin_graphique:
                graphiques_html += f'<img src="{settings.MEDIA_ROOT}/{chemin_graphique}" style="max-width: 100%; margin: 20px 0;">'

    # HTML avec MathJax
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Corrigé CIS - {demande.matiere.nom}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js"></script>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 40px;
                line-height: 1.6;
            }}
            h1 {{ color: #2E7D32; }}
            .reponse {{ 
                background: #f0f8f0; 
                padding: 15px; 
                border-left: 4px solid #2E7D32;
                margin: 20px 0;
                border-radius: 5px;
            }}
            img {{ 
                max-width: 100%; 
                height: auto;
                display: block;
                margin: 20px auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            table {{
                border-collapse: collapse;
                margin: 15px 0;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            th {{
                background-color: #2E7D32;
                color: white;
            }}
            .math {{
                font-size: 1.1em;
            }}
        </style>
    </head>
    <body>
        <h1>📚 Corrigé CIS - {demande.matiere.nom}</h1>
        <p><strong>Date:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        <p><strong>Matière:</strong> {demande.matiere.nom}</p>
        <p><strong>Classe:</strong> {demande.classe.nom if demande.classe else ''}</p>
        <hr>

        <div class="math">
            {corrige_html.replace('\n', '<br>')}
        </div>

        {graphiques_html}

        <script>
            MathJax = {{
                tex: {{
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']]
                }},
                svg: {{
                    fontCache: 'global'
                }}
            }};
        </script>
    </body>
    </html>
    """

    # Générer le PDF
    nom_fichier = f"corrige_{demande.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    dossier_pdf = os.path.join(settings.MEDIA_ROOT, 'corriges')
    os.makedirs(dossier_pdf, exist_ok=True)
    chemin_pdf = os.path.join(dossier_pdf, nom_fichier)

    HTML(string=html_content).write_pdf(chemin_pdf)

    return f'/media/corriges/{nom_fichier}'