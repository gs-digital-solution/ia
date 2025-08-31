import os
from datetime import datetime
from django.conf import settings
from weasyprint import HTML
from .ia_utils import convertir_latex_vers_html, tracer_graphique
import re


def generer_pdf_corrige(corrige_text, graphiques, demande):
    # 1. Convertir LaTeX en HTML avec MathJax
    corrige_html = convertir_latex_vers_html(corrige_text)

    # 2. Nettoyer le contenu - supprimer les répétitions d'énoncés
    corrige_html = _nettoyer_contenu(corrige_html)

    # 3. Générer les graphiques
    graphiques_html = ""
    if graphiques:
        for i, graphique in enumerate(graphiques):
            nom_fichier = f"graph_{demande.id}_{i}_{datetime.now().strftime('%H%M%S')}.png"
            chemin_graphique = tracer_graphique(graphique, nom_fichier)
            if chemin_graphique:
                graphiques_html += f'<img src="{settings.MEDIA_ROOT}/{chemin_graphique}" style="max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 8px;">'

    # 4. HTML avec améliorations
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Corrigé CIS - {demande.matiere.nom}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Source+Serif+Pro:wght@400;600;700&display=swap');

            body {{ 
                font-family: 'Source Serif Pro', serif;
                margin: 30px;
                line-height: 1.7;
                color: #2c3e50;
                font-size: 18px; /* Taille augmentée */
            }}

            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #2E7D32;
            }}

            .header h1 {{ 
                color: #2E7D32;
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 10px;
                font-family: 'Roboto', sans-serif;
            }}

            .header p {{
                color: #7f8c8d;
                font-size: 16px;
                margin: 5px 0;
                font-family: 'Roboto', sans-serif;
            }}

            .app-promo {{
                background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
                font-family: 'Roboto', sans-serif;
            }}

            .app-promo a {{
                color: #FFEB3B;
                text-decoration: none;
                font-weight: 500;
            }}

            .app-promo a:hover {{
                text-decoration: underline;
            }}

            .metadata {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #2E7D32;
                font-family: 'Roboto', sans-serif;
            }}

            .metadata p {{
                margin: 8px 0;
                font-size: 16px;
            }}

            .metadata strong {{
                color: #2E7D32;
            }}

            .corrige-content {{
                margin-top: 30px;
            }}

            .corrige-content h2 {{
                color: #2E7D32;
                font-size: 22px;
                border-bottom: 1px solid #ecf0f1;
                padding-bottom: 10px;
                margin-top: 30px;
                font-family: 'Roboto', sans-serif;
            }}

            .corrige-content h3 {{
                color: #27ae60;
                font-size: 20px;
                margin-top: 25px;
                font-family: 'Roboto', sans-serif;
            }}

            .reponse {{ 
                background: #f0f8f0; 
                padding: 20px; 
                border-left: 4px solid #2E7D32;
                margin: 25px 0;
                border-radius: 8px;
            }}

            .reponse h4 {{
                color: #2E7D32;
                margin-top: 0;
                font-size: 18px;
            }}

            .method {{ 
                background: #e8f4f8; 
                padding: 15px; 
                border-left: 4px solid #2980b9;
                margin: 20px 0;
                border-radius: 8px;
            }}

            .calcul {{ 
                background: #fff3e0; 
                padding: 15px; 
                border-left: 4px solid #f39c12;
                margin: 20px 0;
                border-radius: 8px;
            }}

            img {{ 
                max-width: 100%; 
                height: auto;
                display: block;
                margin: 25px auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}

            table {{
                border-collapse: collapse;
                margin: 25px 0;
                width: 100%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}

            th {{
                background-color: #2E7D32;
                color: white;
                font-weight: 500;
            }}

            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}

            .math {{
                font-size: 1.2em;
                line-height: 1.8;
            }}

            .math .MathJax {{
                font-size: 1.2em !important;
            }}

            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
                color: #7f8c8d;
                font-size: 14px;
                font-family: 'Roboto', sans-serif;
            }}

            @media (max-width: 768px) {{
                body {{
                    margin: 20px;
                    font-size: 16px;
                }}

                .header h1 {{
                    font-size: 24px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📚 Corrigé CIS</h1>
            <p>Correcteur Intelligent de Sujets</p>
        </div>

        <div class="app-promo">
            <strong>Corrigé généré par l'application mobile CIS</strong><br>
            Disponible sur <a href="https://play.google.com/store/apps/details?id=com.cis.app">Google Play Store</a>
        </div>

        <div class="metadata">
            <p><strong>Date:</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
            <p><strong>Matière:</strong> {demande.matiere.nom}</p>
            <p><strong>Classe:</strong> {demande.classe.nom if demande.classe else 'Non spécifiée'}</p>
            <p><strong>Sous-système:</strong> {demande.sous_systeme.nom if demande.sous_systeme else 'Non spécifié'}</p>
        </div>

        <div class="corrige-content">
            <h2>Corrigé détaillé</h2>
            <div class="math">
                {corrige_html}
            </div>
        </div>

        {graphiques_html}

        <div class="footer">
            <p>Corrigé généré automatiquement par CIS - Correcteur Intelligent de Sujets</p>
            <p>© {datetime.now().year} CIS - Tous droits réservés</p>
        </div>

        <script>
            MathJax = {{
                tex: {{
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    processEscapes: true,
                    processEnvironments: true
                }},
                options: {{
                    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
                    renderActions: {{
                        addMenu: [0, '', '']
                    }}
                }},
                svg: {{
                    fontCache: 'global',
                    scale: 1.2  /* Augmente la taille des formules mathématiques */
                }},
                startup: {{
                    typeset: true
                }}
            }};
        </script>
    </body>
    </html>
    """

    # 5. Générer le PDF
    nom_fichier = f"corrige_{demande.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    dossier_pdf = os.path.join(settings.MEDIA_ROOT, 'corriges')
    os.makedirs(dossier_pdf, exist_ok=True)
    chemin_pdf = os.path.join(dossier_pdf, nom_fichier)

    HTML(string=html_content).write_pdf(chemin_pdf)

    return f'/media/corriges/{nom_fichier}'


def _nettoyer_contenu(html_content):
    """
    Nettoie le contenu HTML pour supprimer les répétitions d'énoncés
    et améliorer la structure du document
    """
    # 1. Supprimer les répétitions d'énoncés de questions
    # Pattern pour détecter les énoncés répétés (souvent précédés de "Question", "Exercice", etc.)
    patterns = [
        r'(<p><strong>Question[^<]+</strong>[^<]+</p>).*?(?=<p><strong>Question|</div>)',
        r'(<p><strong>Exercice[^<]+</strong>[^<]+</p>).*?(?=<p><strong>Exercice|</div>)',
        r'(Énoncé[^<]+</strong>[^<]+</p>).*?(?=Énoncé|</div>)'
    ]

    for pattern in patterns:
        html_content = re.sub(pattern, '', html_content, flags=re.IGNORECASE | re.DOTALL)

    # 2. Améliorer la structure des réponses
    html_content = html_content.replace('[Réponse]', '<div class="reponse"><h4>📝 Réponse</h4>')
    html_content = html_content.replace('Méthode:', '</div><div class="method"><h4>🧠 Méthode</h4>')
    html_content = html_content.replace('Calculs:', '</div><div class="calcul"><h4>📊 Calculs</h4>')

    # 3. Fermer les divs ouverts
    html_content = html_content.replace('</div></div>', '</div>')

    return html_content