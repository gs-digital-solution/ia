import os
from datetime import datetime
from django.conf import settings
from .mathjax_renderer import mathjax_renderer


class PDFGenerator:
    """Générateur de PDF avec support MathJax natif"""

    def generate_corrige_pdf(self, corrige_text, graphiques, demande):
        """Génère un PDF de qualité avec MathJax"""

        # Convertir le LaTeX en HTML avec MathJax
        html_content = self._create_html_template(corrige_text, demande)

        # Générer le nom de fichier
        nom_fichier = f"corrige_{demande.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        dossier_pdf = os.path.join(settings.MEDIA_ROOT, 'corriges')
        os.makedirs(dossier_pdf, exist_ok=True)
        chemin_pdf = os.path.join(dossier_pdf, nom_fichier)

        # Générer le PDF avec MathJax
        success = mathjax_renderer.convert_html_to_pdf(html_content, chemin_pdf)

        if success:
            return f'/media/corriges/{nom_fichier}'
        else:
            # Fallback vers une méthode simple
            return self._generate_simple_pdf(corrige_text, chemin_pdf)

    def _create_html_template(self, corrige_text, demande):
        """Crée le template HTML avec MathJax"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Corrigé CIS - {demande.matiere.nom if demande.matiere else ''}</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js"></script>
            <style>
                body {{ 
                    font-family: 'Arial', sans-serif; 
                    margin: 40px;
                    line-height: 1.6;
                    color: #333;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #2E7D32, #4CAF50);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .content {{ 
                    font-size: 14px;
                }}
                .math {{ 
                    margin: 20px 0;
                    padding: 15px;
                    background: #f8f9fa;
                    border-left: 4px solid #2E7D32;
                    border-radius: 5px;
                }}
                .answer {{ 
                    background: #e8f5e9;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 10px;
                    text-align: left;
                }}
                th {{
                    background-color: #2E7D32;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📚 Corrigé CIS - {demande.matiere.nom if demande.matiere else ''}</h1>
                <p><strong>Date:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                <p><strong>Matière:</strong> {demande.matiere.nom if demande.matiere else ''}</p>
                <p><strong>Classe:</strong> {demande.classe.nom if demande.classe else ''}</p>
            </div>

            <div class="content">
                {corrige_text.replace('\n', '<br>')}
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
                    }}
                }};
            </script>
        </body>
        </html>
        """

    def _generate_simple_pdf(self, corrige_text, output_path):
        """Fallback simple pour la génération PDF"""
        from weasyprint import HTML
        simple_html = f"<html><body><pre>{corrige_text}</pre></body></html>"
        HTML(string=simple_html).write_pdf(output_path)
        return output_path


# Instance globale
pdf_generator = PDFGenerator()