import os
import tempfile
import subprocess
from django.conf import settings


class MathJaxRenderer:
    """Moteur de rendu MathJax natif pour compilation LaTeX de qualité"""

    def __init__(self):
        self.mathjax_path = settings.MATHJAX_PATH  # Chemin vers MathJax Node.js
        self.temp_dir = tempfile.gettempdir()

    def render_latex_to_html(self, latex_content):
        """Convertit du LaTeX en HTML avec MathJax"""
        # Création d'un template HTML avec MathJax
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                .math {{ margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="content">
                {latex_content}
            </div>
        </body>
        </html>
        """
        return html_template

    def convert_html_to_pdf(self, html_content, output_path):
        """Convertit HTML en PDF avec MathJax rendu"""
        # Utilisation de Playwright pour le rendu avec MathJax
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()

                # Charger le contenu HTML
                page.set_content(html_content)

                # Attendre que MathJax ait fini le rendu
                page.wait_for_function("window.MathJax && window.MathJax.typesetPromise")
                page.wait_for_timeout(2000)  # Attente supplémentaire

                # Générer le PDF
                pdf_bytes = page.pdf(format='A4', print_background=True)

                with open(output_path, 'wb') as f:
                    f.write(pdf_bytes)

                browser.close()

                return True

        except Exception as e:
            print(f"Erreur Playwright: {e}")
            # Fallback: Utilisation de WeasyPrint sans MathJax
            return self.fallback_pdf_generation(html_content, output_path)

    def fallback_pdf_generation(self, html_content, output_path):
        """Fallback avec WeasyPrint"""
        from weasyprint import HTML
        try:
            HTML(string=html_content).write_pdf(output_path)
            return True
        except Exception as e:
            print(f"Erreur WeasyPrint: {e}")
            return False


# Singleton global
mathjax_renderer = MathJaxRenderer()