import os
import re
import base64
from io import BytesIO
from datetime import datetime
from django.conf import settings
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuration matplotlib pour LaTeX
mpl.use('Agg')
plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.size': 16,
    'axes.unicode_minus': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.dpi': 150,
})

def generer_pdf_corrige(corrige_text, graphiques, demande):
    """Génère un PDF backend avec équations LaTeX précompilées en images"""
    try:
        corrige_html = convertir_latex_vers_html(corrige_text)
        corrige_html = _nettoyer_contenu(corrige_html)
        corrige_html = _precompiler_latex_en_images(corrige_html)
        html_content = _generer_html_ultra_optimise(corrige_html, graphiques, demande)
        pdf_rel_path = _generer_pdf_final(html_content, demande)
        return pdf_rel_path

    except Exception as e:
        print(f"Erreur génération PDF: {e}")
        return _generer_pdf_fallback(corrige_text, demande)

def convertir_latex_vers_html(corrige_text):
    if not corrige_text:
        return ""

    corrige_text = corrige_text.replace(r'\[', r'\\[').replace(r'\]', r'\\]')
    corrige_text = corrige_text.replace(r'\(', r'\\(').replace(r'\)', r'\\)')

    corrige_text = re.sub(r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', r'\\[\1\\]', corrige_text, flags=re.DOTALL)
    corrige_text = re.sub(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', r'\\begin{aligned}\1\\end{aligned}', corrige_text, flags=re.DOTALL)
    corrige_text = re.sub(r'\\begin\{array\}(.*?)\\end\{array\}', r'\\begin{array}\1\\end{array}', corrige_text, flags=re.DOTALL)

    corrige_text = corrige_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return corrige_text

def _nettoyer_contenu(html_content):
    patterns = [
        r'(<p><strong>Question[^<]+</strong>[^<]+</p>).*?(?=<p><strong>Question|</div>)',
        r'(<p><strong>Exercice[^<]+</strong>[^<]+</p>).*?(?=<p><strong>Exercice|</div>)',
        r'(Énoncé[^<]+</strong>[^<]+</p>).*?(?=Énoncé|</div>)'
    ]
    for pattern in patterns:
        html_content = re.sub(pattern, '', html_content, flags=re.IGNORECASE | re.DOTALL)

    html_content = html_content.replace('[Réponse]', '<div class="reponse">')
    html_content = html_content.replace('Méthode:', '</div><div class="method">')
    html_content = html_content.replace('Calculs:', '</div><div class="calcul">')

    html_content += '</div>' * html_content.count('<div class="')
    return html_content

def _precompiler_latex_en_images(html_content):
    patterns = [
        (r'\\\[(.*?)\\\]', True),
        (r'\\\((.*?)\\\)', False),
        (r'\$\$(.*?)\$\$', True),
        (r'\$(.*?)\$', False),
    ]
    try:
        for pattern, is_display in patterns:
            matches = list(re.finditer(pattern, html_content, re.DOTALL))
            for match in matches:
                latex_code = match.group(1).strip()
                img_data = _convertir_latex_vers_image(latex_code, is_display)
                if img_data:
                    if is_display:
                        replacement = f'<div class="math-equation"><img src="{img_data}" alt="formule mathématique"></div>'
                    else:
                        replacement = f'<span class="math-inline"><img src="{img_data}" alt="formule mathématique"></span>'
                    html_content = html_content.replace(match.group(0), replacement)
        return html_content
    except Exception as e:
        print(f"Erreur précompilation LaTeX: {e}")
        return html_content

def _convertir_latex_vers_image(latex_code, is_display=True):
    try:
        figsize = (6, 0.8) if not is_display else (8, 1.2)
        fig = plt.figure(figsize=figsize, dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        fontsize = 18 if not is_display else 22
        ax.text(0.5, 0.5, f'${latex_code}$', fontsize=fontsize, ha='center', va='center', transform=ax.transAxes)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.02, transparent=True)
        plt.close(fig)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_data}"
    except Exception as e:
        print(f"Erreur conversion LaTeX en image: {e}")
        return None

def _generer_html_ultra_optimise(corrige_html, graphiques, demande):
    graphiques_html = ""
    if graphiques:
        from .ia_utils import tracer_graphique
        for i, g in enumerate(graphiques):
            img_path = tracer_graphique(g, f"graph_{demande.id}_{i}_{datetime.now().strftime('%H%M%S')}.png")
            if img_path:
                full_path = os.path.join(settings.MEDIA_ROOT, img_path)
                graphiques_html += f'<img src="file://{full_path}" style="max-width:100%; margin:12px 0;">'

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corrigé CIS - {demande.matiere.nom}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, Cantarell, sans-serif;
            margin: 8px;
            padding: 0;
            font-size: 22px;
            line-height: 1.5;
            color: #000;
            background-color: #fff;
        }}
        .header {{
            text-align: center;
            margin-bottom: 12px;
            border-bottom: 2px solid #2E7D32;
            padding-bottom: 10px;
        }}
        .header h1 {{
            color: #2E7D32;
            font-size: 26px;
            font-weight: 700;
            margin: 0 0 4px 0;
            line-height: 1.2;
        }}
        .corrige-content {{
            margin-top: 12px;
        }}
        .reponse, .method, .calcul {{
            margin: 15px 0;
            padding: 12px;
            border-radius: 5px;
            font-size: 21px;
            line-height: 1.4;
        }}
        .reponse {{ background: #f0f8f0; }}
        .method {{ background: #e8f4f8; }}
        .calcul {{ background: #fff3e0; }}
        .math-equation {{
            margin: 10px 0;
            text-align: center;
            padding: 5px;
        }}
        .math-equation img, .math-inline img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        table {{
            border-collapse: collapse;
            margin: 15px 0;
            width: 100%;
            font-size: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6px;
        }}
        th {{
            background-color: #2E7D32;
            color: white;
            font-weight: 500;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 16px;
        }}
        @page {{
            margin: 0.2cm;
            size: A4;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Corrigé CIS</h1>
        <p>{datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        <p><strong>Matière:</strong> {demande.matiere.nom}</p>
        <p><strong>Classe:</strong> {demande.classe.nom if demande.classe else 'N/A'}</p>
    </div>
    <div class="corrige-content">
        {corrige_html}
    </div>
    {graphiques_html}
    <div class="footer">
        <p>© {datetime.now().year} CIS - Correcteur Intelligent de Sujets</p>
    </div>
</body>
</html>"""

def _generer_pdf_final(html_content, demande):
    nom_fichier = f"corrige_{demande.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    dossier_pdf = os.path.join(settings.MEDIA_ROOT, 'corriges')
    os.makedirs(dossier_pdf, exist_ok=True)
    chemin_pdf = os.path.join(dossier_pdf, nom_fichier)

    font_config = FontConfiguration()
    HTML(string=html_content).write_pdf(
        chemin_pdf,
        stylesheets=[CSS(string='@page {margin: 0.2cm;} body {margin: 0; padding: 10px;}')],
        font_config=font_config,
        optimize_size=('fonts', 'images', 'content')
    )
    return f'/media/corriges/{nom_fichier}'

def _generer_pdf_fallback(corrige_text, demande):
    html_content = f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <style>body {{ font-family: Arial,sans-serif; font-size: 20px; margin: 10px; }}</style>
    </head><body>
    <h2>Corrigé CIS - {demande.matiere.nom}</h2>
    <p>Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    <div>{corrige_text.replace('$','')}</div>
    </body></html>"""

    nom_fichier = f"corrige_{demande.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    dossier_pdf = os.path.join(settings.MEDIA_ROOT, 'corriges')
    os.makedirs(dossier_pdf, exist_ok=True)
    chemin_pdf = os.path.join(dossier_pdf, nom_fichier)
    HTML(string=html_content).write_pdf(chemin_pdf)
    return f'/media/corriges/{nom_fichier}'