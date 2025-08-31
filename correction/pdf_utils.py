import os
import re
import base64
from io import BytesIO
from datetime import datetime
from django.conf import settings
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
from .ia_utils import convertir_latex_vers_html, tracer_graphique
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Configuration matplotlib pour éviter les problèmes de backend
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
    """Génère un PDF optimisé pour mobile avec équations pré-compilées"""
    try:
        # 1. Convertir et nettoyer le contenu
        corrige_html = convertir_latex_vers_html(corrige_text)
        corrige_html = _nettoyer_contenu(corrige_html)

        # 2. PRÉ-COMPILER LeTeX en images base64
        corrige_html = _precompiler_latex_en_images(corrige_html)

        # 3. Générer les graphiques
        graphiques_html = _generer_graphiques_html(graphiques, demande)

        # 4. HTML ultra-optimisé pour mobile
        html_content = _generer_html_ultra_optimise(corrige_html, graphiques_html, demande)

        # 5. Générer le PDF avec configuration ultra-optimisée
        return _generer_pdf_final(html_content, demande)

    except Exception as e:
        print(f"Erreur génération PDF: {e}")
        # Fallback: générer un PDF basique sans équations
        return _generer_pdf_fallback(corrige_text, demande)


def _generer_html_ultra_optimise(corrige_html, graphiques_html, demande):
    """Génère le HTML ultra-optimisé pour mobile"""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
    <title>Corrigé CIS - {demande.matiere.nom}</title>
    <style>
        /* POLICES OPTIMISÉES POUR MOBILE */
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 8px;
            padding: 0;
            line-height: 1.5;
            color: #000;
            font-size: 22px; /* TRÈS GRANDE TAILLE */
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            font-weight: 400;
        }}

        /* EN-TÊTE COMPACTE */
        .header {{
            text-align: center;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 2px solid #2E7D32;
        }}

        .header h1 {{ 
            color: #2E7D32;
            font-size: 26px;
            font-weight: 700;
            margin: 0 0 4px 0;
            line-height: 1.2;
        }}

        .header p {{
            color: #666;
            font-size: 18px;
            margin: 3px 0;
        }}

        /* PROMO APPLICATION */
        .app-promo {{
            background: #2E7D32;
            color: white;
            padding: 8px 10px;
            border-radius: 6px;
            margin: 10px 0;
            text-align: center;
            font-size: 18px;
            line-height: 1.3;
        }}

        .app-promo a {{
            color: #FFEB3B;
            text-decoration: none;
            font-weight: 600;
        }}

        /* MÉTADONNÉES COMPACTES */
        .metadata {{
            background: #f0f8f0;
            padding: 8px 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 3px solid #2E7D32;
            font-size: 18px;
        }}

        .metadata p {{
            margin: 4px 0;
        }}

        .metadata strong {{
            color: #2E7D32;
        }}

        /* CONTENU PRINCIPAL - TRÈS COMPACT */
        .corrige-content {{
            margin-top: 12px;
        }}

        .corrige-content h2 {{
            color: #2E7D32;
            font-size: 24px;
            margin: 18px 0 10px 0;
            font-weight: 700;
            border-bottom: 1px solid #ddd;
            padding-bottom: 4px;
        }}

        /* STRUCTURES DE RÉPONSE ULTRA-COMPACTES */
        .reponse {{ 
            background: #f0f8f0; 
            padding: 12px; 
            margin: 15px 0;
            border-radius: 5px;
            font-size: 21px;
            line-height: 1.4;
        }}

        .method {{ 
            background: #e8f4f8; 
            padding: 12px; 
            margin: 15px 0;
            border-radius: 5px;
            font-size: 21px;
            line-height: 1.4;
        }}

        .calcul {{ 
            background: #fff3e0; 
            padding: 12px; 
            margin: 15px 0;
            border-radius: 5px;
            font-size: 21px;
            line-height: 1.4;
        }}

        /* ÉQUATIONS MATHÉMATIQUES */
        .math-equation {{
            margin: 10px 0;
            text-align: center;
            padding: 5px;
        }}

        .math-equation img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}

        .math-inline {{
            display: inline-block;
            vertical-align: middle;
            margin: 0 1px;
        }}

        .math-inline img {{
            height: 1.3em;
            vertical-align: middle;
        }}

        /* IMAGES ET GRAPHIQUES */
        img {{ 
            max-width: 100%; 
            height: auto;
            display: block;
            margin: 15px auto;
            border-radius: 4px;
        }}

        /* TABLEAUX COMPACTS */
        table {{
            border-collapse: collapse;
            margin: 15px 0;
            width: 100%;
            font-size: 20px;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 6px;
            text-align: left;
        }}

        th {{
            background-color: #2E7D32;
            color: white;
            font-weight: 500;
        }}

        /* PIED DE PAGE DISCRET */
        .footer {{
            text-align: center;
            margin-top: 20px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 16px;
        }}

        /* OPTIMISATIONS D'IMPRESSION - MARGES MINIMALES */
        @page {{
            margin: 0.2cm;
            size: A4;
        }}

        /* ÉLIMINATION TOUS LES ESPACES INUTILES */
        p, ul, ol {{
            margin: 8px 0;
            line-height: 1.4;
        }}

        br {{
            line-height: 0.5;
        }}

        /* POUR LES PETITS ÉCRANS */
        @media (max-width: 480px) {{
            body {{
                font-size: 20px;
                margin: 6px;
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
        <strong>Application mobile CIS</strong><br>
        Disponible sur <a href="https://play.google.com/store/apps/details?id=com.cis.app">Play Store</a>
    </div>

    <div class="metadata">
        <p><strong>Date:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        <p><strong>Matière:</strong> {demande.matiere.nom}</p>
        <p><strong>Classe:</strong> {demande.classe.nom if demande.classe else 'N/A'}</p>
    </div>

    <div class="corrige-content">
        <h2>Corrigé détaillé</h2>
        {corrige_html}
    </div>

    {graphiques_html}

    <div class="footer">
        <p>© {datetime.now().year} CIS - Correcteur Intelligent de Sujets</p>
    </div>
</body>
</html>
"""


def _precompiler_latex_en_images(html_content):
    """Precompile toutes les expressions LaTeX en images base64"""
    try:
        # Expressions régulières pour trouver le LaTeX
        patterns = [
            (r'\\\[(.*?)\\\]', True),  # \[...\] → display
            (r'\\\((.*?)\\\)', False),  # \(...\) → inline
            (r'\$\$(.*?)\$\$', True),  # $$...$$ → display
            (r'\$(.*?)\$', False),  # $...$ → inline
        ]

        for pattern, is_display in patterns:
            matches = list(re.finditer(pattern, html_content, re.DOTALL))
            for match in matches:
                latex_code = match.group(1).strip()

                # Convertir LaTeX en image base64
                img_data = _convertir_latex_vers_image(latex_code, is_display)
                if img_data:
                    # Remplacer par l'image
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
    """Convertit une expression LaTeX en image base64 avec matplotlib"""
    try:
        # Configuration de la figure (taille minimale)
        figsize = (6, 0.8) if not is_display else (8, 1.2)
        fig = plt.figure(figsize=figsize, dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        # Taille de police adaptée
        fontsize = 18 if not is_display else 22

        # Rendre le LaTeX
        ax.text(0.5, 0.5, f'${latex_code}$', fontsize=fontsize,
                ha='center', va='center', transform=ax.transAxes)

        # Sauvegarder en mémoire avec compression optimale
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    pad_inches=0.02, transparent=True)
        plt.close(fig)

        # Convertir en base64
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_data}"

    except Exception as e:
        print(f"Erreur conversion LaTeX: {e}")
        return None


def _generer_graphiques_html(graphiques, demande):
    """Génère le HTML pour les graphiques"""
    if not graphiques:
        return ""

    graphiques_html = ""
    for i, graphique in enumerate(graphiques):
        nom_fichier = f"graph_{demande.id}_{i}_{datetime.now().strftime('%H%M%S')}.png"
        chemin_graphique = tracer_graphique(graphique, nom_fichier)
        if chemin_graphique:
            graphiques_html += f'<img src="{settings.MEDIA_ROOT}/{chemin_graphique}" style="max-width: 100%; margin: 12px 0;">'

    return graphiques_html


def _generer_pdf_final(html_content, demande):
    """Génère le PDF final avec configuration ultra-optimisée"""
    nom_fichier = f"corrige_{demande.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    dossier_pdf = os.path.join(settings.MEDIA_ROOT, 'corriges')
    os.makedirs(dossier_pdf, exist_ok=True)
    chemin_pdf = os.path.join(dossier_pdf, nom_fichier)

    # Configuration ultra-optimisée pour mobile
    font_config = FontConfiguration()

    # Générer le PDF avec marges minimales
    HTML(string=html_content).write_pdf(
        chemin_pdf,
        stylesheets=[CSS(string='''
            @page { margin: 0.2cm; }
            body { margin: 0; padding: 10px; }
            * { max-height: none !important; }
        ''')],
        font_config=font_config,
        optimize_size=('fonts', 'images', 'content')
    )

    return f'/media/corriges/{nom_fichier}'


def _generer_pdf_fallback(corrige_text, demande):
    """Génère un PDF de fallback basique"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; font-size: 20px; margin: 10px; }}
            .header {{ text-align: center; margin-bottom: 15px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Corrigé CIS - {demande.matiere.nom}</h2>
            <p>Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        <div class="content">
            {corrige_text.replace('$', '')}  <!-- Enlever le LaTeX non compilé -->
        </div>
    </body>
    </html>
    """

    nom_fichier = f"corrige_{demande.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    dossier_pdf = os.path.join(settings.MEDIA_ROOT, 'corriges')
    os.makedirs(dossier_pdf, exist_ok=True)
    chemin_pdf = os.path.join(dossier_pdf, nom_fichier)

    HTML(string=html_content).write_pdf(chemin_pdf)
    return f'/media/corriges/{nom_fichier}'


def _nettoyer_contenu(html_content):
    """Nettoie le contenu HTML pour supprimer les répétitions d'énoncés"""
    # Supprimer les répétitions d'énoncés
    patterns = [
        r'(<p><strong>Question[^<]+</strong>[^<]+</p>).*?(?=<p><strong>Question|</div>)',
        r'(<p><strong>Exercice[^<]+</strong>[^<]+</p>).*?(?=<p><strong>Exercice|</div>)',
        r'(Énoncé[^<]+</strong>[^<]+</p>).*?(?=Énoncé|</div>)'
    ]

    for pattern in patterns:
        html_content = re.sub(pattern, '', html_content, flags=re.IGNORECASE | re.DOTALL)

    # Améliorer la structure des réponses
    html_content = html_content.replace('[Réponse]', '<div class="reponse">')
    html_content = html_content.replace('Méthode:', '</div><div class="method">')
    html_content = html_content.replace('Calculs:', '</div><div class="calcul">')

    # Fermer les divs ouverts
    html_content = html_content + '</div>' * html_content.count('<div class="')

    return html_content