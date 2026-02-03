"""
Utils pour la conversion des balises LaTeX
"""
import re


def convertir_balises_latex_mathpix(texte: str) -> str:
    """
    Convertit les balises LaTeX de MathPix ($...$) en format frontend (\(...\)).

    MathPix retourne : $x^2 + y^2 = z^2$
    Frontend attend : \(x^2 + y^2 = z^2\)
    """
    if not texte:
        return texte

    # 1. Nettoyage de base
    texte = texte.replace('\r\n', '\n').replace('\r', '\n')

    # 2. Convertir $$...$$ (display math) en \[...\]
    # Gère les formules multi-lignes
    def convert_display(match):
        content = match.group(1).strip()
        # Fusionner les lignes
        content = re.sub(r'\s*\n\s*', ' ', content)
        return r'\[' + content + r'\]'

    texte = re.sub(r'\$\$([\s\S]+?)\$\$', convert_display, texte)

    # 3. Convertir $...$ (inline math) en \(...\)
    # Algorithme simple pour éviter les faux positifs (prix, devises)
    result = []
    i = 0
    n = len(texte)

    while i < n:
        if texte[i] == '$' and (i == 0 or texte[i - 1] != '\\'):
            # Trouver le $ fermant
            j = i + 1
            while j < n and not (texte[j] == '$' and (j == 0 or texte[j - 1] != '\\')):
                j += 1

            if j < n:  # $ fermant trouvé
                math_content = texte[i + 1:j]

                # Vérifier si c'est un prix (ex: $10, $2.50)
                if re.match(r'^\d+[\.,]?\d*$', math_content):
                    result.append('$' + math_content + '$')
                else:
                    result.append(r'\(' + math_content + r'\)')
                i = j + 1
            else:
                # $ non fermé, laisser tel quel
                result.append('$')
                i += 1
        else:
            result.append(texte[i])
            i += 1

    return ''.join(result)