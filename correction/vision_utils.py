import os
import base64
import requests
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)


def analyser_schema_deepseek_vl(image_path: str) -> dict:
    """
    Analyse un schéma/image avec DeepSeek-VL.
    Retourne un dictionnaire structuré avec la description et les données extraites.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("❌ Clé API DeepSeek non configurée")
        return {
            "legende": "",
            "type_schema": "inconnu",
            "description": "",
            "donnees_extraites": {},
            "erreur": "Clé API manquante"
        }

    try:
        # Encoder l'image en base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Prompt spécialisé pour les schémas scientifiques
        prompt = """
        Tu es un expert en analyse de schémas scientifiques pour des exercices de physique, chimie, mathématiques.

        Analyse cette image et retourne UNIQUEMENT un JSON structuré avec les informations suivantes :

        {
            "legende": "Description courte et précise du schéma en 1 phrase",
            "type_schema": "plan_incline|circuit_electrique|pendule|graphique|treuil|optique|autre",
            "description_detaillee": "Description complète de tout ce que tu vois (formes, flèches, annotations, relations)",
            "donnees_extraites": {
                "angles": [{"valeur": 30, "unite": "degres", "position": "entre la pente et l'horizontale"}],
                "masses": [{"valeur": 2, "unite": "kg", "objet": "bloc"}],
                "longueurs": [{"valeur": 1.5, "unite": "m", "objet": "fil du pendule"}],
                "forces": ["poids", "tension", "frottement", "réaction normale"],
                "composants": ["résistance 10Ω", "batterie 12V", "interrupteur"],
                "relations": ["le bloc descend le long du plan", "le courant circule de la borne positive vers la négative"]
            },
            "contexte": "Ce schéma illustre un exercice sur ..."
        }

        RÈGLES IMPORTANTES :
        - Ne retourne QUE le JSON, sans texte supplémentaire
        - Si un champ n'est pas pertinent, mets une liste vide []
        - Pour les angles, masses, longueurs : extrais TOUTES les valeurs visibles
        - Sois extrêmement précis dans la description
        """

        # Appel à DeepSeek-VL
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-vl",  # Le modèle multimodal
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            },
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"❌ Erreur API DeepSeek-VL: {response.status_code} - {response.text}")
            return {
                "legende": "",
                "type_schema": "inconnu",
                "description": f"Erreur API: {response.status_code}",
                "donnees_extraites": {}
            }

        resultat = response.json()
        contenu = resultat['choices'][0]['message']['content']

        # Extraire le JSON de la réponse
        import json
        import re

        # Chercher un bloc JSON dans la réponse
        match = re.search(r'\{.*\}', contenu, re.DOTALL)
        if match:
            try:
                donnees = json.loads(match.group())
                logger.info(
                    f"✅ DeepSeek-VL: {donnees.get('type_schema', 'inconnu')} - {donnees.get('legende', '')[:100]}")
                return donnees
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Réponse non-JSON, utilisation texte brut")
                return {
                    "legende": contenu[:200],
                    "type_schema": "inconnu",
                    "description": contenu,
                    "donnees_extraites": {}
                }
        else:
            return {
                "legende": contenu[:200],
                "type_schema": "inconnu",
                "description": contenu,
                "donnees_extraites": {}
            }

    except Exception as e:
        logger.error(f"❌ Exception DeepSeek-VL: {e}")
        return {
            "legende": "",
            "type_schema": "inconnu",
            "description": f"Erreur: {str(e)}",
            "donnees_extraites": {}
        }


def analyser_schemas_document_vl(fichier_path: str) -> dict:
    """
    Analyse tous les schémas d'un document avec DeepSeek-VL.
    Version simplifiée qui remplace tout le pipeline BLIP/OpenCV/Tesseract.
    """
    from pdf2image import convert_from_path
    from PIL import Image
    import os
    import tempfile

    resultats = {
        "schemas_detaille": [],
        "nombre_total": 0
    }

    try:
        ext = os.path.splitext(fichier_path)[1].lower()
        logger.info(f"🔍 Analyse document avec DeepSeek-VL: {fichier_path}")

        # Convertir le document en images
        images_temp = []
        if ext == '.pdf':
            images = convert_from_path(fichier_path, dpi=200)
            logger.info(f"📄 PDF converti en {len(images)} pages")

            for i, img in enumerate(images):
                temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                img.save(temp_path, 'PNG')
                images_temp.append((i + 1, temp_path))
        else:
            # C'est déjà une image
            images_temp.append((1, fichier_path))

        # Analyser chaque page avec DeepSeek-VL
        for page_num, img_path in images_temp:
            logger.info(f"   Analyse page {page_num}...")

            # Appel unique à DeepSeek-VL
            analyse = analyser_schema_deepseek_vl(img_path)

            # Si on a détecté quelque chose d'intéressant
            if analyse.get('legende') or analyse.get('type_schema') != 'inconnu':
                schema_data = {
                    "page": page_num,
                    "legende": analyse.get('legende', ''),
                    "type_schema": analyse.get('type_schema', 'inconnu'),
                    "description": analyse.get('description_detaillee', analyse.get('description', '')),
                    "donnees": analyse.get('donnees_extraites', {}),
                    "contexte": analyse.get('contexte', '')
                }
                resultats["schemas_detaille"].append(schema_data)
                logger.info(f"   ✅ Schéma détecté: {schema_data['type_schema']}")
            else:
                logger.info(f"   ❌ Pas de schéma sur cette page")

            # Nettoyer le fichier temporaire si nécessaire
            if img_path != fichier_path:
                try:
                    os.unlink(img_path)
                except:
                    pass

        resultats["nombre_total"] = len(resultats["schemas_detaille"])
        logger.info(f"🎯 Total: {resultats['nombre_total']} schémas détectés")

    except Exception as e:
        logger.error(f"❌ Erreur analyse document: {e}")
        import traceback
        traceback.print_exc()

    return resultats
