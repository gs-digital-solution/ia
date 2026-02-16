import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
import re

logger = logging.getLogger(__name__)

# Variables globales pour BLIP (lazy loading)
_blip_model = None
_blip_processor = None
def get_blip_model():
    """
    Charge le modèle BLIP au premier appel (lazy load).
    """
    global _blip_model, _blip_processor
    if _blip_model is None:
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            _blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(device).eval()
            logger.info("🖼️ BLIP chargé sur %s", device)
        except Exception as e:
            logger.error(f"❌ Erreur chargement BLIP: {e}")
            return None, None
    return _blip_processor, _blip_model

def detecter_formes_opencv(image_cv):
    """
    Détecte les formes géométriques dans une image avec OpenCV.
    Retourne une description structurée.
    """
    resultats = {
        "lignes": 0,
        "cercles": 0,
        "rectangles": 0,
        "fleches": 0,
        "angle_principal": None,
        "description_formes": []
    }

    try:
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Détection des contours
        edges = cv2.Canny(gray, 50, 150)

        # 1) Détection des lignes
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30)
        if lines is not None:
            resultats["lignes"] = len(lines)

            # Calculer l'angle moyen des lignes (pour plan incliné)
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(abs(angle) % 180)

            if angles:
                # Prendre l'angle le plus fréquent (mode approximatif)
                angles_arr = np.array(angles)
                hist, bins = np.histogram(angles_arr, bins=18, range=(0, 180))
                resultats["angle_principal"] = int(bins[np.argmax(hist)] + 5)
                resultats["description_formes"].append(
                    f"{len(lines)} lignes détectées, angle principal ~{resultats['angle_principal']}°"
                )

        # 2) Détection des cercles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=5, maxRadius=100)
        if circles is not None:
            resultats["cercles"] = len(circles[0])
            resultats["description_formes"].append(f"{len(circles[0])} cercles détectés")

        # 3) Détection des rectangles
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                resultats["rectangles"] += 1

        if resultats["rectangles"] > 0:
            resultats["description_formes"].append(f"{resultats['rectangles']} formes rectangulaires")

        # 4) Détection basique des flèches (lignes avec triangle)
        # Version simplifiée - on compte les lignes courtes qui pourraient être des flèches
        if lines is not None:
            fleches_potentielles = sum(1 for line in lines if abs(line[0][2] - line[0][0]) < 50)
            resultats["fleches"] = min(fleches_potentielles // 2, 5)  # Approximation

    except Exception as e:
        logger.warning(f"⚠️ Erreur détection formes: {e}")

    return resultats


def extraire_texte_dans_schema(image_pil):
    """
    Extrait le texte présent DANS le schéma (légendes, valeurs, etc.)
    """
    try:
        # Configuration Tesseract pour reconnaître texte + chiffres
        custom_config = r'--oem 3 --psm 6 -l fra+eng+digits'
        texte = pytesseract.image_to_string(image_pil, config=custom_config)

        # Nettoyer et structurer
        lignes = [l.strip() for l in texte.split('\n') if l.strip()]

        # Extraire les nombres (masses, angles, etc.)
        nombres = []
        for ligne in lignes:
            nombres_ligne = re.findall(r'\d+[.,]?\d*', ligne)
            nombres.extend(nombres_ligne)

        return {
            "texte_complet": texte.strip(),
            "lignes": lignes,
            "nombres_extraits": list(set(nombres))  # Éliminer les doublons
        }
    except Exception as e:
        logger.warning(f"⚠️ Erreur Tesseract sur schéma: {e}")
        return {"texte_complet": "", "lignes": [], "nombres_extraits": []}


def analyser_schema_unique(image_pil, page_num, position_dans_flux=None):
    """
    Analyse complète d'un schéma avec tous les outils gratuits.
    """
    resultats = {
        "page": page_num,
        "position": position_dans_flux,
        "legende": "",
        "type_schema": "inconnu",
        "formes": {},
        "texte": "",
        "nombres": [],
        "description_textuelle": ""
    }

    try:
        # 1) Conversion pour OpenCV
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # 2) Légende avec BLIP
        #processor, model = get_blip_model()
        #if processor and model:
            #inputs = processor(image_pil, return_tensors="pt")
        # out = model.generate(**inputs)
            #resultats["legende"] = processor.decode(out[0], skip_special_tokens=True)

        # 3) Détection des formes avec OpenCV
        resultats["formes"] = detecter_formes_opencv(image_cv)

        # 4) Texte dans le schéma avec Tesseract
        texte_data = extraire_texte_dans_schema(image_pil)
        resultats["texte"] = texte_data["texte_complet"]
        resultats["nombres"] = texte_data["nombres_extraits"]

        # 5) Déduction du type de schéma
        legende_lower = resultats["legende"].lower()
        formes = resultats["formes"]

        if any(word in legende_lower for word in ["circuit", "electrical", "electric", "résistance"]):
            resultats["type_schema"] = "circuit_electrique"
        elif any(word in legende_lower for word in ["inclined", "plan", "ramp", "pente"]):
            resultats["type_schema"] = "plan_incline"
        elif any(word in legende_lower for word in ["pendulum", "pendule"]):
            resultats["type_schema"] = "pendule"
        elif any(word in legende_lower for word in ["graph", "plot", "courbe", "fonction"]):
            resultats["type_schema"] = "graphique"
        elif formes.get("cercles", 0) > 1 and formes.get("lignes", 0) > 2:
            resultats["type_schema"] = "treuil_poulie"
        elif formes.get("rectangles", 0) > 0 and formes.get("lignes", 0) > 3:
            resultats["type_schema"] = "schema_technique"

        # 6) Construire une description textuelle complète
        description = []
        if resultats["legende"]:
            description.append(f"Schéma: {resultats['legende']}")

        if resultats["type_schema"] != "inconnu":
            description.append(f"Type: {resultats['type_schema']}")

        if resultats["formes"]["description_formes"]:
            description.append("Éléments: " + ", ".join(resultats["formes"]["description_formes"][:2]))

        if resultats["nombres"]:
            description.append(f"Valeurs: {', '.join(resultats['nombres'][:5])}")

        if resultats["texte"] and len(resultats["texte"]) < 100:
            description.append(f"Texte: {resultats['texte'][:100]}")

        resultats["description_textuelle"] = " | ".join(description)

    except Exception as e:
        logger.error(f"❌ Erreur analyse schéma page {page_num}: {e}")

    return resultats