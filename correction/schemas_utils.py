import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import logging
import re
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

# ===== CONFIGURATION BLIP (comme dans l'ancien code) =====
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖼️ BLIP device: {device}")

    # Chargement unique au démarrage du module
    _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
        device).eval()
    logger.info("✅ BLIP chargé avec succès")
except Exception as e:
    logger.error(f"❌ Erreur chargement BLIP: {e}")
    _blip_processor = None
    _blip_model = None


def get_blip_caption(image_pil):
    """Génère une légende pour une image avec BLIP (inspiré de decrire_image)"""
    if _blip_processor is None or _blip_model is None:
        return ""

    try:
        # Conversion RGB si nécessaire
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        inputs = _blip_processor(image_pil, return_tensors="pt").to(device)
        out = _blip_model.generate(**inputs, max_new_tokens=50)
        caption = _blip_processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.warning(f"⚠️ Erreur BLIP: {e}")
        return ""


# ===== FONCTIONS EXISTANTES (adaptées) =====

def preprocess_image_opencv(image_pil):
    """Prétraite l'image pour OpenCV (contraste, netteté)"""
    try:
        # Amélioration du contraste avec PIL
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(2.0)

        # Amélioration de la netteté
        enhancer = ImageEnhance.Sharpness(image_pil)
        image_pil = enhancer.enhance(1.5)

        # Conversion en CV2
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return image_cv
    except Exception as e:
        logger.warning(f"⚠️ Erreur preprocessing: {e}")
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def detecter_formes_opencv(image_cv):
    """Détecte les formes géométriques dans une image avec OpenCV"""
    resultats = {
        "lignes": 0,
        "cercles": 0,
        "rectangles": 0,
        "angle_principal": None,
        "description_formes": []
    }

    try:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Détection des lignes (seuils plus bas pour plus de sensibilité)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=15)
        if lines is not None:
            resultats["lignes"] = len(lines)

            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(abs(angle) % 180)

            if angles:
                angles_arr = np.array(angles)
                hist, bins = np.histogram(angles_arr, bins=18, range=(0, 180))
                resultats["angle_principal"] = int(bins[np.argmax(hist)] + 5)
                resultats["description_formes"].append(f"{len(lines)} lignes")

        # Détection des cercles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=5, maxRadius=100)
        if circles is not None:
            resultats["cercles"] = len(circles[0])
            resultats["description_formes"].append(f"{len(circles[0])} cercles")

        # Détection des rectangles
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                resultats["rectangles"] += 1

        if resultats["rectangles"] > 0:
            resultats["description_formes"].append(f"{resultats['rectangles']} rectangles")

    except Exception as e:
        logger.warning(f"⚠️ Erreur détection formes: {e}")

    return resultats


def extraire_texte_dans_schema(image_pil):
    """Extrait le texte présent dans le schéma (inspiré de extraire_texte_image)"""
    try:
        # Prétraitement comme dans l'ancien code
        image = image_pil.convert("L")

        # Amélioration du contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.5)

        # Amélioration de la netteté
        image = image.filter(ImageFilter.SHARPEN)

        # Configuration Tesseract
        custom_config = r'--oem 3 --psm 6 -l fra+eng+digits'
        texte = pytesseract.image_to_string(image, config=custom_config)

        # Extraction des nombres
        nombres = re.findall(r'\d+[.,]?\d*', texte)

        return {
            "texte_complet": texte.strip(),
            "nombres_extraits": list(set(nombres))
        }
    except Exception as e:
        logger.warning(f"⚠️ Erreur Tesseract: {e}")
        return {"texte_complet": "", "nombres_extraits": []}


def analyser_schema_unique(image_pil, page_num):
    """Analyse complète d'un schéma avec BLIP + OpenCV + Tesseract"""
    resultats = {
        "page": page_num,
        "legende": "",
        "type_schema": "inconnu",
        "formes": {},
        "texte": "",
        "nombres": [],
        "description": ""
    }

    try:
        # 1) Légende avec BLIP
        resultats["legende"] = get_blip_caption(image_pil)

        # 2) Prétraitement pour OpenCV
        image_cv = preprocess_image_opencv(image_pil)

        # 3) Détection des formes
        resultats["formes"] = detecter_formes_opencv(image_cv)

        # 4) Texte dans le schéma
        texte_data = extraire_texte_dans_schema(image_pil)
        resultats["texte"] = texte_data["texte_complet"]
        resultats["nombres"] = texte_data["nombres_extraits"]

        # 5) Déduction du type de schéma
        legende_lower = resultats["legende"].lower()
        formes = resultats["formes"]

        if any(word in legende_lower for word in ["circuit", "electrical", "electric"]):
            resultats["type_schema"] = "circuit_electrique"
        elif any(word in legende_lower for word in ["inclined", "plan", "ramp", "pente"]):
            resultats["type_schema"] = "plan_incline"
        elif any(word in legende_lower for word in ["pendulum", "pendule"]):
            resultats["type_schema"] = "pendule"
        elif any(word in legende_lower for word in ["graph", "plot", "courbe"]):
            resultats["type_schema"] = "graphique"
        elif formes.get("cercles", 0) > 1 and formes.get("lignes", 0) > 2:
            resultats["type_schema"] = "treuil_poulie"

        # 6) Construction de la description
        description_parts = []
        if resultats["legende"]:
            description_parts.append(resultats["legende"])
        if resultats["formes"]["description_formes"]:
            description_parts.append("avec " + ", ".join(resultats["formes"]["description_formes"]))
        if resultats["nombres"]:
            description_parts.append("valeurs: " + ", ".join(resultats["nombres"][:3]))

        resultats["description"] = " | ".join(description_parts)

    except Exception as e:
        logger.error(f"❌ Erreur analyse page {page_num}: {e}")

    return resultats