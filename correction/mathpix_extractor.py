"""
Module d'extraction scientifique avec Mathpix et analyse avancée.
Pour les départements scientifiques uniquement.
"""
import os
import tempfile
import json
import base64
import requests
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)

# ======= CONFIGURATION MATHPIX =======
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")


def validate_mathpix_config() -> bool:
    """Vérifie que les clés Mathpix sont configurées."""
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        logger.error("❌ Configuration Mathpix manquante. Vérifiez MATHPIX_APP_ID et MATHPIX_APP_KEY")
        return False
    return True


def call_mathpix_api(image_path: str, formats: List[str] = None) -> Dict:
    """
    Appelle l'API Mathpix pour extraire texte + LaTeX.

    Args:
        image_path: Chemin vers l'image/PDF converti en image
        formats: Formats de sortie souhaités (par défaut: ["text", "latex_simplified"])

    Returns:
        Dict: Réponse JSON de Mathpix
    """
    if not validate_mathpix_config():
        raise ValueError("Configuration Mathpix non valide")

    if formats is None:
        formats = ["text", "latex_simplified", "latex_styled"]

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    headers = {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_APP_KEY,
        "Content-type": "application/json"
    }

    payload = {
        "src": f"data:image/png;base64,{img_b64}",
        "formats": formats,
        "format_options": {
            "text": {
                "transforms": ["rm_spaces", "rm_newlines"]
            },
            "latex_simplified": {
                "transforms": ["rm_spaces"]
            }
        }
    }

    try:
        response = requests.post(
            "https://api.mathpix.com/v3/text",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur API Mathpix: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"   Détails: {e.response.text}")
        raise


def preprocess_image_for_mathpix(image_path: str) -> str:
    """
    Prétraitement d'image avec OpenCV pour améliorer la reconnaissance Mathpix.
    Retourne le chemin de l'image prétraitée.
    """
    try:
        # Charger l'image
        arr = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        # Désinclinaison (deskew)
        coords = np.column_stack(np.where(img > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            (h, w) = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        # Binarisation adaptative
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51, 15
        )

        # Réduction du bruit par ouverture morphologique
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

        # Sauvegarder l'image traitée
        temp_dir = tempfile.gettempdir()
        processed_path = os.path.join(temp_dir, f"mathpix_processed_{os.path.basename(image_path)}")
        cv2.imwrite(processed_path, img)

        return processed_path

    except Exception as e:
        logger.error(f"❌ Erreur prétraitement image: {e}")
        return image_path  # Retourner l'original en cas d'erreur


def extract_scientific_content_mathpix(image_path: str) -> Dict:
    """
    Extraction scientifique complète avec Mathpix.

    Returns:
        Dict avec structure:
        {
            "text": str,                    # Texte extrait
            "latex_blocks": List[str],      # Blocs LaTeX détectés
            "equations": List[Dict],        # Équations avec positions
            "tables": List[Dict],           # Tableaux détectés
            "confidence": float,            # Confiance globale
            "raw_response": Dict            # Réponse brute Mathpix
        }
    """
    logger.info(f"🔬 Extraction Mathpix pour: {image_path}")

    # Prétraitement de l'image
    processed_path = preprocess_image_for_mathpix(image_path)

    try:
        # Appel API Mathpix
        mathpix_result = call_mathpix_api(processed_path)

        # Extraire le texte
        text = mathpix_result.get("text", "").strip()

        # Extraire les blocs LaTeX
        latex_simplified = mathpix_result.get("latex_simplified", "").strip()
        latex_styled = mathpix_result.get("latex_styled", "").strip()

        # Parser les équations et leurs positions
        equations = []
        if "latex_list" in mathpix_result:
            for eq in mathpix_result["latex_list"]:
                equations.append({
                    "latex": eq.get("latex", ""),
                    "confidence": eq.get("confidence", 0),
                    "bounds": eq.get("bounds", {})
                })

        # Détection des tableaux
        tables = []
        if "tables" in mathpix_result:
            for table in mathpix_result["tables"]:
                tables.append({
                    "latex": table.get("latex", ""),
                    "text": table.get("text", ""),
                    "confidence": table.get("confidence", 0)
                })

        # Construire la liste des blocs LaTeX
        latex_blocks = []
        if latex_simplified:
            latex_blocks.append(latex_simplified)
        if latex_styled and latex_styled != latex_simplified:
            latex_blocks.append(latex_styled)

        # Confiance globale
        confidence = mathpix_result.get("confidence", 0)

        result = {
            "text": text,
            "latex_blocks": latex_blocks,
            "equations": equations,
            "tables": tables,
            "confidence": confidence,
            "raw_response": mathpix_result
        }

        logger.info(
            f"✅ Extraction Mathpix réussie: {len(text)} caractères, {len(latex_blocks)} blocs LaTeX, confiance: {confidence:.2f}")

        return result

    finally:
        # Nettoyer l'image prétraitée si différente de l'originale
        if processed_path != image_path and os.path.exists(processed_path):
            try:
                os.unlink(processed_path)
            except:
                pass


def extract_pdf_with_mathpix(pdf_path: str) -> Dict:
    """
    Extraction complète d'un PDF scientifique avec Mathpix.
    Traite chaque page et fusionne les résultats.

    Returns:
        Dict avec structure similaire à extract_scientific_content_mathpix
    """
    logger.info(f"📄 Extraction PDF avec Mathpix: {pdf_path}")

    try:
        # Convertir le PDF en images (pages)
        pages = convert_from_path(pdf_path, dpi=200)
        logger.info(f"   📑 PDF converti: {len(pages)} pages")

        all_text = []
        all_latex_blocks = []
        all_equations = []
        all_tables = []
        total_confidence = 0

        for page_idx, page in enumerate(pages, 1):
            logger.info(f"   🔍 Traitement page {page_idx}/{len(pages)}...")

            # Sauvegarder la page en image temporaire
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                page.save(tmp.name, "PNG")
                temp_image_path = tmp.name

            try:
                # Extraire avec Mathpix
                page_result = extract_scientific_content_mathpix(temp_image_path)

                # Ajouter l'index de page aux résultats
                page_text = f"\n\n--- PAGE {page_idx} ---\n\n{page_result['text']}"
                all_text.append(page_text)

                # Ajouter les blocs LaTeX avec référence de page
                for latex in page_result["latex_blocks"]:
                    all_latex_blocks.append(f"Page {page_idx}: {latex}")

                # Ajouter les équations avec référence de page
                for eq in page_result["equations"]:
                    eq["page"] = page_idx
                    all_equations.append(eq)

                # Ajouter les tableaux avec référence de page
                for table in page_result["tables"]:
                    table["page"] = page_idx
                    all_tables.append(table)

                total_confidence += page_result["confidence"]

                logger.info(
                    f"   ✅ Page {page_idx}: {len(page_result['text'])} caractères, {len(page_result['latex_blocks'])} formules")

            except Exception as e:
                logger.error(f"   ❌ Erreur page {page_idx}: {e}")
                # Ajouter un marqueur d'erreur
                all_text.append(f"\n\n--- PAGE {page_idx} (ERREUR EXTRACTION) ---\n\n")
            finally:
                # Nettoyer l'image temporaire
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)

        # Calculer la confiance moyenne
        avg_confidence = total_confidence / len(pages) if pages else 0

        # Fusionner les résultats
        result = {
            "text": "\n".join(all_text),
            "latex_blocks": all_latex_blocks,
            "equations": all_equations,
            "tables": all_tables,
            "confidence": avg_confidence,
            "total_pages": len(pages),
            "extraction_method": "mathpix_pdf"
        }

        logger.info(f"✅ Extraction PDF terminée: {len(result['text'])} caractères, {len(all_latex_blocks)} formules")

        return result

    except Exception as e:
        logger.error(f"❌ Erreur extraction PDF: {e}")
        raise


def extract_image_with_mathpix(image_path: str) -> Dict:
    """
    Extraction d'une image scientifique avec Mathpix.
    """
    logger.info(f"🖼️ Extraction image avec Mathpix: {image_path}")

    try:
        result = extract_scientific_content_mathpix(image_path)
        result["extraction_method"] = "mathpix_image"
        return result

    except Exception as e:
        logger.error(f"❌ Erreur extraction image: {e}")
        raise


def analyze_scientific_document(file_path: str) -> Dict:
    """
    Analyse scientifique unifiée: détecte le type de fichier et utilise Mathpix.

    Args:
        file_path: Chemin vers le fichier (PDF ou image)

    Returns:
        Dict: Résultats d'analyse scientifique
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return extract_pdf_with_mathpix(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
        return extract_image_with_mathpix(file_path)
    else:
        raise ValueError(f"Format de fichier non supporté: {ext}")