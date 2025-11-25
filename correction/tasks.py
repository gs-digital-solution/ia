from celery import shared_task
from .ia_utils import generer_corrige_par_exercice
from correction.models import DemandeCorrection
from resources.models import Matiere

@shared_task
def generer_un_exercice(demande_id, exercice, contexte, matiere_id, vision):
    """
    récupère l'objet demande depuis son id, puis appelle le correcteur
    pour qu'il dispose d'un vrai 'demande' dans generer_corrige_par_exercice.
    """
    # 1) Récupérer la demande complète
    demande = None
    try:
        demande = DemandeCorrection.objects.get(id=demande_id)
    except Exception:
        # si l'id est None ou introuvable, on laisse demande=None
        pass

    # 2) Récupérer la matière
    mat = Matiere.objects.get(id=matiere_id)

    # 3) Appeler la fonction cœur en passant la vraie demande
    corrige, graphs = generer_corrige_par_exercice(
        texte_exercice=exercice,
        contexte=contexte,
        matiere=mat,
        donnees_vision=vision,
        demande=demande
    )
    return {'corrige': corrige, 'graphs': graphs or []}