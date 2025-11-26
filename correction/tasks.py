from celery import shared_task
from correction.models import DemandeCorrection
from resources.models import Matiere


@shared_task
def generer_un_exercice(demande_id, exercice, contexte, matiere_id, vision):
    # Importe à l'intérieur pour casser le circular import
    from .ia_utils import generer_corrige_par_exercice

    # 1) Récupère l'objet demande (ou None)
    demande = None
    try:
        demande = DemandeCorrection.objects.get(id=demande_id)
    except DemandeCorrection.DoesNotExist:
        pass

    # 2) Récupère la Matière
    mat = Matiere.objects.get(id=matiere_id)

    # 3) Appelle la fonction cœur
    corrige, graphs = generer_corrige_par_exercice(
        texte_exercice=exercice,
        contexte=contexte,
        matiere=mat,
        donnees_vision=vision,
        demande=demande
    )
    return {'corrige': corrige, 'graphs': graphs or []}