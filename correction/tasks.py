from celery import shared_task
from correction.models  import DemandeCorrection, SoumissionIA
from resources.models   import Matiere
from abonnement.services import debiter_credit_abonnement
from .pdf_utils        import generer_pdf_corrige


@shared_task(
    queue='child',
    name='correction.tasks.generer_un_exercice'
)
def generer_un_exercice(demande_id, exercice, contexte, matiere_id, vision):
    """
    Sous-tâche child queue : corrige UN exercice.
    """
    # Import tardif pour éviter circular import
    from .ia_utils import generer_corrige_par_exercice

    # 1) Charger la DemandeCorrection (peut être None)
    try:
        demande = DemandeCorrection.objects.get(id=demande_id)
    except DemandeCorrection.DoesNotExist:
        demande = None

    # 2) Charger la Matière
    matiere = Matiere.objects.get(id=matiere_id)

    # 3) Appeler le cœur de la correction
    corrige, graphs = generer_corrige_par_exercice(
        texte_exercice=exercice,
        contexte=contexte,
        matiere=matiere,
        donnees_vision=vision,
        demande=demande
    )
    return {'corrige': corrige, 'graphs': graphs or []}