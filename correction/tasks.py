from celery import shared_task

@shared_task
def generer_un_exercice(demande_id, exercice, contexte, matiere_id, vision):
    from correction.ia_utils import generer_corrige_par_exercice
    from resources.models import Matiere
    mat = Matiere.objects.get(id=matiere_id)
    corrige, graphs = generer_corrige_par_exercice(
        exercice, contexte, mat, vision, demande=None
    )
    return {"corrige": corrige, "graphs": graphs}