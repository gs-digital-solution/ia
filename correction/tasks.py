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


@shared_task(queue='root', name='correction.tasks.finalize_corrige')
def finalize_corrige(results, demande_id, matiere_id, contexte):
       """
       Callback Celery sur la queue 'root' quand tous les generer_un_exercice
       ont renvoyé leur 'corrige' et leurs 'graphs'.
       """
       from correction.models import DemandeCorrection, SoumissionIA
       from .pdf_utils import generer_pdf_corrige
       from abonnement.services import debiter_credit_abonnement

       # 1) Assemblage
       corrige_final = "".join(
           f"\n\n## 📝 Exercice {i+1}\n\n{out['corrige']}"
           for i, out in enumerate(results) if out.get('corrige')
       )
       graph_list = [g for out in results for g in out.get('graphs', [])]

       # 2) Génération PDF
       pdf_path = generer_pdf_corrige({
           "titre_corrige": contexte,
           "corrige_html":  corrige_final,
           "soumission_id": demande_id
       }, demande_id)

       # 3) Débit du crédit
       demande = DemandeCorrection.objects.get(id=demande_id)
       debiter_credit_abonnement(demande.user)

       # 4) Mise à jour en base
       soumission = SoumissionIA.objects.get(demande=demande)
       soumission.resultat_json = {
           'corrige_text': corrige_final,
           'pdf_url':      pdf_path,
           'graphiques':   graph_list
       }
       soumission.statut = 'termine'
       soumission.progression = 100
       soumission.save()

       demande.corrigé = corrige_final
       demande.save()
       return True