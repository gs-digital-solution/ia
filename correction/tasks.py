from correction.models import DemandeCorrection
from resources.models import Matiere
from celery import shared_task, chord
from .ia_utils import format_corrige_pdf_structure, flatten_multiline_latex_blocks
from correction.models import SoumissionIA, DemandeCorrection
from abonnement.services import debiter_credit_abonnement
from .pdf_utils import generer_pdf_corrige

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




@shared_task
def callback_final_decoupe(results, demande_id, contexte, matiere_id, exemples_corriges, lecons_contenus):
    """
    Cette tâche est lancée quand tous les generer_un_exercice ont répondu.
    - results est une liste de dicts {'corrige':..., 'graphs':[...]}
    - On assemble tout, génère le PDF, débite le crédit, met à jour SoumissionIA
    """
    # 1) Reconstruction du corrigé complet et agrégation des graphiques
    tous_corriges, tous_graphiques = [], []
    for i, out in enumerate(results, 1):
        corrige = out.get('corrige','')
        graphs  = out.get('graphs',[])
        if corrige:
            tous_corriges.append(f"\n\n## 📝 Exercice {i}\n\n{corrige}")
        if graphs:
            tous_graphiques.extend(graphs)
    corrige_final = "".join(tous_corriges) or "Erreur: Aucun corrigé généré"

    # 2) Générer le PDF
    soum = SoumissionIA.objects.get(demande__id=demande_id)
    pdf_path = generer_pdf_corrige({
        "titre_corrige": contexte,
        "corrige_html": corrige_final,
        "soumission_id": demande_id
    }, demande_id)

    # 3) Débiter le crédit
    if not debiter_credit_abonnement(soum.demande.user):
        soum.statut = 'erreur_credit'
        soum.save()
        return False

    # 4) Mise à jour finale
    soum.statut = 'termine'
    soum.progression = 100
    soum.resultat_json = {
        "corrige_text": corrige_final,
        "pdf_url": pdf_path,
        "graphiques": tous_graphiques,
    }
    soum.save()

    # Aussi mettre à jour l’énoncé corrigé
    dem = DemandeCorrection.objects.get(id=demande_id)
    dem.corrigé = corrige_final
    dem.save()

    return True