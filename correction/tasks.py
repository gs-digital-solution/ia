from celery import shared_task

from correction.models import DemandeCorrection, SoumissionIA
from resources.models import Matiere
from abonnement.services import debiter_credit_abonnement
from .pdf_utils import generer_pdf_corrige


@shared_task
def generer_un_exercice(demande_id, exercice, contexte, matiere_id, vision):
    """
    Sous-tâche pour corriger un seul exercice.
    Reprend la fonction cœur en lui passant un vrai 'demande'.
    """
    from .ia_utils import generer_corrige_par_exercice

    demande = _get_demande_correction(demande_id)
    matiere = _get_matiere(matiere_id)

    corrige, graphs = generer_corrige_par_exercice(
        texte_exercice=exercice,
        contexte=contexte,
        matiere=matiere,
        donnees_vision=vision,
        demande=demande
    )

    return {
        'corrige': corrige,
        'graphs': graphs or []
    }


@shared_task
def callback_final_decoupe(results, demande_id, contexte, matiere_id, exemples_corriges, lecons_contenus):
    """
    Callback de chord :
    - Assemble tous les corrigés d'exercices (results)
    - Génère le PDF
    - Débite le crédit
    - Met à jour la soumission et la demande
    """
    corrige_final, tous_graphiques = _assembler_corriges_et_graphiques(results)
    soumission = _get_soumission_ia(demande_id)

    if not _debiter_credits_abonnement(soumission):
        return False

    pdf_path = _generer_pdf_corrige(soumission, contexte, corrige_final, demande_id)
    _mettre_a_jour_soumission(soumission, corrige_final, pdf_path, tous_graphiques)
    _mettre_a_jour_demande_correction(demande_id, corrige_final)

    return True


def _get_demande_correction(demande_id):
    """Récupère une demande de correction ou retourne None si non trouvée."""
    try:
        return DemandeCorrection.objects.get(id=demande_id)
    except DemandeCorrection.DoesNotExist:
        # cas possible si on lance en test sans demande
        return None


def _get_matiere(matiere_id):
    """Récupère la matière correspondante."""
    return Matiere.objects.get(id=matiere_id)


def _assembler_corriges_et_graphiques(results):
    """Assemble tous les corrigés et agrège les graphiques."""
    tous_corriges = []
    tous_graphiques = []

    for index, resultat in enumerate(results, 1):
        corrige = resultat.get('corrige', '')
        graphs = resultat.get('graphs', [])

        if corrige:
            exercice_formate = f"\n\n## 📝 Exercice {index}\n\n{corrige}"
            tous_corriges.append(exercice_formate)
            tous_graphiques.extend(graphs)

    corrige_final = "".join(tous_corriges) or "Erreur: aucun corrigé généré"
    return corrige_final, tous_graphiques


def _get_soumission_ia(demande_id):
    """Récupère la soumission IA correspondante."""
    return SoumissionIA.objects.get(demande__id=demande_id)


def _debiter_credits_abonnement(soumission):
    """Débite les crédits d'abonnement et gère les erreurs."""
    if not debiter_credit_abonnement(soumission.demande.user):
        soumission.statut = 'erreur_credit'
        soumission.save()
        return False
    return True


def _generer_pdf_corrige(soumission, contexte, corrige_final, demande_id):
    """Génère le PDF du corrigé."""
    donnees_pdf = {
        "titre_corrige": contexte,
        "corrige_html": corrige_final,
        "soumission_id": demande_id
    }
    return generer_pdf_corrige(donnees_pdf, demande_id)


def _mettre_a_jour_soumission(soumission, corrige_final, pdf_path, graphiques):
    """Met à jour la soumission IA avec les résultats finaux."""
    soumission.statut = 'termine'
    soumission.progression = 100
    soumission.resultat_json = {
        'corrige_text': corrige_final,
        'pdf_url': pdf_path,
        'graphiques': graphiques,
    }
    soumission.save()


def _mettre_a_jour_demande_correction(demande_id, corrige_final):
    """Met à jour la demande de correction avec le corrigé final."""
    demande = DemandeCorrection.objects.get(id=demande_id)
    demande.corrigé = corrige_final
    demande.save()