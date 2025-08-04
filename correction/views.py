
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .forms import DemandeCorrectionForm
from .models import DemandeCorrection
from resources.models import SousSysteme, Departement, Classe, Matiere, TypeExercice, Lecon, ExerciceCorrige
from .models import AppConfig
from django.contrib import messages
from django.urls import reverse
from .ia_utils import generer_corrige_ia_et_graphique, extraire_texte_fichier, tracer_graphique

# ci-dessous les vues (selections) pour soumettre un exercice à corriger


# AJAX
@login_required
def ajax_sous_systemes(request):
    pays_id = request.GET.get('pays_id')
    data = list(SousSysteme.objects.filter(pays_id=pays_id).values('id', 'nom'))
    return JsonResponse(data, safe=False)

@login_required
def ajax_departements(request):
    pays_id = request.GET.get('pays_id')
    data = list(Departement.objects.filter(pays_id=pays_id).values('id', 'nom'))
    return JsonResponse(data, safe=False)

@login_required
def ajax_classes(request):
    ss_id = request.GET.get('ss_id')
    data = list(Classe.objects.filter(sous_systeme_id=ss_id).values('id', 'nom'))
    return JsonResponse(data, safe=False)

@login_required
def ajax_matieres(request):
    classe_id = request.GET.get('classe_id')
    data = list(Matiere.objects.filter(classe_id=classe_id).values('id', 'nom'))
    return JsonResponse(data, safe=False)

@login_required
def ajax_types_exercices(request):
    departement_id = request.GET.get('departement_id')
    data = list(TypeExercice.objects.filter(departement_id=departement_id).values('id', 'nom'))
    return JsonResponse(data, safe=False)

@login_required
def ajax_lecons(request):
    matiere_id = request.GET.get('matiere_id')
    lecons = Lecon.objects.filter(matiere_id=matiere_id)
    data = list(lecons.values('id', 'titre'))
    return JsonResponse(data, safe=False)

# ci-dessous la vue pour la soumission d'un exercice

@login_required
def soumettre_exercice(request):
    if request.method == "POST":
        form = DemandeCorrectionForm(request.POST, request.FILES)
        if form.is_valid():
            demande = form.save(commit=False)
            demande.user = request.user
            demande.save()
            form.save_m2m()

            contexte = (
                f"Pays : {demande.pays.nom if demande.pays else 'NC'}, "
                f"Sous-système : {demande.sous_systeme.nom if demande.sous_systeme else 'NC'}, "
                f"Classe : {demande.classe.nom if demande.classe else 'NC'}, "
                f"Matière : {demande.matiere.nom if demande.matiere else 'NC'}, "
                f"Type d'exercice : {demande.type_exercice.nom if demande.type_exercice else 'NC'}"
            )

            lecons_contenus = list(demande.lecons.all().values_list('titre', 'contenu'))
            exemples_corriges = list(
                ExerciceCorrige.objects.filter(
                    matiere=demande.matiere,
                    type_exercice=demande.type_exercice,
                ).values_list('contenu_corrige', flat=True)[:3]  # Limité à 3 exemples pour la taille du prompt
            )

            texte_enonce = extraire_texte_fichier(
                demande.fichier) if demande.fichier else "Aucun énoncé soumis ou extraction impossible."

            try:
                retour_corrige = generer_corrige_ia_et_graphique(
                    texte_enonce,
                    contexte,
                    lecons_contenus=lecons_contenus,
                    exemples_corriges=exemples_corriges
                )
                if isinstance(retour_corrige, tuple) and len(retour_corrige) == 2:
                    corrige_txt, graph_info = retour_corrige
                else:
                    corrige_txt, graph_info = str(retour_corrige), None
            except Exception as e:
                corrige_txt, graph_info = f"[ERREUR IA] Correction indisponible. ({e})", None

            img_url = ""
            if graph_info and isinstance(graph_info, dict) and graph_info != {} and "expression" in graph_info and \
                    graph_info["expression"]:
                nom_g = f"graph_corrige_{demande.id}.png"
                img_url = tracer_graphique(graph_info, nom_g)

                if img_url:
                    img_html = f'<div style="text-align:center;margin-top:0.6em;"><img src="/media/{img_url}" style="max-width:97%;border-radius:8px;" alt="Graphique corrigé"></div>'
                    corrige_txt = corrige_txt + "\n" + img_html

            demande.corrigé = corrige_txt
            demande.save()
            return redirect('correction:voir_corrige', demande_id=demande.id)
    else:
        form = DemandeCorrectionForm()

    return render(request, "correction/soumettre.html", {"form": form})


# ci-dessous la vue pour afficher le corrigé d'un exercice soumis
@login_required
def voir_corrige(request, demande_id):
    demande = get_object_or_404(DemandeCorrection, id=demande_id, user=request.user)
    # Récupère le paramètre pdf_enabled
    app_config = AppConfig.objects.first()
    pdf_enabled = app_config.pdf_enabled if app_config else False
    return render(request, "correction/voir_corrige.html", {"demande": demande, "pdf_enabled": pdf_enabled})


# ci-dessous la vue pour afficher l'historique d'un utilisateur
@login_required
def historique(request):
    demandes = DemandeCorrection.objects.filter(user=request.user).order_by('-date_soumission')
    return render(request, "correction/historique.html", {'demandes': demandes})


# ci-dessous la vue pour afficher le message de confirmation de suppression d'un historique
@login_required
def supprimer_demande(request, demande_id):
    demande = get_object_or_404(DemandeCorrection, id=demande_id, user=request.user)
    if request.method == "POST":
        demande.delete()
        messages.success(request, "L'exercice et son corrigé ont bien été supprimés.")
        return redirect('correction:historique')
    return render(request, "correction/confirm_supprimer.html", {"demande": demande})


