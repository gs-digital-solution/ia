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
import re
from .forms import CustomUserCreationForm, SECRET_QUESTIONS  # Import la liste
from resources.models import Pays, SousSysteme
from django.contrib.auth import get_user_model

# AJAX views (inchangées)
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


# Modification MAJEURE : injection multi-graphique à l'endroit voulu dans le texte.
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
                ).values_list('contenu_corrige', flat=True)[:3]
            )

            texte_enonce = extraire_texte_fichier(
                demande.fichier
            ) if demande.fichier else "Aucun énoncé soumis ou extraction impossible."

            try:
                corrige_txt, graph_list = generer_corrige_ia_et_graphique(
                    texte_enonce,
                    contexte,
                    lecons_contenus=lecons_contenus,
                    exemples_corriges=exemples_corriges
                )
            except Exception as e:
                corrige_txt, graph_list = f"[ERREUR IA] Correction indisponible. ({e})", None

            print("graph_list =", graph_list)  # DEBUG

            if graph_list and isinstance(graph_list, list):
                img_htmls = []
                img_urls = []
                for idx, graph_info in enumerate(graph_list, 1):
                    img_url = ""
                    print(f">>> graph_info ({idx}) =", graph_info)
                    # Décompacte tout type de dict 'graphique' pour TOUS les graphes (fonction, polygone, histogramme, ...)
                    if graph_info and isinstance(graph_info, dict) and "graphique" in graph_info:
                        graph_info = graph_info["graphique"]
                    # On n'exige PAS "expression", on affiche tout type de graphe généré
                    if graph_info and isinstance(graph_info, dict) and graph_info.get("type"):
                        nom_g = f"graph_corrige_{demande.id}_{idx}.png"
                        img_url = tracer_graphique(graph_info, nom_g)
                        print(f">>> tracer_graphique img_url ({idx}) = {img_url}")
                        if img_url:
                            full_img_url = f'/media/{img_url}'
                            img_html = (
                                f'<div style="text-align:center;margin-top:0.6em;">'
                                f'<img src="{full_img_url}" style="max-width:97%;border-radius:8px;" alt="Graphique généré">'
                                f'</div>'
                            )
                        else:
                            img_html = "<div><b>Erreur génération graphique.</b></div>"
                    else:
                        img_html = ""
                    img_htmls.append(img_html)
                    img_urls.append(img_url)
                # IMPORTANT : remplacer SEULEMENT la première apparition de chaque placeholder
                for idx, img_html in enumerate(img_htmls, 1):
                    corrige_txt = corrige_txt.replace(f"[[GRAPHIC_{idx}]]", img_html, 1)
                # Supprime les placeholders encore résiduels (doublon IA ou bug HTML)
                for idx in range(1, len(img_htmls)+1):
                    corrige_txt = corrige_txt.replace(f"[[GRAPHIC_{idx}]]", "")

            elif graph_list and isinstance(graph_list, dict):
                if "graphique" in graph_list:
                    graph_info = graph_list["graphique"]
                else:
                    graph_info = graph_list
                if graph_info.get("type"):
                    nom_g = f"graph_corrige_{demande.id}.png"
                    img_url = tracer_graphique(graph_info, nom_g)
                    print(">>> tracer_graphique img_url (single dict) =", img_url)
                    if img_url:
                        img_html = (
                            f'<div style="text-align:center;margin-top:0.6em;">'
                            f'<img src="/media/{img_url}" style="max-width:97%;border-radius:8px;" alt="Graphique généré">'
                            f'</div>'
                        )
                        corrige_txt += "\n" + img_html
            print("CORRIGE HTML FINAL :", corrige_txt)
            demande.corrigé = corrige_txt
            demande.save()
            return redirect('correction:voir_corrige', demande_id=demande.id)
    else:
        form = DemandeCorrectionForm()

    return render(request, "correction/soumettre.html", {"form": form})

# Vue de consultation du corrigé (inchangée)
@login_required
def voir_corrige(request, demande_id):
    demande = get_object_or_404(DemandeCorrection, id=demande_id, user=request.user)
    # Récupère le paramètre pdf_enabled
    app_config = AppConfig.objects.first()
    pdf_enabled = app_config.pdf_enabled if app_config else False
    return render(request, "correction/voir_corrige.html", {"demande": demande, "pdf_enabled": pdf_enabled})


# Historique utilisateur (inchangé)
@login_required
def historique(request):
    demandes = DemandeCorrection.objects.filter(user=request.user).order_by('-date_soumission')
    return render(request, "correction/historique.html", {'demandes': demandes})

# Suppression d'une correction (inchangé)
@login_required
def supprimer_demande(request, demande_id):
    demande = get_object_or_404(DemandeCorrection, id=demande_id, user=request.user)
    if request.method == "POST":
        demande.delete()
        messages.success(request, "L'exercice et son corrigé ont bien été supprimés.")
        return redirect('correction:historique')
    return render(request, "correction/confirm_supprimer.html", {"demande": demande})


@login_required
def inscription(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, "correction/inscription.html", {
        "form": form,
        "secret_questions": SECRET_QUESTIONS
    })

# vue pour mot de passe oublié
def mot_de_passe_oublie(request):
    message = ""
    if request.method == 'POST':
        whatsapp = request.POST.get('whatsapp_number')
        user = get_user_model().objects.filter(whatsapp_number=whatsapp).first()
        if user:
            question = user.secret_question
            return render(request, "correction/reset_password.html", {"question": question, "whatsapp": whatsapp})
        else:
            message = "Numéro WhatsApp inconnu ou non inscrit."
    return render(request, "correction/mot_de_passe_oublie.html", {"message": message})

# vue pour reset password
def reset_password(request):
    message = ""
    if request.method == 'POST':
        whatsapp = request.POST.get('whatsapp')
        answer = request.POST.get('secret_answer')
        new_pwd = request.POST.get('new_password')
        user = get_user_model().objects.filter(whatsapp_number=whatsapp).first()
        if user and user.check_secret(answer):
            user.set_password(new_pwd)
            user.save()
            message = "Mot de passe réinitialisé avec succès !!! Vous pouvez maintenant vous connecter."
            return redirect('correction:login')
        else:
            question = user.secret_question if user else "Question inconnue"
            message = "Réponse incorrecte."
            return render(request, "correction/reset_password.html",
                          {"message": message, "question": question, "whatsapp": whatsapp})