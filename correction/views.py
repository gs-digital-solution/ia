from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

from abonnement.models import UserAbonnement
from .forms import DemandeCorrectionForm, ProfilUserForm
from .models import DemandeCorrection
from resources.models import SousSysteme, Departement, Classe, Matiere, TypeExercice, Lecon, ExerciceCorrige
from .models import AppConfig
from django.contrib import messages
from django.urls import reverse
from .ia_utils import generer_corrige_ia_et_graphique, tracer_graphique
import re
from .forms import CustomUserCreationForm, SECRET_QUESTIONS  # Import la liste
from resources.models import Pays, SousSysteme
from django.contrib.auth import get_user_model
from .decorators import role_required
from django.urls import reverse
from abonnement.services import user_abonnement_actif

from .decorators import only_admin
from .models import CustomUser
from django.http import HttpResponseRedirect
from .ia_utils import generer_corrige_ia_et_graphique_async

from .models import FeedbackCorrection, DemandeCorrection
from .forms import FeedbackCorrectionForm
from django.utils import timezone


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
def extraire_texte_fichier(fichier):
    pass


@login_required
def soumettre_exercice(request):
    # Vérifie si la correction est bloquée par l'admin
    app_config = AppConfig.objects.first()
    if app_config and not app_config.correction_enabled:
        msg = app_config.message_bloquant or "La fonctionnalité correction est temporairement indisponible. Merci de mettre à jour l'application ou de réessayer plus tard."
        return render(request, 'correction/blocage_correction.html', {'message_bloquant': msg})

    # ---- AJOUT ICI : vérification abonnement ---
    # Vérification abonnement et quota
    abonnement = UserAbonnement.objects.filter(
        utilisateur=request.user,
        statut='actif',
        exercice_restants__gt=0,
        date_fin__gt=timezone.now()
    ).first()

    if not abonnement:
        # Aucun abonnement actif ou quota épuisé
        if not abonnement:
            return redirect('correction:plus_de_credit')

    if request.method == "POST":
        form = DemandeCorrectionForm(request.POST, request.FILES)
        if form.is_valid():
            demande = form.save(commit=False)
            demande.user = request.user
            demande.save()
            form.save_m2m()

            # ---------- Pipeline IA TRAITEMENT DIRECT (synchrone) ----------
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

            # --- Génère le corrigé IA ici (le code peut prendre du temps ici !) ---
            corrige_txt, graph_list = generer_corrige_ia_et_graphique(
                texte_enonce,
                contexte,
                lecons_contenus=lecons_contenus,
                exemples_corriges=exemples_corriges,
                matiere=demande.matiere
            )
            demande.corrigé = corrige_txt
            demande.save()
            # Décrémenter le quota
            abonnement.exercice_restants -= 1
            if abonnement.exercice_restants <= 0:
                abonnement.statut = 'epuise'
            abonnement.save()
            return redirect('correction:voir_corrige', demande_id=demande.id)
    else:
        # Préremplissage intelligent (pays, sous-système via User)
        user = request.user
        initial = {}
        if hasattr(user, 'pays') and user.pays_id:
            initial['pays'] = user.pays_id
        if hasattr(user, 'sous_systeme') and user.sous_systeme_id:
            initial['sous_systeme'] = user.sous_systeme_id
        form = DemandeCorrectionForm(initial=initial)

    # En GET ou en erreur de formulaire, on revient sur la page soumettre classique en affichant credit restant
        return render(request, "correction/soumettre.html", {
       "form": form,
       "credit_restants": abonnement.exercice_restants if abonnement else 0
   })

# Vue de consultation du corrigé (inchangée)
@login_required
def voir_corrige(request, demande_id):
    demande = get_object_or_404(DemandeCorrection, id=demande_id, user=request.user)
    app_config = AppConfig.objects.first()
    pdf_enabled = app_config.pdf_enabled if app_config else False

    # Vérifie si l'utilisateur a déjà laissé un feedback sur ce corrigé
    try:
        feedback = FeedbackCorrection.objects.get(user=request.user, correction=demande)
        has_feedback = True
    except FeedbackCorrection.DoesNotExist:
        feedback = None
        has_feedback = False

    form = None
    if not has_feedback and request.user.is_authenticated:
        form = FeedbackCorrectionForm()

    return render(request, "correction/voir_corrige.html", {
        "demande": demande,
        "pdf_enabled": pdf_enabled,
        "has_feedback": has_feedback,
        "form": form,
        "correction": demande,  # utile pour le feedback
    })

# Historique utilisateur (inchangé)
@login_required
def historique(request):
    demandes = DemandeCorrection.objects.filter(user=request.user).order_by('-date_soumission')
    return render(request, "correction/historique.html", {'demandes': demandes})
    # Récupère l’abonnement actif ou le dernier abonnement utilisé
    abonnement = (
    UserAbonnement.objects.filter(utilisateur=request.user)
        .order_by('-date_debut').first()
    )
    return render(request, "correction/historique.html", {
        'demandes': demandes,
        'abonnement': abonnement
    })

# Suppression d'une correction (inchangé)
@login_required
def supprimer_demande(request, demande_id):
    demande = get_object_or_404(DemandeCorrection, id=demande_id, user=request.user)
    if request.method == "POST":
        demande.delete()
        messages.success(request, "L'exercice et son corrigé ont bien été supprimés.")
        return redirect('correction:historique')
    return render(request, "correction/confirm_supprimer.html", {"demande": demande})


#@login_required ( protection retirée pour pas obligé que la création de compte nécessite qu'on soit connecté
def inscription(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Compte créé avec succès ! Connectez-vous ci-dessous.")
            return redirect('correction:login')  # Redirige vers le login après succès
    else:
        form = CustomUserCreationForm()
    return render(request, "correction/inscription.html", {
        "form": form,
        "secret_questions": SECRET_QUESTIONS
    })

# vue pour mot de passe oublié
def mot_de_passe_oublie(request):
    if request.method == 'POST':
        whatsapp = request.POST.get('whatsapp_number', '').strip()
        user = get_user_model().objects.filter(whatsapp_number=whatsapp).first()
        if user:
            # Redirection vers reset-password AVEC whatsapp en GET
            return redirect(f'{reverse("correction:reset_password")}?whatsapp={whatsapp}')
        else:
            messages.error(request, "Numéro WhatsApp inconnu ou non inscrit.")
            return redirect('correction:mot_de_passe_oublie')  # <-- Redirect ici pour garder le message
    return render(request, "correction/mot_de_passe_oublie.html")


# vue pour reset password
def reset_password(request):
    whatsapp = request.GET.get('whatsapp') or request.POST.get('whatsapp')
    user = get_user_model().objects.filter(whatsapp_number=whatsapp).first()
    if request.method == 'GET':
        if user:
            return render(request, "correction/reset_password.html", {
                "question": user.secret_question,
                "whatsapp": whatsapp
            })
        else:
            messages.error(request, "Session expirée ou numéro manquant.")
            return redirect('correction:mot_de_passe_oublie')

    if request.method == 'POST':
        answer = request.POST.get('secret_answer')
        new_pwd = request.POST.get('new_password')
        if user and user.check_secret(answer):
            user.set_password(new_pwd)
            user.save()
            messages.success(request, "Mot de passe réinitialisé avec succès ! Connectez-vous ci-dessous.")
            return redirect('correction:login')  # Redirect pour message de succès
        else:
            error_msg = "Réponse incorrecte." if user else "Numéro WhatsApp inconnu."
            return render(request, "correction/reset_password.html", {
                "question": user.secret_question if user else "Question inconnue",
                "whatsapp": whatsapp,
                "message": error_msg,
            })


@login_required
def profil(request):
    user = request.user
    if request.method == 'POST':
        form = ProfilUserForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profil mis à jour avec succès !")
            return redirect('correction:profil')
    else:
        form = ProfilUserForm(instance=user)
    return render(request, 'correction/profil.html', {
        'form': form,
        'gmail': user.gmail,
        'code_promo': user.code_promo,
    })


# préparation des vues pour chaque role
@role_required('prof', 'admin')
def espace_prof(request):
    return render(request, 'correction/espace_prof.html')  # template à créer selon besoin

@role_required('admin')
def espace_admin(request):
    # Page ultra-réservée
    return render(request, 'correction/espace_admin.html')  # template à créer

@role_required('admin, investisseur')
def espace_admin(request):
    # Page ultra-réservée
    return render(request, 'correction/espace_investisseur.html')  # template à créer



@only_admin
def admin_dashboard(request):
    n_users = CustomUser.objects.count()
    n_admin = CustomUser.objects.filter(role='admin').count()
    n_prof = CustomUser.objects.filter(role='prof').count()
    n_eleve = CustomUser.objects.filter(role='eleve').count()
    n_invest = CustomUser.objects.filter(role='investisseur').count()
    n_corriges = DemandeCorrection.objects.count()
    return render(request, "correction/admin_dashboard.html", {
        "n_users": n_users, "n_admin": n_admin, "n_prof": n_prof,
        "n_eleve": n_eleve, "n_invest": n_invest, "n_corriges": n_corriges
    })

@only_admin
def admin_users_list(request):
    role = request.GET.get("role", "")
    q = CustomUser.objects.all()
    if role:
        q = q.filter(role=role)
    q = q.order_by('-date_joined')
    return render(request, "correction/admin_users.html", {
        "users": q, "role_filter": role
    })

@only_admin
def user_toggle_active(request, user_id):
    user = CustomUser.objects.get(pk=user_id)
    user.is_active = not user.is_active
    user.save()
    messages.success(request, f"Utilisateur {'activé' if user.is_active else 'désactivé'}")
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))

@only_admin
def user_delete(request, user_id):
    user = CustomUser.objects.get(pk=user_id)
    user.delete()
    messages.success(request, "Utilisateur supprimé")
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))

# Vues pour profil utilisateur côté admin et historique (pour admin avancée)

@only_admin
def user_profil_admin(request, user_id):
    from .models import CustomUser
    user = CustomUser.objects.get(pk=user_id)
    return render(request, "correction/user_profil_admin.html", {"user": user})

@only_admin
def user_corrections(request, user_id):
    from .models import CustomUser, DemandeCorrection
    user = CustomUser.objects.get(pk=user_id)
    corrections = DemandeCorrection.objects.filter(user=user).order_by('-date_soumission')
    return render(request, "correction/user_corrections.html", {"user": user, "corrections": corrections})




# Vues pour gestion feedback coté admin native django
@only_admin
def admin_feedback_list(request):
    feedbacks = FeedbackCorrection.objects.select_related('user', 'correction').order_by('-created_at')
    user_id = request.GET.get('user')
    corrig_id = request.GET.get('corrig')

    if user_id:
        feedbacks = feedbacks.filter(user__id=user_id)
    if corrig_id:
        feedbacks = feedbacks.filter(correction__id=corrig_id)

    return render(request, "correction/admin_feedbacks.html", {
        "feedbacks": feedbacks,
        "user_id": user_id,
        "corrig_id": corrig_id,
        "users": CustomUser.objects.all(),
        "corrections": DemandeCorrection.objects.all()
    })
# Vues pour gestion feedback coté utilisateur
@login_required
def donner_feedback(request, demande_id):
    correction = get_object_or_404(DemandeCorrection, id=demande_id)
    fb, created = FeedbackCorrection.objects.get_or_create(user=request.user, correction=correction)
    if request.method == 'POST':
        form = FeedbackCorrectionForm(request.POST, instance=fb)
        if form.is_valid():
            form.save()
            from django.contrib import messages
            messages.success(request, "Merci pour votre retour !")
            return redirect('correction:voir_corrige', demande_id=demande_id)
    else:
        form = FeedbackCorrectionForm(instance=fb)
    return render(request, 'correction/feedback.html', {'form': form, 'correction': correction})


# les deux Vues ci-dessous sont pour attente traitement ( la deuxièe est non utilisée à cause des difficulté avec celery)
@login_required
def attente_traitement(request, demande_id):
    #demande = get_object_or_404(DemandeCorrection, id=demande_id, user=request.user)
    return render(request, "correction/traitement_en_cours.html",
                  {"demande_id": demande_id, "delai_redirection": 60})



@login_required
def etat_traitement_ajax(request, demande_id):
    from .models import DemandeCorrection
    demande = get_object_or_404(DemandeCorrection, id=demande_id, user=request.user)
    # On vérifie : corrigé présent/non nul ?
    done = (demande.corrigé and demande.corrigé.strip() != "")
    return JsonResponse({"done": done})




@login_required
def plus_de_credit(request):
    """
    Affiche un message d'avertissement UX + boutons vers le paiement et les codes promo.
    """
    return render(request, "correction/plus_de_credit.html")



