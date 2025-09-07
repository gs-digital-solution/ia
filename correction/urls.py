from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import PasswordChangeView
from django.contrib.auth.views import PasswordChangeDoneView
from correction.views import admin_dashboard, admin_users_list, user_toggle_active, user_delete
from .api_views import CorrigeHTMLView, CorrigePDFView

app_name = 'correction'

urlpatterns = [
    path(
        'soumettre/',
        views.soumettre_exercice,
        name='soumettre'),
    path(
        'ajax/sous-systemes/',
        views.ajax_sous_systemes,
        name='ajax_sous_systemes'),
    path(
        'ajax/departements/',
        views.ajax_departements,
        name='ajax_departements'),
    path(
        'ajax/classes/',
        views.ajax_classes,
        name='ajax_classes'),
    path(
        'ajax/matieres/',
        views.ajax_matieres,
        name='ajax_matieres'),
    path(
        'ajax/types-exercices/',
        views.ajax_types_exercices,
        name='ajax_types_exercices'),
    path(
        'ajax/lecons/',
        views.ajax_lecons,
        name='ajax_lecons'),
    path(
    'corrige/<int:demande_id>/',
    views.voir_corrige, name='voir_corrige'),
    path(
    'historique/',
    views.historique,
    name='historique'),
    path(
     'supprimer/<int:demande_id>/',
      views.supprimer_demande,
      name='supprimer_demande'),

    path(
        'inscription/',
        views.inscription,
        name='inscription'),

    path(
        'login/',
        auth_views.LoginView.as_view(template_name='correction/login.html'),
        name='login'),

    path(
        'logout/',
        auth_views.LogoutView.as_view(),
        name='logout'),
    path(
        'mot-de-passe-oublie/',
        views.mot_de_passe_oublie,
        name='mot_de_passe_oublie'),
    path(
        'reset-password/',
        views.reset_password,
        name='reset_password'),

    path(
       'profil/',
        views.profil,
        name='profil'),

    path(
    'password-change/',
        PasswordChangeView.as_view(template_name='correction/password_change.html'),
        name='password_change'),


   path(
       'password-change/done/',
       PasswordChangeDoneView.as_view(template_name='correction/password_change_done.html'),
       name='password_change_done'),

  path
      ('espace-prof/',
       views.espace_prof,
       name='espace_prof'),

path
    ('espace-admin/',
     views.espace_admin,
     name='espace_admin'),


path(
    'admin-dashboard/',
    admin_dashboard,
    name='admin_dashboard'),
path(
    'admin-users/',
    admin_users_list,
    name='admin_users'),
path(
    'admin-user/<int:user_id>/toggle-active/',
    user_toggle_active,
    name='user_toggle_active'),
path(
    'admin-user/<int:user_id>/delete/',
    user_delete,
    name='user_delete'),

path(
    'admin-user/<int:user_id>/profil/',
    views.user_profil_admin,
    name='user_profil_admin'),
path(
    'admin-user/<int:user_id>/corrections/',
    views.user_corrections,
    name='user_corrections'),

path(
    'admin-feedbacks/',
    views.admin_feedback_list,
    name='admin_feedback_list'),

path(
    'corrige/<int:demande_id>/feedback/',
    views.donner_feedback,
    name='donner_feedback' ),

path(
    'traitement/<int:demande_id>/',
    views.attente_traitement,
    name='attente_traitement'),
path(
    'etat-traitement/<int:demande_id>/',
    views.etat_traitement_ajax,
    name='etat_traitement_ajax'),

path(
    'plus-de-credit/',
    views.plus_de_credit,
    name='plus_de_credit'),
# route pour la vraie page web qui sera affichée sur flutter
      path(
            'corrige/<int:soumission_id>/view/',
             CorrigeHTMLView.as_view(),
             name = 'corrige_html_view'),

# route pour imprimer la  page web qui sera affichée sur flutter
path(
     'corrige/<int:soumission_id>/pdf/',
      CorrigePDFView.as_view(),
      name='corrige_pdf_view'),

]

