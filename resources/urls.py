from django.urls import path
from . import views


urlpatterns = [
    path(
        'api/sous-systemes/<int:pays_id>/',
        views.sous_systeme_par_pays,
        name='liste_sous_systemes_par_pays'
    ),
path(
    'ajouter-classe/',
    views.ajouter_classe,
    name='ajouter_classe'),

    path(
        'api/classes/<int:ss_id>/',
        views.classes_par_sous_systeme,
        name='liste_classes_par_ss'),
    path(
        'ajouter-matiere/',
        views.ajouter_matiere,
        name='ajouter_matiere'),

    path(
        'api/matieres/<int:classe_id>/',
        views.matieres_par_classe,
        name='liste_matieres_par_classe'),
    path(
        'ajouter-lecon/',
        views.ajouter_lecon,
        name='ajouter_lecon'),
    path(
        'lecon/<int:lecon_id>/pdf/',
        views.lecon_to_pdf,
        name='lecon_to_pdf'),

    path(
        'api/departements/<int:pays_id>/',
        views.departements_par_pays,
        name='liste_departements_par_pays'),
    path(
        'api/types_exercices/<int:departement_id>/',
        views.types_exercices_par_departement,
        name='liste_types_exercices_par_departement'),
    path(
    'ajouter-exercice-corrige/',
    views.ajouter_exercice_corrige,
    name='ajouter_exercice_corrige'),
    path(
        'api/lecons/<int:matiere_id>/',
        views.lecons_par_matiere,
        name='liste_lecons_par_matiere'),
]