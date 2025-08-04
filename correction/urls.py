from django.urls import path
from . import views

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



]