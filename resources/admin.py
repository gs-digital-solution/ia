from django.contrib import admin

from .models import Pays
from .models import SousSysteme
from .models import Classe
from .models import Matiere
from .models import Departement
from .models import TypeExercice
from .models import Lecon
from .models import ExerciceCorrige
from .models import PromptIA

@admin.register(Pays)
class PaysAdmin(admin.ModelAdmin):
    list_display = ('nom', 'code', 'indicatif')
    search_fields = ('nom', 'code')
    list_filter = ('code',)

@admin.register(SousSysteme)
class SousSystemeAdmin(admin.ModelAdmin):
    list_display = ('nom', 'pays', 'date_creation')
    list_filter = ('pays',)
    search_fields = ('nom', 'pays__nom')
    autocomplete_fields = ['pays']  # Pour les performances avec beaucoup de pays


@admin.register(Classe)
class ClasseAdmin(admin.ModelAdmin):
    list_display = ('code', 'nom', 'sous_systeme', 'pays', 'actif')
    list_filter = ('actif', 'sous_systeme__pays', 'sous_systeme')
    search_fields = ('code', 'nom', 'sous_systeme__nom')
    list_select_related = ('sous_systeme', 'sous_systeme__pays')  # Optimisation des requêtes

    def pays(self, obj):
        return obj.pays  # Utilise la propriété calculée

    pays.short_description = ('Pays')
    pays.admin_order_field = 'sous_systeme__pays__nom'

@admin.register(Matiere)
class MatiereAdmin(admin.ModelAdmin):
    list_display = ('id', 'nom', 'code', 'classe')
    list_filter = ('classe',)
    search_fields = ('nom', 'code')

@admin.register(Departement)
class DepartementAdmin(admin.ModelAdmin):
    list_display = ('id', 'nom', 'pays')
    list_filter = ('pays',)
    search_fields = ('nom',)


@admin.register(TypeExercice)
class TypeExerciceAdmin(admin.ModelAdmin):
    list_display = ('id', 'nom', 'departement')
    list_filter = ('departement',)
    search_fields = ('nom',)


@admin.register(Lecon)
class LeconAdmin(admin.ModelAdmin):
    list_display = ('id', 'titre', 'matiere')
    list_filter = ('matiere',)
    search_fields = ('titre',)

@admin.register(ExerciceCorrige)
class ExerciceCorrigeAdmin(admin.ModelAdmin):
    list_display = ('intitule', 'matiere', 'type_exercice', 'date_creation')
    list_filter = ('matiere', 'type_exercice')
    search_fields = ('intitule', 'contenu_exercice', 'contenu_corrige')
    readonly_fields = ('date_creation',)
    filter_horizontal = ("lecons_associees",)

@admin.register(PromptIA)
class PromptIAAdmin(admin.ModelAdmin):
    list_display = (
        "matiere", "classe", "sous_systeme", "departement", "type_exercice", "pays", "updated_at"
    )
    search_fields = (
        "matiere__nom", "classe__nom", "sous_systeme__nom", "departement__nom", "type_exercice__nom", "pays__nom"
    )
    list_filter = (
        "pays", "departement", "type_exercice", "sous_systeme", "classe"
    )