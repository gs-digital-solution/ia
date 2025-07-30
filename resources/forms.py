from django import forms
from ckeditor_uploader.widgets import CKEditorUploadingWidget
from .models import Lecon,ExerciceCorrige



class LeconForm(forms.ModelForm):
    """
    Formulaire pour la création et édition de Leçon
    Utilise CKEditor pour le champ de contenu riche
    """
    class Meta:
        model = Lecon
        fields = [
            'matiere',
            'titre', 
            'contenu',
            'fichier_pdf'
        ]
        widgets = {
            'contenu': CKEditorUploadingWidget(
                config_name='default',
                attrs={'class': 'ckeditor-content'}
            )
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Personnalisation supplémentaire des champs si nécessaire
        self.fields['fichier_pdf'].required = False


class ExerciceCorrigeForm(forms.ModelForm):
    class Meta:
        model = ExerciceCorrige
        fields = ['matiere', 'type_exercice', 'intitule', 'contenu_exercice', 'contenu_corrige', 'lecons_associees', 'fichier_exo', 'fichier_corrige']
        widgets = {
            'contenu_exercice': CKEditorUploadingWidget(config_name='default'),
            'contenu_corrige': CKEditorUploadingWidget(config_name='default'),
        }
    lecons_associees = forms.ModelMultipleChoiceField(
        queryset=Lecon.objects.none(),  # On charge dynamiquement en JS
        required=False,
        widget=forms.CheckboxSelectMultiple,
        label="Leçons associées"
    )