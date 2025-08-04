from django import forms
from .models import DemandeCorrection

class DemandeCorrectionForm(forms.ModelForm):
    class Meta:
        model = DemandeCorrection
        fields = [
            'pays', 'sous_systeme', 'departement', 'classe',
            'matiere', 'type_exercice', 'fichier', 'lecons'
        ]
    def clean_fichier(self):
        fichier = self.cleaned_data.get('fichier')
        if fichier:
            content_type = fichier.content_type
            if content_type not in ['application/pdf', 'image/jpeg', 'image/png']:
                raise forms.ValidationError(
                    "Merci de déposer un fichier PDF ou une image (.jpeg, .png) uniquement."
                )
            # Option : limite de taille (en octets, ici 8 Mo)
            max_size = 3 * 1024 * 1024  # 8 MB
            if fichier.size > max_size:
                raise forms.ValidationError(
                    "Fichier trop volumineux (max 8 Mo)."
                )
        return fichier