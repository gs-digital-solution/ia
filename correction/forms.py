from django import forms
from .models import DemandeCorrection
from .models import CustomUser
from resources.models import Pays, SousSysteme

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
            max_size = 3 * 1024 * 1024  # 3 MB
            if fichier.size > max_size:
                raise forms.ValidationError(
                    "Fichier trop volumineux (max 3 Mo)."
                )
        return fichier

#Les questions secretes pour modifier un mot de passe
SECRET_QUESTIONS = [
    "Quel est le prénom de votre mère ?",
    "Quel est le nom de votre premier animal ?",
    "Dans quelle ville êtes-vous né(e) ?",
    "Quel est votre film préféré ?",
    "Quel est le métier rêvé d’enfant ?"
]

#formulaire de gestion des utilisateurs
class CustomUserCreationForm(forms.ModelForm):
    password1 = forms.CharField(widget=forms.PasswordInput, label="Mot de passe")
    password2 = forms.CharField(widget=forms.PasswordInput, label="Confirmer le mot de passe")

    secret_question = forms.ChoiceField(choices=[(q, q) for q in SECRET_QUESTIONS], label="Question secrète")

    class Meta:
        model = CustomUser
        fields = ('first_name', 'whatsapp_number', 'pays', 'sous_systeme', 'secret_question', 'secret_answer', 'password1', 'password2')

    def clean(self):
        cleaned = super().clean()
        pwd1 = cleaned.get("password1")
        pwd2 = cleaned.get("password2")
        if pwd1 != pwd2:
            raise forms.ValidationError("Les mots de passe ne correspondent pas.")
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password1'])
        user.username = self.cleaned_data['whatsapp_number']  # Pour login via numéro
        if commit:
            user.save()
        return user