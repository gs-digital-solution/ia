from django import forms
from .models import DemandeCorrection
from .models import CustomUser
from resources.models import Pays, SousSysteme
import random
import string
from django import forms
from .models import FeedbackCorrection


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
    "Quel est le nom de votre premier amour ?",
    "Dans quelle ville êtes-vous né(e) ?",
    "Quel est votre film préféré ?",
    "Quel est le nom de votre artiste préféré ?"
]

#formulaire de gestion des utilisateurs
class CustomUserCreationForm(forms.ModelForm):
    password1 = forms.CharField(widget=forms.PasswordInput, label="Mot de passe")
    password2 = forms.CharField(widget=forms.PasswordInput, label="Confirmer le mot de passe")
    secret_question = forms.ChoiceField(choices=[(q, q) for q in SECRET_QUESTIONS], label="Question secrète")
    gmail = forms.EmailField(label="Compte Gmail", required=True)
    class Meta:
        model = CustomUser
        fields = ('first_name', 'whatsapp_number', 'pays', 'sous_systeme', 'gmail', 'secret_question', 'secret_answer', 'password1', 'password2')

    def clean(self):
        cleaned = super().clean()
        pwd1 = cleaned.get("password1")
        pwd2 = cleaned.get("password2")
        whatsapp_number = cleaned.get("whatsapp_number")
        gmail = cleaned.get("gmail")
        if pwd1 != pwd2:
            raise forms.ValidationError("Les mots de passe ne correspondent pas.")
        if whatsapp_number and CustomUser.objects.filter(whatsapp_number=whatsapp_number).exists():
            raise forms.ValidationError("Un compte existe déjà avec ce numéro.")
        if gmail and CustomUser.objects.filter(gmail=gmail).exists():
            raise forms.ValidationError("Un compte existe déjà avec ce Gmail.")
        return cleaned

    def generate_unique_code(self):
        while True:
            part1 = ''.join(random.choices(string.ascii_uppercase, k=3))
            part2 = ''.join(random.choices(string.digits, k=3))
            code = f"{part2}{part1}"
            if not CustomUser.objects.filter(code_promo=code).exists():
                return code

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password1'])
        user.username = self.cleaned_data['whatsapp_number']
        user.code_promo = self.generate_unique_code()
        if commit:
            user.save()
        return user

from .forms import SECRET_QUESTIONS

class ProfilUserForm(forms.ModelForm):
    secret_question = forms.ChoiceField(choices=[(q, q) for q in SECRET_QUESTIONS], label="Question secrète")

    class Meta:
        model = CustomUser
        fields = ('first_name', 'whatsapp_number', 'pays', 'sous_systeme', 'secret_question', 'secret_answer')
        # Ne PAS mettre gmail ni code_promo ici = non éditables

    # Pour empêcher double numéro ?
    def clean_whatsapp_number(self):
        whatsapp = self.cleaned_data['whatsapp_number']
        qs = CustomUser.objects.exclude(pk=self.instance.pk).filter(whatsapp_number=whatsapp)
        if qs.exists():
            raise forms.ValidationError("Ce numéro est déjà associé à un autre compte.")
        return whatsapp




### .PROFIL:  Formulaire de modification (correction/forms.py)*

class ProfilUserForm(forms.ModelForm):
    secret_question = forms.ChoiceField(choices=[(q, q) for q in SECRET_QUESTIONS], label="Question secrète")

    class Meta:
        model = CustomUser
        fields = ('first_name', 'whatsapp_number', 'pays', 'sous_systeme', 'secret_question', 'secret_answer')
        # Ne PAS mettre gmail ni code_promo ici = non éditables

    # Pour empêcher double numéro ?
    def clean_whatsapp_number(self):
        whatsapp = self.cleaned_data['whatsapp_number']
        qs = CustomUser.objects.exclude(pk=self.instance.pk).filter(whatsapp_number=whatsapp)
        if qs.exists():
            raise forms.ValidationError("Ce numéro est déjà associé à un autre compte.")
        return whatsapp



#formulaire de gestion des FEEDBACKS
class FeedbackCorrectionForm(forms.ModelForm):
    class Meta:
        model = FeedbackCorrection
        fields = ['note', 'comment']
        widgets = {
            'note': forms.RadioSelect,
            'comment': forms.Textarea(attrs={'placeholder': 'Votre avis (facultatif)...','rows':2})
        }