from django import forms

class CodePromoForm(forms.Form):
    code_promo = forms.CharField(
        label="Code promo reçu",
        required=True,
        max_length=12,
        widget=forms.TextInput(attrs={'placeholder': 'Entrez un code promo différent du votre...'})
    )