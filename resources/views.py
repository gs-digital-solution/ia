from django.http import JsonResponse
from django.shortcuts import render , get_object_or_404,redirect
from django.db import IntegrityError
from .models import Pays, SousSysteme, Classe, Matiere, Lecon,ExerciceCorrige,TypeExercice,Departement
from .forms import LeconForm
from weasyprint import HTML
from django.http import HttpResponse
from .forms import ExerciceCorrigeForm
from django import forms
from ckeditor_uploader.widgets import CKEditorUploadingWidget

def sous_systeme_par_pays(request, pays_id):
    sous_systemes = SousSysteme.objects.filter(pays_id=pays_id)
    data = [{"id": ss.id, "nom": ss.nom} for ss in sous_systemes]
    return JsonResponse(data, safe=False)


def ajouter_classe(request):
    pays = Pays.objects.all()
    message = ""

    if request.method == "POST":
        pays_id = request.POST.get("pays")
        ss_id = request.POST.get("sous_systeme")
        nom = request.POST.get("nom")
        code = request.POST.get("code")

        if pays_id and ss_id and nom and code:
            try:
                ss = SousSysteme.objects.get(id=ss_id)
                Classe.objects.create(
                    sous_systeme=ss,
                    nom=nom,
                    code=code
                )
                message = "Classe enregistrée avec succès !"
            except IntegrityError:
                message = "Erreur : il existe déjà une classe avec ce code pour ce sous-système."

            # Après la création (ou l'erreur), on recharge quand même le formulaire
            return render(
                request,
                'resources/ajouter_classe.html',
                {'pays': pays, 'message': message}
            )
        else:
            message = "Tous les champs sont obligatoires."
            return render(
                request,
                'resources/ajouter_classe.html',
                {'pays': pays, 'message': message}
            )

    return render(
        request,
        'resources/ajouter_classe.html',
        {'pays': pays} if not message else {'pays': pays, 'message': message}
    )

#ci-dessous, la vue dynamique pour ajouter une matière

def sous_systeme_par_pays(request, pays_id):
    sous_systemes = SousSysteme.objects.filter(pays_id=pays_id)
    data = [{"id": ss.id, "nom": ss.nom} for ss in sous_systemes]
    return JsonResponse(data, safe=False)

def classes_par_sous_systeme(request, ss_id):
    classes = Classe.objects.filter(sous_systeme_id=ss_id)
    data = [{"id": c.id, "nom": c.nom, "code": c.code} for c in classes]
    return JsonResponse(data, safe=False)

def ajouter_matiere(request):
    pays = Pays.objects.all()
    message = ""
    if request.method == "POST":
        pays_id = request.POST.get("pays")
        ss_id = request.POST.get("sous_systeme")
        classe_id = request.POST.get("classe")
        nom = request.POST.get("nom")
        code = request.POST.get("code")

        if pays_id and ss_id and classe_id and nom and code:
            try:
                classe = Classe.objects.get(id=classe_id)
                Matiere.objects.create(classe=classe, nom=nom, code=code)
                message = "Matière enregistrée avec succès !"
            except IntegrityError:
                message = "Erreur : il existe déjà une matière avec ce code pour cette classe."
            except Classe.DoesNotExist:
                message = "Erreur : classe inexistante (recharge la page ?)"
        else:
            message = "Tous les champs sont obligatoires."
    return render(request, 'resources/ajouter_matiere.html', {'pays': pays, 'message': message})

#ci-dessous, la vue dynamique pour ajouter une lecon
def ajouter_lecon(request):
    pays = Pays.objects.all()
    message = ""
    lecon_id = None
    edit_id = request.GET.get('edit_id')
    lecon_instance = None
    # Pour édition d'une leçon
    if edit_id:
        lecon_instance = get_object_or_404(Lecon, id=edit_id)
    if request.method == "POST":
        form = LeconForm(request.POST, request.FILES, instance=lecon_instance)
        print('POST:', request.POST)
        print('Valeur reçue pour matière:', request.POST.get("matiere"))
        if form.is_valid():
            try:
                lecon = form.save()
                lecon_id = lecon.id
                message = "Leçon enregistrée avec succès !" if not edit_id else "Leçon modifiée avec succès !"
                form = LeconForm()
            except IntegrityError:
                message = "Erreur : une leçon avec ce titre existe déjà pour cette matière."
        else:
            message = "Merci de vérifier les champs saisis."
    else:
        form = LeconForm(instance=lecon_instance)
    return render(request, 'resources/ajouter_lecon.html', {
        'pays': pays,
        'form': form,
        'message': message,
        'lecon_id': lecon_id or edit_id
    })

# Voir le PDF
def lecon_to_pdf(request, lecon_id):
    lecon = get_object_or_404(Lecon, id=lecon_id)
    html_content = f"""
    <html>
    <head>
      <meta charset="UTF-8">
      <style>body {{ font-family: sans-serif; margin: 2em; }} h1 {{ color:#1b7f21; }}</style>
    </head>
    <body>
      <h1>{lecon.titre}</h1>
      {lecon.contenu}
    </body>
    </html>
    """
    pdf = HTML(string=html_content, base_url=request.build_absolute_uri('/')).write_pdf()
    resp = HttpResponse(pdf, content_type='application/pdf')
    resp['Content-Disposition'] = f'inline; filename="Lecon_{lecon.id}.pdf"'
    return resp

# 👇 Sans changement pour les API JS déjà présentes
def sous_systeme_par_pays(request, pays_id):
    sous_systemes = SousSysteme.objects.filter(pays_id=pays_id)
    return JsonResponse([{"id": s.id, "nom": s.nom} for s in sous_systemes], safe=False)

def classes_par_sous_systeme(request, ss_id):
    classes = Classe.objects.filter(sous_systeme_id=ss_id)
    return JsonResponse([{"id": c.id, "nom": c.nom, "code": c.code} for c in classes], safe=False)

def matieres_par_classe(request, classe_id):
    matieres = Matiere.objects.filter(classe_id=classe_id)
    return JsonResponse([{"id": m.id, "nom": m.nom, "code": m.code} for m in matieres], safe=False)

#ci-dessous, la vue dynamique pour ajouter un ExerciceCorrige

def departements_par_pays(request, pays_id):
    departs = Departement.objects.filter(pays_id=pays_id)
    return JsonResponse([{'id': d.id, 'nom': d.nom} for d in departs], safe=False)

def types_exercices_par_departement(request, departement_id):
    types = TypeExercice.objects.filter(departement_id=departement_id)
    return JsonResponse([{'id': t.id, 'nom': t.nom} for t in types], safe=False)

def sous_systeme_par_pays(request, pays_id):
    sous_systemes = SousSysteme.objects.filter(pays_id=pays_id)
    return JsonResponse([{"id": ss.id, "nom": ss.nom} for ss in sous_systemes], safe=False)

def classes_par_sous_systeme(request, ss_id):
    classes = Classe.objects.filter(sous_systeme_id=ss_id)
    return JsonResponse([{"id": c.id, "nom": c.nom, "code": c.code} for c in classes], safe=False)

def matieres_par_classe(request, classe_id):
    matieres = Matiere.objects.filter(classe_id=classe_id)
    return JsonResponse([{"id": m.id, "nom": m.nom, "code": m.code} for m in matieres], safe=False)

def lecons_par_matiere(request, matiere_id):
    lecons = Lecon.objects.filter(matiere_id=matiere_id)
    return JsonResponse([{"id": l.id, "titre": l.titre} for l in lecons], safe=False)

# -------- Formulaire pour ExerciceCorrige --------

class ExerciceCorrigeForm(forms.ModelForm):
    class Meta:
        model = ExerciceCorrige
        fields = ['matiere', 'type_exercice', 'intitule', 'contenu_exercice', 'contenu_corrige', 'lecons_associees', 'fichier_exo', 'fichier_corrige']
        widgets = {
            'contenu_exercice': CKEditorUploadingWidget(config_name='default'),
            'contenu_corrige': CKEditorUploadingWidget(config_name='default'),
        }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rendre le queryset dynamique selon la matière sélectionnée dans le formulaire
        matiere_id = None
        data = self.data or self.initial
        if 'matiere' in data:
            try:
                matiere_id = int(data.get('matiere'))
            except (TypeError, ValueError):
                pass
        elif self.instance and self.instance.matiere_id:
            matiere_id = self.instance.matiere_id
        if matiere_id:
            self.fields['lecons_associees'].queryset = Lecon.objects.filter(matiere_id=matiere_id)
        else:
            self.fields['lecons_associees'].queryset = Lecon.objects.all()

# -------- Vue d'ajout exercice corrigé --------

def ajouter_exercice_corrige(request):
    pays = Pays.objects.all()
    message = ""
    exercice_id = None
    if request.method == "POST":
        print("---- DONNÉES FORMULAIRE POST ----")
        for k, v in request.POST.items():
            print(f"{k}: {v}")
        print("Fichiers:", request.FILES)

        form = ExerciceCorrigeForm(request.POST, request.FILES)
        if form.is_valid():
            ex = form.save(commit=False)
            ex.save()
            form.save_m2m()
            exercice_id = ex.id
            message = "Exercice corrigé enregistré avec succès !"
            form = ExerciceCorrigeForm()
        else:
            print("-- Erreurs de validation --", form.errors)
            message = "Merci de vérifier les champs saisis."
    else:
        form = ExerciceCorrigeForm()
    return render(request, 'resources/ajouter_exercice_corrige.html', {
        'pays': pays,
        'form': form,
        'message': message,
        'exercice_id': exercice_id
    })

