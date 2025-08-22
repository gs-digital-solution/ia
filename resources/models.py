from django.db import models
from django.core.validators import RegexValidator
from django.utils.translation import gettext_lazy as _
from ckeditor_uploader.fields import RichTextUploadingField

#TABLE PAYS
class Pays(models.Model):
    nom = models.CharField(
        max_length=100,
        verbose_name=_("Nom du pays"),
        help_text=_("Nom complet du pays en français"),
        unique=True
    )

    code = models.CharField(
        max_length=3,
        verbose_name=_("Code ISO Alpha-3"),
        help_text=_("Code pays à 3 lettres (ex: FRA, CMR, USA)"),
        unique=True,
        validators=[
            RegexValidator(
                regex=r'^[A-Z]{3}$',
                message='Le code doit contenir exactement 3 lettres majuscules'
            )
        ]
    )

    indicatif = models.CharField(
        max_length=10,
        verbose_name=_("Indicatif téléphonique"),
        help_text=_("Indicatif international avec le + (ex: +33, +237)"),
        validators=[
            RegexValidator(
                regex='^\+[0-9]{1,4}$',
                message="L'indicatif doit commencer par + et contenir 1 à 4 chiffres"
            )
        ]
    )

    class Meta:
        verbose_name = _("Pays")
        verbose_name_plural = _("Pays")
        ordering = ['nom']
        constraints = [
            models.UniqueConstraint(fields=['code'], name='code_pays_unique'),
            models.UniqueConstraint(fields=['indicatif'], name='indicatif_unique')
        ]

    def __str__(self):
        return f"{self.nom} ({self.code}) {self.indicatif}"

    def save(self, *args, **kwargs):
        # Normalisation des champs
        self.code = self.code.upper()
        self.indicatif = self.indicatif.strip()
        super().save(*args, **kwargs)

    def clean(self):
        from django.core.exceptions import ValidationError

        # Validation supplémentaire
        if not self.code.isalpha():
            raise ValidationError({'code': 'Le code ne doit contenir que des lettres'})

        if not self.indicatif.startswith('+'):
            raise ValidationError({'indicatif': "L'indicatif doit commencer par +"})

#TABLE SOUS-SYSTEME
class SousSysteme(models.Model):
    pays = models.ForeignKey(
        "Pays",
        on_delete=models.CASCADE,
        verbose_name=_("Pays associé"),
        help_text=_("Pays auquel ce sous-système appartient"),
        related_name="sous_systemes"  # Nom pour la relation inverse
    )

    nom = models.CharField(
        max_length=100,
        verbose_name=_("Nom du sous-système"),
        help_text=_("Désignation complète du sous-système"),
        db_index=True  # Index pour améliorer les recherches
    )

    date_creation = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_("Date de création")
    )

    date_mise_a_jour = models.DateTimeField(
        auto_now=True,
        verbose_name=_("Dernière mise à jour")
    )

    class Meta:
        verbose_name = _("Sous-système")
        verbose_name_plural = _("Sous-systèmes")
        ordering = ['pays', 'nom']  # Tri par pays puis par nom
        constraints = [
            models.UniqueConstraint(
                fields=['pays', 'nom'],
                name='unique_sous_systeme_par_pays'
            )
        ]

    def __str__(self):
        return f"{self.nom} ({self.pays.nom})"  # Correction: double underscore

    def save(self, *args, **kwargs):
        self.nom = self.nom.strip()  # Nettoyage des espaces
        super().save(*args, **kwargs)


class Classe(models.Model):
    sous_systeme = models.ForeignKey(
        'SousSysteme',
        on_delete=models.CASCADE,
        verbose_name=_("Sous-système parent"),
        help_text=_("Sous-système auquel cette classe appartient"),
        related_name="classes"  # Permet d'accéder à toutes les classes d'un sous-système
    )

    nom = models.CharField(
        max_length=100,
        verbose_name=_("Nom de la classe"),
        help_text=_("Désignation complète de la classe"),

    )

    code = models.CharField(
        max_length=20,
        verbose_name=_("Code unique"),
        help_text=_("Code d'identification court (2-20 caractères)"),
        db_index=True
    )

    date_creation = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_("Date de création")
    )

    actif = models.BooleanField(
        default=True,
        verbose_name=_("Actif"),
        help_text=_("Indique si cette classe est active")
    )

    class Meta:
        verbose_name = _("Classe")
        verbose_name_plural = _("Classes")
        ordering = ['sous_systeme', 'nom']
        constraints = [
            models.UniqueConstraint(
                fields=['sous_systeme', 'code'],
                name='unique_code_par_sous_systeme'
            )
        ]

    def __str__(self):
        return f"{self.code} - {self.nom} ({self.sous_systeme})"

    @property
    def pays(self):
        """Propriété calculée pour accéder directement au pays associé"""
        return self.sous_systeme.pays

    def save(self, *args, **kwargs):
        # Normalisation des champs
        self.code = self.code.upper().strip()
        self.nom = self.nom.strip()
        super().save(*args, **kwargs)

    def clean(self):
        from django.core.exceptions import ValidationError
        if not self.code.isalnum():
            raise ValidationError({'code': 'Le code ne doit contenir que des caractères alphanumériques'})

class Matiere(models.Model):
    classe = models.ForeignKey(
        'Classe',
        on_delete=models.CASCADE,
        verbose_name="Classe associée",
        help_text="Classe à laquelle cette matière est rattachée"
    )
    nom = models.CharField(
        max_length=100,
        verbose_name="Nom de la matière",
        help_text="Intitulé complet de la matière"
    )
    code = models.CharField(
        max_length=20,
        verbose_name="Code matière",
        help_text="Code court identifiant la matière"
    )

    class Meta:
        verbose_name = "Matière"
        verbose_name_plural = "Matières"
        ordering = ['classe', 'nom']
        constraints = [
            models.UniqueConstraint(
                fields=['classe', 'code'],
                name='unique_code_par_classe',
                violation_error_message="Ce code de matière existe déjà pour cette classe"
            )
        ]

    def __str__(self):
        return f"{self.nom} ({self.code}) - {self.classe.nom}"


class Departement(models.Model):
    pays = models.ForeignKey('Pays', on_delete=models.CASCADE)
    nom = models.CharField(max_length=100)
    def __str__(self):
      return f"{self.nom} ({self.pays.nom})"

class TypeExercice(models.Model):
    departement = models.ForeignKey('Departement', on_delete=models.CASCADE)
    nom = models.CharField(max_length=100)
    def __str__(self):
        return f"{self.nom} ({self.departement.nom})"

class Lecon(models.Model):
    matiere = models.ForeignKey('Matiere', on_delete=models.CASCADE)
    titre = models.CharField(max_length=200)
    contenu = RichTextUploadingField(config_name='default')  # Supporte l’image collée/uploadée
    fichier_pdf = models.FileField(upload_to='lecons_pdfs/', blank=True, null=True)

    def __str__(self):
        return self.titre

class ExerciceCorrige(models.Model):
    matiere = models.ForeignKey('Matiere', on_delete=models.CASCADE)
    type_exercice = models.ForeignKey('TypeExercice', on_delete=models.CASCADE, verbose_name="Type d'exercice")
    intitule = models.CharField(max_length=200, help_text="Titre ou résumé court de l'exercice")
    contenu_exercice = RichTextUploadingField(config_name='default', verbose_name="Énoncé de l'exercice")
    contenu_corrige = RichTextUploadingField(config_name='default', verbose_name="Corrigé complet")
    lecons_associees = models.ManyToManyField('Lecon', blank=True, related_name='exercices', verbose_name="Leçons liées")
    fichier_exo = models.FileField(upload_to='exo_corriges/', blank=True, null=True, verbose_name="Fichier exercice (optionnel)")
    fichier_corrige = models.FileField(upload_to='exo_corriges/', blank=True, null=True, verbose_name="Fichier corrigé (optionnel)")
    date_creation = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Exercice corrigé"
        verbose_name_plural = "Exercices corrigés"
        ordering = ['-date_creation']

    def __str__(self):
        return f"{self.intitule} ({self.matiere.nom} - {self.type_exercice.nom})"