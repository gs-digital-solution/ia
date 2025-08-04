from django.db import models

# Create your models here.
from django.db import models
from django.conf import settings
from resources.models import (Pays, SousSysteme, Classe, Matiere, Lecon,Departement,TypeExercice)

class DemandeCorrection(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    pays = models.ForeignKey(Pays, on_delete=models.SET_NULL, null=True)
    sous_systeme = models.ForeignKey(SousSysteme, on_delete=models.SET_NULL, null=True)
    classe = models.ForeignKey(Classe, on_delete=models.SET_NULL, null=True)
    matiere = models.ForeignKey(Matiere, on_delete=models.SET_NULL, null=True)
    departement=models.ForeignKey(Departement,on_delete=models.SET_NULL, null=True)
    type_exercice = models.ForeignKey(TypeExercice, on_delete=models.SET_NULL, null=True)
    lecons = models.ManyToManyField(Lecon, blank=True)
    fichier = models.FileField(upload_to='soumissions/', blank=True, null=True)
    date_soumission = models.DateTimeField(auto_now_add=True)
    # On peut ajouter le corrigé ici après traitement
    corrigé = models.TextField(blank=True)  # Stockage d’un corrigé (format texte/HTML)

    def __str__(self):
        return f"{self.user.username} - {self.date_soumission:%d/%m/%Y %H:%M}"

#pour activer ou désactiver le bouton PDF qui imprime le corrigé
class AppConfig(models.Model):
    pdf_enabled = models.BooleanField(default=True, verbose_name="Afficher le bouton PDF")

    def save(self, *args, **kwargs):
        # Toujours un seul objet de config
        self.pk = 1
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = 'Paramètre Application'