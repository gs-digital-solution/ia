from django.db import models

# Create your models here.
from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from resources.models import Pays, SousSysteme, Classe, Matiere, Lecon, Departement, TypeExercice

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
    from django.conf import settings
    enonce_texte = models.TextField(blank=True, default="")


    def __str__(self):
        return f"{self.user.username} - {self.date_soumission:%d/%m/%Y %H:%M}"

#pour activer ou désactiver le bouton PDF qui imprime le corrigé
class AppConfig(models.Model):
    pdf_enabled = models.BooleanField(default=True, verbose_name="Afficher le bouton PDF")
    correction_enabled = models.BooleanField(default=True, verbose_name="Autoriser la soumission/correction")
    message_bloquant = models.TextField(default="", blank=True, verbose_name="Message personnalisé de blocage")
    # ...
    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)
    class Meta:
        verbose_name = 'Paramètre Application'

    def save(self, *args, **kwargs):
        # Toujours un seul objet de config
        self.pk = 1
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = 'Paramètre Application'


#pour la gestion des utilisateurs
class CustomUser(AbstractUser):
    whatsapp_number = models.CharField(max_length=20, unique=True)
    pays = models.ForeignKey(Pays, null=True, blank=True, on_delete=models.SET_NULL, related_name="users")
    sous_systeme = models.ForeignKey(SousSysteme, null=True, blank=True, on_delete=models.SET_NULL, related_name="users")
    secret_question = models.CharField(max_length=128)
    secret_answer = models.CharField(max_length=128)
    first_name = models.CharField(max_length=64)
    gmail = models.EmailField(max_length=254, unique=True, blank=True, null=True)
    device_id = models.CharField(max_length=150, blank=True, null=True)
    code_promo = models.CharField(max_length=6, unique=True, blank=True, null=True)
    ROLE_CHOICES = [
        ('eleve', 'Élève'),
        ('prof', 'Professeur'),
        ('admin', 'Administrateur'),
        ('investisseur', 'investisseur'),
    ]
    role = models.CharField(max_length=100, choices=ROLE_CHOICES, default='eleve', blank=True, null=True)

    def check_secret(self, answer):
        return self.secret_answer.strip().lower() == answer.strip().lower()



#pour la gestion des feedbacks
class FeedbackCorrection(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    correction = models.ForeignKey('DemandeCorrection', on_delete=models.CASCADE, related_name='feedbacks')
    note = models.IntegerField(
        choices=[(5, "⭐⭐⭐⭐⭐"), (4, "⭐⭐⭐⭐"), (3, "⭐⭐⭐"), (2, "⭐⭐"), (1, "⭐")],
        default=5
    )
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user','correction')

    def __str__(self):
        return f"FB by {self.user} on {self.correction} - {self.note}"

### *B. Modèle Historique des connexions device*
class DeviceConnectionHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="device_histories")
    device_id = models.CharField(max_length=150)
    connection_date = models.DateTimeField(auto_now_add=True)
    successful = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user} - {self.device_id} - {self.connection_date} - {'OK' if self.successful else 'REFUSÉ'}"

### *C. Modèle pour la demande de migration device*
class DeviceMigrationRequest(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='migration_requests')
    previous_device_id = models.CharField(max_length=150)
    new_device_id = models.CharField(max_length=150)
    justification = models.TextField(blank=True)  # Raison invoquée par l’utilisateur
    status = models.CharField(max_length=18, choices=[
        ("pending", "En attente"),
        ("accepted", "Validée"),
        ("rejected", "Refusée"),
    ], default="pending")
    request_date = models.DateTimeField(auto_now_add=True)
    decision_date = models.DateTimeField(blank=True, null=True)
    admin_comment = models.TextField(blank=True, null=True)
    # Champ pour date de création du compte utilisateur concerné
    user_date_joined = models.DateTimeField(blank=True, null=True)

    def save(self, *args, **kwargs):
        # Si pas déjà renseigné, ajoute la date inscrite du user
        if not self.user_date_joined and self.user:
            self.user_date_joined = self.user.date_joined
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Migration {self.user} : {self.previous_device_id} -> {self.new_device_id} ({self.status})"

class SoumissionIA(models.Model):
        user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
        demande = models.ForeignKey(DemandeCorrection, on_delete=models.CASCADE)
        statut = models.CharField(max_length=200, choices=[
            ('en_attente', 'En attente'),
            ('extraction', 'Extraction texte'),
            ('analyse_ia', 'Analyse IA'),
            ('generation_graphiques', 'Génération graphiques'),
            ('formatage_pdf', 'Formatage PDF'),
            ('termine', 'Terminé'),
            ('erreur', 'Erreur')
        ], default='en_attente')
        progression = models.IntegerField(default=0)  # 0-100
        date_creation = models.DateTimeField(auto_now_add=True)
        date_maj = models.DateTimeField(auto_now=True)
        resultat_json = models.JSONField(null=True, blank=True)

        def __str__(self):
            return f"{self.user.username} - {self.statut} ({self.progression}%)"

