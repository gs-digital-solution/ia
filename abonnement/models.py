# Create your models here.
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
from correction.models import CustomUser
#from abonnement.models import PromoCode


class SubscriptionType(models.Model):
    """
    Définit le type d'abonnement disponible (ex: Normal, Promo, Promo avec réduction)
    Tout est MODIFIABLE dans l'admin (prix, quota, etc)
    """
    TYPE_CHOICES = [
        ('normal', 'Standard'),
        ('gratuit_promo', 'Gratuit via code promo'),
        ('reduction_promo', 'Réduction via code promo'),
    ]
    code = models.CharField(max_length=32, choices=TYPE_CHOICES, unique=True)
    nom = models.CharField(max_length=64)
    description = models.TextField(blank=True)
    prix_base = models.DecimalField(max_digits=9, decimal_places=2, default=0.0, help_text="Prix en FCFA ou autre")
    taux_reduction = models.PositiveIntegerField(default=0, help_text="Taux de réduction (%)")
    nombre_exercices_total = models.PositiveIntegerField(default=10, help_text="Quota d'exercices")
    duree_jours = models.PositiveIntegerField(default=30, help_text="Durée de validité abonnement (en jours)")
    actif = models.BooleanField(default=True, help_text="Afficher cette offre ?")

    def __str__(self):
        return f"{self.nom} ({self.get_code_display()})"

    class Meta:
        verbose_name = "Type d'abonnement"
        verbose_name_plural = "Types d'abonnement"


class PromoCode(models.Model):
    """
    Code promo attribuable à un utilisateur, partageable, et utilisable PAR d'autres pour profiter d'une offre.
    """
    code = models.CharField(max_length=12, unique=True)
    proprietaire = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='codes_proprios', on_delete=models.CASCADE)
    partage_count = models.PositiveIntegerField(default=0, help_text="Nb de fois que ce code a été utilisé (partage viral)")
    actif = models.BooleanField(default=True, help_text="Ce code peut-il encore être utilisé/parrainé ?")
    date_creation = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.code} (parr. {self.proprietaire.username})"


class UserAbonnement(models.Model):
    """
    Lien entre utilisateur, type d'abonnement, code promo éventuellement utilisé,
    suivi du crédit restant/durée et du statut.
    """
    utilisateur = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    abonnement = models.ForeignKey(SubscriptionType, on_delete=models.PROTECT)
    code_promo_utilise = models.ForeignKey(PromoCode, null=True, blank=True, on_delete=models.SET_NULL)
    date_debut = models.DateTimeField(auto_now_add=True)
    date_fin = models.DateTimeField(null=True, blank=True)
    exercice_restants = models.PositiveIntegerField(default=0)
    statut = models.CharField(max_length=24, choices=[('actif', 'Actif'), ('epuise', 'Épuisé'), ('expire', 'Expiré')], default='actif')

    def save(self, *args, **kwargs):
        """ Génère automatiquement la date de fin et les crédits si nouvel abo"""
        if not self.pk:
            # Nouveau
            if not self.date_fin and self.abonnement.duree_jours:
                self.date_fin = self.date_debut + timedelta(days=self.abonnement.duree_jours)
            if self.exercice_restants == 0:
                self.exercice_restants = self.abonnement.nombre_exercices_total
        else:
            # Check l'expiration/crédit
            if self.exercice_restants == 0:
                self.statut = 'epuise'
            elif self.date_fin and self.date_fin < timezone.now():
                self.statut = 'expire'
            else:
                self.statut = 'actif'
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.utilisateur.username} | {self.abonnement.nom} ({self.statut})"

    class Meta:
        verbose_name = "Abonnement utilisateur"
        verbose_name_plural = "Abonnements utilisateurs"


class GlobalSubscriptionConfig(models.Model):
    """
    Paramètres généraux modifiables en admin (nb de gratuits, quota spécial, options dispo, etc)
    Singleton (un seul en base)
    """
    nb_gratuit_par_utilisateur = models.PositiveIntegerField(default=1)
    options_gratuit_promo = models.BooleanField(default=True)     # Affichage de l'option
    options_reduction_promo = models.BooleanField(default=True)
    options_normal = models.BooleanField(default=True)
    # Option : quota exceptionnel tous utilisateurs (événement, bonus, ...)

    def save(self, *args, **kwargs):
        # Forcer singleton
        self.pk = 1
        super().save(*args, **kwargs)

    def __str__(self):
        return "Configuration globale abonnements"

    class Meta:
        verbose_name = "Config abonnement (global)"


# Import le bon CustomUser

@receiver(post_save, sender=CustomUser)
def creer_code_promo_pour_nouvel_utilisateur(sender, instance, created, **kwargs):
    """
    Génère et lie automatiquement un PromoCode à chaque nouveau CustomUser,
    avec le code stocké dans user.code_promo
    """
    if created and instance.code_promo:
        # On ne crée le PromoCode que si pas déjà existant (protection double création user)
        PromoCode.objects.get_or_create(
            code=instance.code_promo,
            proprietaire=instance
        )
