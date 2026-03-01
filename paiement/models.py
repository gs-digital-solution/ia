from django.db import models

# Create your models here.
from django.conf import settings
from django.db import models
from resources.models import Pays
from abonnement.models import SubscriptionType
### A. *Paiement Méthodes Publiques (gérées depuis le dashboard admin)*
class PaymentMethod(models.Model):
    """
    Table admin pour toutes les méthodes publiques de paiement.
    (MTN Mobile Money, Orange Money, Lygos, etc.)
    """
    TYPE_CHOICES = [
        ('INTERNE', 'Paiement interne (via app)'),
        ('EXTERNE', 'Paiement externe (redirection)'),
        ('IFRAME', 'Paiement par iframe intégré'),  # NOUVEAU - iKeePay
    ]

    code = models.CharField(max_length=64, unique=True)  # Ex: "PAIEMENTMARCHAND_MTN_CM", "LYGOS_CM"
    pays = models.ForeignKey(Pays, on_delete=models.CASCADE, related_name="payment_methods")
    nom_affiche = models.CharField(max_length=64)  # "MTN Mobile Money" ou "Lygos"
    operateur = models.CharField(max_length=64)  # MTN, Orange, Lygos, Wave...
    type_paiement = models.CharField(max_length=10, choices=TYPE_CHOICES, default='INTERNE')

    # Pour les paiements INTERNES (via app)
    ussd = models.CharField(max_length=64, blank=True)  # *126#, etc.
    service_code = models.CharField(
        max_length=100, blank=True, null=True,
        help_text="Code Touchpay à passer (ex: PAIEMENTMARCHAND_MTN_CM)"
    )

    # Pour les paiements EXTERNES (redirection)
    lien_externe = models.URLField(max_length=500, blank=True, null=True, help_text="URL de redirection")
    instructions_externes = models.TextField(
        blank=True,
        default="Cliquez sur le lien suivant pour payer. Après paiement, faites une capture d'écran du message de paiement et envoyez-la par WhatsApp.",
        help_text="Instructions pour les paiements externes"
    )

    logo_url = models.URLField(blank=True, null=True)
    description = models.TextField(blank=True)
    actif = models.BooleanField(default=True)
    extra_config = models.JSONField(blank=True, null=True)  # Autres params spécifiques par méthode
    priorite = models.IntegerField(default=1, help_text="Ordre d'affichage")

    def __str__(self):
        return f"{self.nom_affiche} ({self.operateur}, {self.pays.code})"

    def est_externe(self):
        return self.type_paiement == 'EXTERNE'

    def est_iframe(self):
        """Nouvelle méthode utilitaire"""
        return self.type_paiement == 'IFRAME'

### B. *Historique des paiements/transactions*
class PaymentTransaction(models.Model):
    """
    Historique de chaque tentative de paiement.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    abonnement = models.ForeignKey(SubscriptionType, on_delete=models.PROTECT)
    payment_method = models.ForeignKey(PaymentMethod, on_delete=models.PROTECT)
    amount = models.DecimalField(max_digits=9, decimal_places=2)
    phone = models.CharField(max_length=20)
    transaction_id = models.CharField(max_length=128, unique=True)
    status = models.CharField(max_length=32, default="INITIATED")  # INITIATED, PROCESSING, SUCCESS, FAIL
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    raw_response = models.JSONField(blank=True, null=True)

    def __str__(self):
        return f"{self.payment_method.nom_affiche} - {self.transaction_id} ({self.status})"
