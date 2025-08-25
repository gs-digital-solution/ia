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
    (MTN Mobile Money, Orange Money, etc.)
    """
    code = models.CharField(max_length=64, unique=True)   # Ex: "PAIEMENTMARCHAND_MTN_CM"
    pays = models.ForeignKey(Pays, on_delete=models.CASCADE, related_name="payment_methods")
    nom_affiche = models.CharField(max_length=64)         # "MTN Mobile Money"
    operateur = models.CharField(max_length=32)           # MTN, Orange, Wave...
    ussd = models.CharField(max_length=32, blank=True)    # *126#, etc.
    logo_url = models.URLField(blank=True, null=True)
    description = models.TextField(blank=True)
    actif = models.BooleanField(default=True)
    extra_config = models.JSONField(blank=True, null=True)  # Autres params spécifiques par méthode

    def __str__(self):
        return f"{self.nom_affiche} ({self.operateur}, {self.pays.code})"


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
