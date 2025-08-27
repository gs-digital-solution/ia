import string
import random
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import DeviceMigrationRequest, CustomUser
from django.utils import timezone


def generate_unique_promo_code():
    """ Génère un code promo unique à 6 caractères """
    from .models import CustomUser
    while True:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if not CustomUser.objects.filter(code_promo=code).exists():
            return code

@receiver(post_save, sender=CustomUser)
def set_promo_code(sender, instance, created, **kwargs):
    if created and not instance.code_promo:
        instance.code_promo = generate_unique_promo_code()
        instance.save()

# ce code remplace l'ancien device_id par le nouveau lorsque l'admin valide la demande de migration
@receiver(post_save, sender=DeviceMigrationRequest)
def handle_migration_accept(sender, instance, created, **kwargs):
    if not created and instance.status == "accepted":
        user = instance.user
        prev = user.device_id
        user.device_id = instance.new_device_id
        user.save()
        instance.decision_date = timezone.now()  # Met à jour la date si besoin
        instance.save()
        # (Optionnel : logge l'acceptation ou envoie une notif admin/user)



#pour changer les deviced id des utilisateurs dont la demande de mgration a été validée par l'admin
@receiver(post_save, sender=DeviceMigrationRequest)
def handle_migration_accept(sender, instance, created, **kwargs):
    # On ne traite QUE quand une migration passe à "accepted" (et n'est pas déjà traitée)
    if not created and instance.status == "accepted":
        user = instance.user
        if user.device_id != instance.new_device_id:
            user.device_id = instance.new_device_id
            user.save()
            instance.decision_date = timezone.now()
            instance.save()