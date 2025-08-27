
from django.utils import timezone
from .models import PromoCode, SubscriptionType, UserAbonnement, GlobalSubscriptionConfig


def activer_abonnement_via_promo(user, code_promo_saisi):
    """
    Active un abonnement 'gratuit_promo' pour user avec le code saisi,
    en respectant les règles métiers.
    Retourne (abonnement, msg/erreur)
    """
    try:
        promo = PromoCode.objects.get(code=code_promo_saisi, actif=True)
    except PromoCode.DoesNotExist:
        return None, "Ce code promo n'existe pas ou n'est plus actif."

    # 1. Le user n'a pas déjà utilisé ce code
    deja_utilise = UserAbonnement.objects.filter(
        utilisateur=user, code_promo_utilise=promo
    ).exists()
    if deja_utilise:
        return None, "Vous avez déjà utilisé ce code promo."

    # 2. On peut imposer un max d'abos gratuits par user (config global)
    config = GlobalSubscriptionConfig.objects.first()
    quota_gratuit = config.nb_gratuit_par_utilisateur if config else 1
    cpt = UserAbonnement.objects.filter(
        utilisateur=user,
        abonnement__code='gratuit_promo',
    ).count()
    if cpt >= quota_gratuit:
        return None, "Vous avez atteint le maximum d'abonnements gratuits."

    # 3. Récupère le type/grille d'abo 'gratuit_promo'
    try:
        sub_type = SubscriptionType.objects.get(code='gratuit_promo', actif=True)
    except SubscriptionType.DoesNotExist:
        return None, "Aucun abonnement promotionnel n'est proposé actuellement."

    # 4. Crée l'abonnement utilisateur
    abo = UserAbonnement.objects.create(
        utilisateur=user,
        abonnement=sub_type,
        code_promo_utilise=promo,
        exercice_restants=sub_type.nombre_exercices_total,
        date_debut=timezone.now(),
        date_fin=timezone.now() + timezone.timedelta(days=sub_type.duree_jours)
    )

    # 5. (option) Incrémente le nombre d'utilisations sur le PromoCode
    promo.partage_count += 1
    promo.save()

    return abo, "Abonnement gratuit activé avec succès !"



def user_abonnement_actif(user):
    """Retourne True si l'utilisateur a au moins un abonnement actif (non expiré ET avec crédits restants)"""
    return UserAbonnement.objects.filter(
        utilisateur=user,
        exercice_restants__gt=0,
        statut='actif',
        date_fin__gt=timezone.now()
    ).exists()

# gestion credit depuis flutter
def debiter_credit_abonnement(user):
    """
    Débite un crédit de l'abonnement actif de l'utilisateur
    """
    abonnement_actif = UserAbonnement.objects.filter(
        utilisateur=user,
        exercice_restants__gt=0,
        statut='actif',
        date_fin__gt=timezone.now()
    ).first()

    if abonnement_actif:
        abonnement_actif.exercice_restants -= 1
        abonnement_actif.save()
        return True
    return False