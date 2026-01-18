from django.contrib import admin
from .models import SubscriptionType, PromoCode, UserAbonnement, GlobalSubscriptionConfig

admin.site.register(SubscriptionType)
admin.site.register(PromoCode)
admin.site.register(UserAbonnement)
admin.site.register(GlobalSubscriptionConfig)

