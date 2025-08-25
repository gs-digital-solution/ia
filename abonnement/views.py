from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import CodePromoForm
from .services import activer_abonnement_via_promo

@login_required
def activer_abonnement_gratuit(request):
    if request.method == "POST":
        form = CodePromoForm(request.POST)
        if form.is_valid():
            code = form.cleaned_data['code_promo'].strip().upper()
            abo, msg = activer_abonnement_via_promo(request.user, code)
            if abo:
                messages.success(request, msg)
                return redirect('correction:soumettre')  # à adapter selon ta route principale utilisateur
            else:
                messages.error(request, msg)
    else:
        form = CodePromoForm()
    return render(request, "abonnement/activation_via_code.html", {"form": form})