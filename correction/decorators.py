from functools import wraps
from django.http import HttpResponseForbidden

def role_required(*allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            # user doit être connecté
            if not request.user.is_authenticated:
                from django.shortcuts import redirect
                from django.urls import reverse
                return redirect(reverse('correction:login') + '?next=' + request.path)
            if hasattr(request.user, 'role') and request.user.role in allowed_roles:
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden("Vous n'avez pas le droit d'accéder à cette page.")
        return _wrapped_view
    return decorator


def only_admin(view_func):
    from functools import wraps
    from django.http import HttpResponseForbidden
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not (request.user.is_authenticated and getattr(request.user, "role", None) == "admin"):
            return HttpResponseForbidden("⛔ Accès interdit : réservé aux admins")
        return view_func(request, *args, **kwargs)
    return _wrapped