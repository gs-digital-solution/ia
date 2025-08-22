document.addEventListener('DOMContentLoaded', function() {
    const paysSelect = document.getElementById('pays-select');
    const ssSelect = document.getElementById('ss-select');
    const API_URL = window.APP_CONFIG.apiUrl;

    // Le CSRF token est utile si tu fais POST, pas pour un GET, mais prêt pour l'évolution
    // const CSRF_TOKEN = window.APP_CONFIG.csrfToken;

    paysSelect.addEventListener('change', function() {
        const paysId = paysSelect.value;
        ssSelect.innerHTML = '<option value="">Chargement en cours...</option>';
        ssSelect.disabled = true;

        if (!paysId) {
            updateSsSelect([], "Sélectionnez d'abord un pays");
            return;
        }

        fetch(API_URL + paysId + "/")
            .then(response => response.json())
            .then(data => {
                updateSsSelect(data, "Aucun sous-système disponible pour ce pays");
            })
            .catch(error => {
                console.error("Erreur API", error);
                updateSsSelect([], "Erreur de chargement", true);
            });
    });

    function updateSsSelect(items, emptyMessage, isError = false) {
        ssSelect.innerHTML = '';

        if (items.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = emptyMessage;
            option.disabled = true;

            if (isError) {
                option.style.color = 'red';
            }

            ssSelect.appendChild(option);
            ssSelect.disabled = true;
        } else {
            items.forEach(ss => {
                const option = document.createElement('option');
                option.value = ss.id;
                option.textContent = ss.nom;
                ssSelect.appendChild(option);
            });
            ssSelect.disabled = false;
        }
    }
});