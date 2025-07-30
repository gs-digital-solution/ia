document.addEventListener('DOMContentLoaded', function() {
    const paysSelect = document.getElementById('pays-select');
    const ssSelect = document.getElementById('ss-select');
    const classeSelect = document.getElementById('classe-select');
    const API_SOUS_SYSTEMES = window.APP_CONFIG.apiSousSystemes;
    const API_CLASSES = window.APP_CONFIG.apiClasses;

    paysSelect.addEventListener('change', () => {
        let paysId = paysSelect.value;
        ssSelect.innerHTML = '<option>Chargement...</option>';
        ssSelect.disabled = true;
        classeSelect.innerHTML = '<option>Sélectionne d\'abord un sous-système</option>';
        classeSelect.disabled = true;
        if (!paysId) {
            ssSelect.innerHTML = '<option>Sélectionne d\'abord un pays</option>';
            return;
        }
        fetch(API_SOUS_SYSTEMES + paysId + '/')
            .then(res => res.json())
            .then(data => {
                ssSelect.innerHTML = '';
                if (data.length === 0) {
                    ssSelect.innerHTML = "<option>Aucun sous-système</option>";
                } else {
                    data.forEach(function(ss) {
                        let option = document.createElement('option');
                        option.value = ss.id;
                        option.textContent = ss.nom;
                        ssSelect.appendChild(option);
                    });
                    ssSelect.disabled = false;
                }
            });
    });

    ssSelect.addEventListener('change', () => {
        let ssId = ssSelect.value;
        classeSelect.innerHTML = '<option>Chargement...</option>';
        classeSelect.disabled = true;
        if (!ssId) {
            classeSelect.innerHTML = '<option>Sélectionne d\'abord un sous-système</option>';
            return;
        }
        fetch(API_CLASSES + ssId + '/')
            .then(res => res.json())
            .then(data => {
                classeSelect.innerHTML = '';
                if (data.length === 0) {
                    classeSelect.innerHTML = "<option>Aucune classe disponible</option>";
                } else {
                    data.forEach(function(cl) {
                        let option = document.createElement('option');
                        option.value = cl.id;
                        option.textContent = cl.nom + " (" + cl.code + ")";
                        classeSelect.appendChild(option);
                    });
                    classeSelect.disabled = false;
                }
            });
    });
});