document.addEventListener('DOMContentLoaded', function() {
    const paysSelect = document.getElementById('pays-select');
    const ssSelect = document.getElementById('ss-select');
    const classeSelect = document.getElementById('classe-select');
    const matiereSelect = document.getElementById('matiere-select');
    const formMatiereWidget = document.getElementById('id_matiere'); // hidden
    const leconForm = document.getElementById('lecon-form');

    paysSelect.addEventListener('change', function() {
        let paysId = paysSelect.value;
        resetSelect(ssSelect, 'Chargement...', true);
        resetSelect(classeSelect, 'Sélectionnez d\'abord un sous-système', true);
        resetSelect(matiereSelect, 'Sélectionnez d\'abord une classe', true);
        formMatiereWidget.value = '';
        if (!paysId) {
            resetSelect(ssSelect, 'Sélectionne d\'abord un pays', true);
            return;
        }
        fetch(window.APP_CONFIG.apiSousSystemes + paysId + '/')
            .then(res => res.json())
            .then(data => {
                populateSelect(ssSelect, data, 'Sélectionner un sous-système', 'nom');
                ssSelect.disabled = false;
            });
    });

    ssSelect.addEventListener('change', function() {
        let ssId = ssSelect.value;
        resetSelect(classeSelect, 'Chargement...', true);
        resetSelect(matiereSelect, 'Sélectionnez d\'abord une classe', true);
        formMatiereWidget.value = '';
        if (!ssId) {
            resetSelect(classeSelect, 'Sélectionne d\'abord un sous-système', true);
            return;
        }
        fetch(window.APP_CONFIG.apiClasses + ssId + '/')
            .then(res => res.json())
            .then(data => {
                populateSelect(classeSelect, data, 'Sélectionner une classe', 'nom', 'code');
                classeSelect.disabled = false;
            });
    });

    classeSelect.addEventListener('change', function() {
        let classeId = classeSelect.value;
        resetSelect(matiereSelect, 'Chargement...', true);
        formMatiereWidget.value = '';
        if (!classeId) {
            resetSelect(matiereSelect, 'Sélectionne d\'abord une classe', true);
            return;
        }
        fetch(window.APP_CONFIG.apiMatieres + classeId + '/')
            .then(res => res.json())
            .then(data => {
                populateSelect(matiereSelect, data, 'Sélectionner une matière', 'nom', 'code');
                matiereSelect.disabled = false;
            });
    });

    matiereSelect.addEventListener('change', function() {
        if (formMatiereWidget && matiereSelect.value) {
            formMatiereWidget.value = matiereSelect.value;
        }
    });

    leconForm.addEventListener('submit', function() {
        if (formMatiereWidget && matiereSelect.value) {
            formMatiereWidget.value = matiereSelect.value;
        }
    });

    function resetSelect(selectElement, placeholder, isDisabled) {
        selectElement.innerHTML = `<option>${placeholder}</option>`;
        selectElement.disabled = isDisabled;
    }

    function populateSelect(selectElement, data, defaultOption, ...properties) {
        selectElement.innerHTML = `<option value="">${defaultOption}</option>`;
        data.forEach(item => {
            let option = document.createElement('option');
            option.value = item.id;
            option.textContent = properties.map(prop => item[prop]).join(" - ");
            selectElement.appendChild(option);
        });
    }
});