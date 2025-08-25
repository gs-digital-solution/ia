document.addEventListener('DOMContentLoaded', function() {
    const paysSelect = document.getElementById('pays-select');
    const departementSelect = document.getElementById('departement-select');
    const typeSelect = document.getElementById('type-select');
    const ssSelect = document.getElementById('ss-select');
    const classeSelect = document.getElementById('classe-select');
    const matiereSelect = document.getElementById('matiere-select');
    const leconsUl = document.getElementById('liste-lecons');
    const formMatiereWidget = document.getElementById('id_matiere');
    const form = document.getElementById('exercice-corrige-form');

    paysSelect.addEventListener('change', () => {
        resetSelect(departementSelect, 'Département', true);
        resetSelect(typeSelect, 'Type d\'exercice', true);
        resetSelect(ssSelect, 'Sous-système', true);
        resetSelect(classeSelect, 'Classe', true);
        resetSelect(matiereSelect, 'Matière', true);
        departementSelect.disabled = true; typeSelect.disabled = true; ssSelect.disabled = true;
        classeSelect.disabled = true; matiereSelect.disabled = true;
        clearLecons();
        formMatiereWidget.value = '';
        if (!paysSelect.value) return;
        fetch(window.APP_CONFIG.apiDepartements + paysSelect.value + '/')
            .then(res => res.json())
            .then(data => {
                populateSelect(departementSelect, data, 'Département', 'nom');
                departementSelect.disabled = false;
            });
        fetch(window.APP_CONFIG.apiSousSystemes + paysSelect.value + '/')
            .then(res => res.json())
            .then(data => {
                populateSelect(ssSelect, data, 'Sous-système', 'nom');
                ssSelect.disabled = false;
            });
    });

    departementSelect.addEventListener('change', () => {
        resetSelect(typeSelect, 'Type d\'exercice', true);
        typeSelect.disabled = true;
        if (!departementSelect.value) return;
        fetch(window.APP_CONFIG.apiTypesExercices + departementSelect.value + '/')
            .then(res => res.json())
            .then(data => {
                populateSelect(typeSelect, data, 'Type d\'exercice', 'nom');
                typeSelect.disabled = false;
            });
    });

    ssSelect.addEventListener('change', () => {
        resetSelect(classeSelect, 'Classe', true);
        resetSelect(matiereSelect, 'Matière', true);
        classeSelect.disabled = true; matiereSelect.disabled = true;
        clearLecons(); formMatiereWidget.value = '';
        if (!ssSelect.value) return;
        fetch(window.APP_CONFIG.apiClasses + ssSelect.value + '/')
            .then(res => res.json())
            .then(data => {
                populateSelect(classeSelect, data, 'Classe', 'nom', 'code');
                classeSelect.disabled = false;
            });
    });

    classeSelect.addEventListener('change', () => {
        resetSelect(matiereSelect, 'Matière', true);
        matiereSelect.disabled = true; clearLecons(); formMatiereWidget.value = '';
        if (!classeSelect.value) return;
        fetch(window.APP_CONFIG.apiMatieres + classeSelect.value + '/')
            .then(res => res.json())
            .then(data => {
                populateSelect(matiereSelect, data, 'Matière', 'nom', 'code');
                matiereSelect.disabled = false;
            });
    });

    matiereSelect.addEventListener('change', () => {
        formMatiereWidget.value = matiereSelect.value || '';
        clearLecons();
        if (!matiereSelect.value) return;
        fetch(window.APP_CONFIG.apiLecons + matiereSelect.value + '/')
            .then(res => res.json())
            .then(data => {
                data.forEach(lecon => {
                    let li = document.createElement('li');
                    // one checkbox per leçon, même name!
                    let cb = document.createElement('input');
                    cb.type = "checkbox";
                    cb.value = lecon.id;
                    cb.name = "lecons_associees";
                    li.appendChild(cb);
                    li.appendChild(document.createTextNode(" " + lecon.titre));
                    leconsUl.appendChild(li);
                });
            });
    });

    function clearLecons() {
        leconsUl.innerHTML = "";
    }
    function resetSelect(sel, placeholder, dis) {
        sel.innerHTML = `<option>${placeholder}</option>`;
        sel.disabled = dis;
    }
    function populateSelect(sel, arr, label, ...props) {
        sel.innerHTML = `<option value="">${label}</option>`;
        arr.forEach(item => {
            let opt = document.createElement('option');
            opt.value = item.id;
            opt.textContent = props.map(p => item[p]).join(" - ");
            sel.appendChild(opt);
        });
    }
    // Synchronise la value matière à l’envoi
    form.addEventListener('submit', () => {
        if (formMatiereWidget && matiereSelect.value) formMatiereWidget.value = matiereSelect.value;
    });
});