document.addEventListener('DOMContentLoaded', function () {
    function updateOptions(url, data, $select, placeholder) {
        fetch(url + '?' + new URLSearchParams(data))
            .then(response => response.json())
            .then(data => {
                $select.innerHTML = '';
                let opt = document.createElement('option');
                opt.value = '';
                opt.text = placeholder;
                $select.appendChild(opt);
                data.options.forEach((item) => {
                    let option = document.createElement('option');
                    option.value = item.id;
                    option.text = item.nom || item.titre;
                    $select.appendChild(option);
                });
                // Pour leçons : active/désactive sélecteur selon dispo
                if ($select.id === "id_lecons") {
                    $select.disabled = !(data.options.length > 0);
                    var event = new Event('ajax-lecons-updated');
                    document.dispatchEvent(event);
                }
            });
    }

    const paysSelect = document.getElementById('id_pays');
    const ssSelect = document.getElementById('id_sous_systeme');
    const departementSelect = document.getElementById('id_departement');
    const classeSelect = document.getElementById('id_classe');
    const matiereSelect = document.getElementById('id_matiere');
    const typeExoSelect = document.getElementById('id_type_exercice');
    const leconsSelect = document.getElementById('id_lecons');

    function disableAllExceptPays() {
        if (ssSelect) ssSelect.disabled = true;
        if (departementSelect) departementSelect.disabled = true;
        if (classeSelect) classeSelect.disabled = true;
        if (matiereSelect) matiereSelect.disabled = true;
        if (typeExoSelect) typeExoSelect.disabled = true;
        if (leconsSelect) leconsSelect.disabled = true;
    }
    disableAllExceptPays();

    if (paysSelect) {
        paysSelect.addEventListener('change', function () {
            updateOptions('/correction/ajax/sous-systemes/', {pays_id: this.value}, ssSelect, 'Sélectionnez un sous-système');
            if (ssSelect) ssSelect.disabled = false;
            updateOptions('/correction/ajax/departements/', {pays_id: this.value}, departementSelect, 'Sélectionnez un département');
            if (departementSelect) departementSelect.disabled = false;
            if (classeSelect) { classeSelect.innerHTML=''; classeSelect.disabled = true; }
            if (matiereSelect) { matiereSelect.innerHTML=''; matiereSelect.disabled = true; }
            if (typeExoSelect) { typeExoSelect.innerHTML=''; typeExoSelect.disabled = true; }
            if (leconsSelect) { leconsSelect.innerHTML=''; leconsSelect.disabled = true; }
        });
    }
    if (ssSelect) {
        ssSelect.addEventListener('change', function () {
            updateOptions('/correction/ajax/classes/', {ss_id: this.value}, classeSelect, 'Sélectionnez une classe');
            if (classeSelect) classeSelect.disabled = false;
            if (matiereSelect) { matiereSelect.innerHTML=''; matiereSelect.disabled = true; }
            if (leconsSelect) { leconsSelect.innerHTML=''; leconsSelect.disabled = true; }
        });
    }
    if (classeSelect) {
        classeSelect.addEventListener('change', function () {
            updateOptions('/correction/ajax/matieres/', {classe_id: this.value}, matiereSelect, 'Sélectionnez une matière');
            if (matiereSelect) matiereSelect.disabled = false;
            if (leconsSelect) { leconsSelect.innerHTML=''; leconsSelect.disabled = true; }
        });
    }
    if (departementSelect) {
        departementSelect.addEventListener('change', function () {
            updateOptions('/correction/ajax/types-exercices/', {departement_id: this.value}, typeExoSelect, 'Sélectionnez un type d\'exercice');
            if (typeExoSelect) typeExoSelect.disabled = false;
        });
    }
    if (matiereSelect) {
        matiereSelect.addEventListener('change', function () {
            updateOptions('/correction/ajax/lecons/', {matiere_id: this.value}, leconsSelect, 'Sélectionnez une leçon');
            if (leconsSelect) leconsSelect.disabled = false;
        });
    }
});