document.addEventListener('DOMContentLoaded', function() {
    //Current year for the footer
    let year = document.getElementById("year");
    year.textContent = new Date().getFullYear();

    let output = document.getElementById("output");
    let thought = document.getElementById("thought").value();

    if (!thought) {
        output.classList.add("hidden");
    } else {
        continue;
    }
});
