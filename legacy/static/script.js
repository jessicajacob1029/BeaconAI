document.addEventListener('DOMContentLoaded', function () {

    // Login form validation
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function (e) {
            const username = document.getElementById('loginUsername').value.trim();
            const password = document.getElementById('loginPassword').value.trim();

            if (!username || !password) {
                alert("Please fill in both username and password.");
                e.preventDefault();
            }
        });
    }

    // Register form validation
    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        registerForm.addEventListener('submit', function (e) {
            const username = document.getElementById('registerUsername').value.trim();
            const password = document.getElementById('registerPassword').value.trim();

            if (!username || !password) {
                alert("Please fill in both fields to register.");
                e.preventDefault();
            } else if (password.length < 4) {
                alert("Password must be at least 4 characters long.");
                e.preventDefault();
            }
        });
    }

    // Highlight active navigation button
    const navButtons = document.querySelectorAll('.nav-button');
    navButtons.forEach(button => {
        button.addEventListener('click', function () {
            navButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Optional console message
    console.log("Hospital Management Website is ready!");
});
