<script>
document.getElementById("submitButton").addEventListener("click", function() {
    let selectedValue = document.getElementById("reportDropdown").value;
    
    fetch('/handle_dropdown_value', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ value: selectedValue })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
    });
});
</script>
