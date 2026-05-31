// Utility functions for Trading System GUI

function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;
    
    const alertId = 'alert-' + Date.now();
    const alertHTML = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    alertContainer.insertAdjacentHTML('beforeend', alertHTML);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.remove();
        }
    }, 5000);
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

function getDateString(date) {
    return date.toISOString().split('T')[0];
}

function setDefaultDates() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 1);
    
    const startInput = document.getElementById('startDate');
    const endInput = document.getElementById('endDate');
    
    if (startInput) startInput.value = getDateString(startDate);
    if (endInput) endInput.value = getDateString(endDate);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setDefaultDates();
});
