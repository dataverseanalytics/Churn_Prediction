// API Base URL
const API_BASE = 'http://localhost:8000';

// State
let attributes = [];
let weights = {};
let enabledAttributes = new Set();
let riskChart = null;
let mlAvailable = false;
let useMlPrediction = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    await loadAttributes();
    setupEventListeners();
});

// Load available attributes from API
async function loadAttributes() {
    try {
        const response = await fetch(`${API_BASE}/api/attributes`);
        const data = await response.json();

        if (data.success) {
            attributes = data.attributes;
            weights = { ...data.default_weights };

            // Enable all attributes by default
            attributes.forEach(attr => enabledAttributes.add(attr));

            renderAttributeCheckboxes();
            renderWeightSliders();
        }
    } catch (error) {
        console.error('Error loading attributes:', error);
        showError('Failed to load attributes');
    }
}

// Render attribute checkboxes
function renderAttributeCheckboxes() {
    const container = document.getElementById('attributesGrid');
    container.innerHTML = '';

    attributes.forEach(attr => {
        const div = document.createElement('div');
        div.className = 'attribute-checkbox';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `attr-${attr}`;
        checkbox.checked = true;
        checkbox.addEventListener('change', (e) => handleAttributeToggle(attr, e.target.checked));

        const label = document.createElement('label');
        label.htmlFor = `attr-${attr}`;
        label.textContent = formatAttributeName(attr);

        div.appendChild(checkbox);
        div.appendChild(label);
        container.appendChild(div);
    });
}

// Render weight sliders
function renderWeightSliders() {
    const container = document.getElementById('weightsContainer');
    container.innerHTML = '';

    attributes.forEach(attr => {
        const div = document.createElement('div');
        div.className = 'weight-item enabled';
        div.id = `weight-${attr}`;

        const labelDiv = document.createElement('div');
        labelDiv.className = 'weight-label';

        const nameSpan = document.createElement('span');
        nameSpan.textContent = formatAttributeName(attr);

        const valueSpan = document.createElement('span');
        valueSpan.className = 'weight-value';
        valueSpan.id = `value-${attr}`;
        valueSpan.textContent = `${weights[attr]}%`;

        labelDiv.appendChild(nameSpan);
        labelDiv.appendChild(valueSpan);

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = '50';
        slider.step = '0.5';
        slider.value = weights[attr];
        slider.id = `slider-${attr}`;
        slider.addEventListener('input', (e) => handleWeightChange(attr, parseFloat(e.target.value)));

        div.appendChild(labelDiv);
        div.appendChild(slider);
        container.appendChild(div);
    });

    updateTotalWeight();
}

// Handle attribute toggle
function handleAttributeToggle(attr, enabled) {
    if (enabled) {
        enabledAttributes.add(attr);
        document.getElementById(`weight-${attr}`).classList.add('enabled');
        document.getElementById(`slider-${attr}`).disabled = false;
    } else {
        enabledAttributes.delete(attr);
        document.getElementById(`weight-${attr}`).classList.remove('enabled');
        document.getElementById(`slider-${attr}`).disabled = true;
    }
}

// Handle weight change
function handleWeightChange(attr, value) {
    weights[attr] = value;
    document.getElementById(`value-${attr}`).textContent = `${value}%`;
    updateTotalWeight();
}

// Update total weight display
function updateTotalWeight() {
    const total = Object.values(weights).reduce((sum, val) => sum + val, 0);
    const totalElement = document.getElementById('totalWeight');
    totalElement.textContent = `${total.toFixed(1)}%`;

    // Color code based on validity
    if (Math.abs(total - 100) < 0.1) {
        totalElement.style.color = '#10b981'; // Green
    } else {
        totalElement.style.color = '#ef4444'; // Red
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('calculateBtn').addEventListener('click', calculateRisk);
    document.getElementById('mlModeToggle').addEventListener('change', handleModeToggle);

    // Check ML status on load
    checkMlStatus();
}

// Check if ML model is available
async function checkMlStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/ml/status`);
        const data = await response.json();
        mlAvailable = data.ml_available;

        if (!mlAvailable) {
            document.getElementById('mlModeToggle').disabled = true;
            document.querySelector('.prediction-mode-section').title = "ML model not trained available. Run model_trainer.py";
        }
    } catch (error) {
        console.error('Error checking ML status:', error);
        mlAvailable = false;
    }
}

// Handle mode toggle
function handleModeToggle(e) {
    useMlPrediction = e.target.checked;

    const attributesSection = document.getElementById('attributesSection');
    const weightsSection = document.getElementById('weightsSection');
    const tableHeader = document.querySelector('#membersTable thead tr');

    if (useMlPrediction) {
        // ML Mode: Disable manual configuration
        attributesSection.classList.add('disabled-section');
        weightsSection.classList.add('disabled-section');

        // Update table headers for ML
        if (tableHeader) {
            tableHeader.innerHTML = `
                <th>Member ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Invoice Balance</th>
                <th>Churn Probability</th>
                <th>Risk Category</th>
            `;
        }
    } else {
        // Rule-Based Mode: Enable configuration
        attributesSection.classList.remove('disabled-section');
        weightsSection.classList.remove('disabled-section');

        // Restore table headers
        if (tableHeader) {
            tableHeader.innerHTML = `
                <th>Member ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Invoice Balance</th>
                <th>Risk Score</th>
                <th>Risk Category</th>
            `;
        }
    }
}

// Calculate risk scores
async function calculateRisk() {
    // If Rule-Based Mode, validate weights
    if (!useMlPrediction) {
        const totalWeight = Object.values(weights).reduce((sum, val) => sum + val, 0);

        if (Math.abs(totalWeight - 100) > 0.1) {
            alert('Total weight must equal 100%');
            return;
        }

        if (enabledAttributes.size === 0) {
            alert('Please select at least one attribute');
            return;
        }
    }

    try {
        let endpoint = useMlPrediction ? '/api/ml/predict-churn' : '/api/calculate-risk';
        let payload = useMlPrediction ? {} : {
            weights: weights,
            enabled_attributes: Array.from(enabledAttributes)
        };

        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Failed to calculate risk');
        }
    } catch (error) {
        console.error('Error calculating risk:', error);
        showError('Failed to calculate risk scores');
    }
}

// Display results
function displayResults(data) {
    // Update statistics
    document.getElementById('totalMembers').textContent = data.statistics.total;
    document.getElementById('lowRisk').textContent = data.statistics.low_risk;
    document.getElementById('mediumRisk').textContent = data.statistics.medium_risk;
    document.getElementById('highRisk').textContent = data.statistics.high_risk;

    // Update table
    updateTable(data.results);

    // Update chart
    updateChart(data.statistics);
}

// Update members table
function updateTable(results) {
    const tbody = document.getElementById('membersTableBody');
    tbody.innerHTML = '';

    if (results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="loading-message">No members found requiring risk assessment</td></tr>';
        return;
    }

    results.forEach(member => {
        const row = document.createElement('tr');
        row.className = 'fade-in';

        row.innerHTML = `
            <td>${member.member_id}</td>
            <td>${member.name}</td>
            <td>${member.email}</td>
            <td>$${member.invoice_balance.toFixed(2)}</td>
            <td>${useMlPrediction ? member.churn_probability : member.risk_score}%</td>
            <td><span class="risk-badge risk-${member.risk_category}">${member.risk_category}</span></td>
        `;

        tbody.appendChild(row);
    });
}

// Update risk distribution chart
function updateChart(statistics) {
    const ctx = document.getElementById('riskChart').getContext('2d');

    if (riskChart) {
        riskChart.destroy();
    }

    riskChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [statistics.low_risk, statistics.medium_risk, statistics.high_risk],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    '#10b981',
                    '#f59e0b',
                    '#ef4444'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#cbd5e1',
                        font: {
                            size: 14
                        },
                        padding: 20
                    }
                },
                title: {
                    display: true,
                    text: 'Risk Distribution',
                    color: '#f1f5f9',
                    font: {
                        size: 18,
                        weight: 'bold'
                    },
                    padding: 20
                }
            }
        }
    });
}

// Utility functions
function formatAttributeName(attr) {
    return attr.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function showError(message) {
    alert(`Error: ${message}`);
}
