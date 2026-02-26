document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const resultDiv = document.getElementById('result');

    // Show loading
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `
        <div class="loading" style="display:block;">
            <div class="spinner"></div>
            <p style="font-size:1.2em; color:#667eea;">Analyzing patient data...</p>
        </div>`;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `
                <div style="text-align:center; padding:30px;">
                    <h2 style="color:#e74c3c;">⚠️ Error</h2>
                    <p style="font-size:1.1em; margin-top:10px;">${data.error}</p>
                </div>`;
            return;
        }

        const isHigh = data.risk_class === 'high';

        const featuresHTML = data.top_features.map((f, i) => `
            <div class="feature-item">
                <strong>${i + 1}. ${f.name}</strong>
                <span style="float:right; color:#888;">Value: ${f.value.toFixed(2)}</span>
                <div class="bar-container" style="height:25px; margin-top:8px;">
                    <div class="bar-fill low-risk" style="width:${f.importance.toFixed(1)}%; justify-content:flex-end; padding-right:8px;">
                        ${f.importance.toFixed(1)}%
                    </div>
                </div>
            </div>`).join('');

        const recommendationsHTML = data.recommendations.map(r =>
            `<li>${r}</li>`).join('');

        resultDiv.innerHTML = `
            <div class="result-header ${isHigh ? 'high-risk' : 'low-risk'}">
                <h2>${isHigh ? '⚠️ HIGH RISK DETECTED' : '✅ LOW RISK'}</h2>
                <div class="risk-badge ${data.risk_class}">
                    ${data.prediction}
                </div>
            </div>

            <div class="probability-bars">
                <h3>📊 Prediction Confidence</h3>

                <div class="probability-bar">
                    <label>🔴 High Risk Probability</label>
                    <div class="bar-container">
                        <div class="bar-fill high-risk" style="width:${data.high_risk_probability.toFixed(1)}%">
                            ${data.high_risk_probability.toFixed(1)}%
                        </div>
                    </div>
                </div>

                <div class="probability-bar">
                    <label>🟢 Low Risk Probability</label>
                    <div class="bar-container">
                        <div class="bar-fill low-risk" style="width:${data.low_risk_probability.toFixed(1)}%">
                            ${data.low_risk_probability.toFixed(1)}%
                        </div>
                    </div>
                </div>
            </div>

            <div class="top-features">
                <h3>📈 Top 5 Contributing Features</h3>
                ${featuresHTML}
            </div>

            <div class="recommendations">
                <h3>${isHigh ? '⚠️ Recommendations' : '✅ Health Maintenance Tips'}</h3>
                <ul>${recommendationsHTML}</ul>
            </div>

            <div style="text-align:center; margin-top:30px;">
                <button onclick="resetForm()" class="btn-predict" 
                    style="background: linear-gradient(135deg, #636e72 0%, #2d3436 100%); max-width:300px;">
                    🔄 New Prediction
                </button>
            </div>`;

    } catch (err) {
        resultDiv.innerHTML = `
            <div style="text-align:center; padding:30px;">
                <h2 style="color:#e74c3c;">⚠️ Connection Error</h2>
                <p style="margin-top:10px;">Could not connect to server. Please try again.</p>
            </div>`;
    }

    // Scroll to result
    resultDiv.scrollIntoView({ behavior: 'smooth' });
});

function resetForm() {
    document.getElementById('predictionForm').reset();
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
