/**
 * ==============================================================================
 * AI-Driven Coating Formulation & Property Optimization
 * Frontend Controller: static/script.js
 * Description: Handles client validation, REST API calls, dynamic chart updates,
 *              live formulation balancing, prediction history, and CSV export.
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
    // --------------------------------------------------------------------------
    // DOM Element References
    // --------------------------------------------------------------------------
    const form = document.getElementById("predictionForm");
    const predictBtn = document.getElementById("predictBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnText = document.getElementById("btnText");
    const resetBtn = document.getElementById("resetBtn");
    const downloadBtn = document.getElementById("downloadBtn");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const formAlert = document.getElementById("formAlert");
    const formAlertMsg = document.getElementById("formAlertMsg");

    // Inputs
    const resinInput = document.getElementById("resinContent");
    const pigmentInput = document.getElementById("pigmentContent");
    const solventInput = document.getElementById("solventContent");
    const additiveInput = document.getElementById("additiveContent");
    const curingAgentInput = document.getElementById("curingAgentContent");
    const tempInput = document.getElementById("curingTemp");
    const timeInput = document.getElementById("curingTime");
    const enforceStrict100 = document.getElementById("enforceStrict100");

    // Live Composition Indicator
    const compSumDisplay = document.getElementById("compSumDisplay");
    const compStatusBadge = document.getElementById("compStatusBadge");
    const compProgress = document.getElementById("compProgress");

    // Results Dashboard
    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultsContent = document.getElementById("resultsContent");
    const predictionStatusBadge = document.getElementById("predictionStatusBadge");

    // Metric Values
    const valHardness = document.getElementById("valHardness");
    const valAdhesion = document.getElementById("valAdhesion");
    const valFlexibility = document.getElementById("valFlexibility");
    const valGloss = document.getElementById("valGloss");
    const valDryingTime = document.getElementById("valDryingTime");

    // Tables & Chart
    const summaryTableBody = document.getElementById("summaryTableBody");
    const historyTableBody = document.getElementById("historyTableBody");
    const chartCanvas = document.getElementById("predictionChart");

    // Header & Info Elements
    const backendStatusBadge = document.getElementById("backendStatusBadge");
    const backendStatusText = document.getElementById("backendStatusText");
    const demoBanner = document.getElementById("demoBanner");
    const modelArchText = document.getElementById("modelArchText");
    const modelTaskText = document.getElementById("modelTaskText");
    const modelR2Text = document.getElementById("modelR2Text");
    const modelMaeText = document.getElementById("modelMaeText");

    let currentPredictionData = null;
    let predictionChart = null;

    // --------------------------------------------------------------------------
    // 1. Health Check & Model Info
    // --------------------------------------------------------------------------
    async function checkBackendHealth() {
        try {
            const res = await fetch("/health");
            if (!res.ok) throw new Error("Health check failed");
            const data = await res.json();

            if (data.model_loaded) {
                backendStatusBadge.className = "status-badge status-ready";
                backendStatusText.textContent = "AI Model Ready ✓";

                if (data.is_demo) {
                    demoBanner.style.display = "flex";
                } else {
                    demoBanner.style.display = "none";
                }
            } else {
                backendStatusBadge.className = "status-badge status-error";
                backendStatusText.textContent = "Model File Missing";
            }
        } catch (err) {
            console.error("Health check error:", err);
            backendStatusBadge.className = "status-badge status-error";
            backendStatusText.textContent = "Backend Offline";
        }
    }

    async function loadModelInfo() {
        try {
            const res = await fetch("/api/info");
            if (!res.ok) return;
            const data = await res.json();

            if (data.model_name) modelArchText.textContent = data.model_name;
            if (data.task) modelTaskText.textContent = data.task;
            if (data.metrics && data.metrics.Mean_R2 !== undefined) {
                modelR2Text.textContent = data.metrics.Mean_R2.toFixed(4);
            }
            if (data.metrics && data.metrics.Mean_MAE !== undefined) {
                modelMaeText.textContent = data.metrics.Mean_MAE.toFixed(3);
            }
        } catch (err) {
            console.warn("Could not fetch model info:", err);
        }
    }

    // --------------------------------------------------------------------------
    // 2. Dynamic Formulation Balance Indicator
    // --------------------------------------------------------------------------
    function updateCompositionBar() {
        const r = parseFloat(resinInput.value) || 0;
        const p = parseFloat(pigmentInput.value) || 0;
        const s = parseFloat(solventInput.value) || 0;
        const a = parseFloat(additiveInput.value) || 0;
        const c = parseFloat(curingAgentInput.value) || 0;

        const total = r + p + s + a + c;
        compSumDisplay.textContent = total.toFixed(1) + "%";

        const clampedWidth = Math.min(Math.max((total / 100) * 100, 0), 100);
        compProgress.style.width = clampedWidth + "%";

        if (Math.abs(total - 100.0) < 0.5) {
            compStatusBadge.className = "comp-badge valid";
            compStatusBadge.textContent = "✓ Valid (100%)";
            compProgress.style.backgroundColor = "var(--success-green)";
        } else {
            compStatusBadge.className = "comp-badge warning";
            compStatusBadge.textContent = "⚠ Target 100%";
            compProgress.style.backgroundColor = "var(--warning-amber)";
        }
    }

    [resinInput, pigmentInput, solventInput, additiveInput, curingAgentInput].forEach(inp => {
        inp.addEventListener("input", updateCompositionBar);
    });

    // --------------------------------------------------------------------------
    // 3. Client-Side Input Validation
    // --------------------------------------------------------------------------
    function validateFormInputs() {
        hideAlert();
        let isValid = true;
        const fields = [
            { el: resinInput, name: "Resin Content", min: 0, max: 100 },
            { el: pigmentInput, name: "Pigment Content", min: 0, max: 100 },
            { el: solventInput, name: "Solvent Content", min: 0, max: 100 },
            { el: additiveInput, name: "Additive Content", min: 0, max: 50 },
            { el: curingAgentInput, name: "Curing Agent Content", min: 0, max: 50 },
            { el: tempInput, name: "Curing Temperature", min: 0, max: 300 },
            { el: timeInput, name: "Curing Time", min: 0, max: 600 }
        ];

        fields.forEach(f => {
            f.el.classList.remove("input-invalid");
            const val = parseFloat(f.el.value);

            if (isNaN(val) || f.el.value.trim() === "") {
                f.el.classList.add("input-invalid");
                showAlert(`Please enter a valid numeric value for ${f.name}.`);
                isValid = false;
            } else if (val < f.min) {
                f.el.classList.add("input-invalid");
                showAlert(`${f.name} cannot be negative (min: ${f.min}).`);
                isValid = false;
            } else if (val > f.max) {
                f.el.classList.add("input-invalid");
                showAlert(`${f.name} exceeds operational range (max: ${f.max}).`);
                isValid = false;
            }
        });

        if (!isValid) return false;

        // Check strict 100% total if configured
        if (enforceStrict100.checked) {
            const r = parseFloat(resinInput.value);
            const p = parseFloat(pigmentInput.value);
            const s = parseFloat(solventInput.value);
            const a = parseFloat(additiveInput.value);
            const c = parseFloat(curingAgentInput.value);
            const sum = r + p + s + a + c;

            if (Math.abs(sum - 100.0) > 1.0) {
                showAlert(`Strict formulation constraint enabled: Components sum to ${sum.toFixed(1)}% (Must equal 100.0 ± 1.0%).`);
                return false;
            }
        }

        return true;
    }

    function showAlert(msg) {
        formAlertMsg.textContent = msg;
        formAlert.style.display = "flex";
    }

    function hideAlert() {
        formAlert.style.display = "none";
        formAlertMsg.textContent = "";
    }

    // --------------------------------------------------------------------------
    // 4. Submit & Prediction Flow
    // --------------------------------------------------------------------------
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        if (!validateFormInputs()) return;

        // Prepare JSON payload with exact feature schema
        const payload = {
            Resin_Content: parseFloat(resinInput.value),
            Pigment_Content: parseFloat(pigmentInput.value),
            Solvent_Content: parseFloat(solventInput.value),
            Additive_Content: parseFloat(additiveInput.value),
            Curing_Agent_Content: parseFloat(curingAgentInput.value),
            Curing_Temperature: parseFloat(tempInput.value),
            Curing_Time: parseFloat(timeInput.value)
        };

        // UI Loading State
        predictBtn.disabled = true;
        btnSpinner.style.display = "inline-block";
        btnText.textContent = "Running AI prediction...";
        emptyState.style.display = "none";
        resultsContent.style.display = "none";
        loadingState.style.display = "block";
        predictionStatusBadge.className = "badge badge-neutral";
        predictionStatusBadge.textContent = "Computing...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Unable to generate prediction from the server.");
            }

            // Success: Display Results
            currentPredictionData = data;
            displayResults(data);

            // Store in LocalStorage History
            savePredictionToHistory(payload, data.predictions);

            predictionStatusBadge.className = "badge badge-info";
            predictionStatusBadge.textContent = "Prediction Complete ✓";
            btnText.textContent = "Prediction completed.";

            setTimeout(() => {
                btnText.textContent = "PREDICT COATING PROPERTIES";
                predictBtn.disabled = false;
                btnSpinner.style.display = "none";
            }, 1200);

        } catch (error) {
            console.error("Prediction Error:", error);
            showAlert("Unable to generate prediction. Please check that the ML model is available and the Flask server is running.");
            loadingState.style.display = "none";
            emptyState.style.display = "block";
            predictionStatusBadge.className = "badge badge-neutral";
            predictionStatusBadge.textContent = "Error";
            btnText.textContent = "PREDICT COATING PROPERTIES";
            predictBtn.disabled = false;
            btnSpinner.style.display = "none";
        }
    });

    // --------------------------------------------------------------------------
    // 5. Render Results & Interactive Chart
    // --------------------------------------------------------------------------
    function displayResults(data) {
        loadingState.style.display = "none";
        resultsContent.style.display = "block";

        const p = data.predictions;

        // Metric Cards
        valHardness.textContent = p.Hardness !== undefined ? p.Hardness.toFixed(1) : "--";
        valAdhesion.textContent = p.Adhesion !== undefined ? p.Adhesion.toFixed(1) : "--";
        valFlexibility.textContent = p.Flexibility !== undefined ? p.Flexibility.toFixed(1) : "--";
        valGloss.textContent = p.Gloss !== undefined ? p.Gloss.toFixed(1) : "--";
        valDryingTime.textContent = p.Drying_Time !== undefined ? p.Drying_Time.toFixed(1) : "--";

        // Render Summary Table
        renderSummaryTable(data.input_summary, data.units);

        // Render Chart.js Bar Chart
        renderChart(p, data.units);
    }

    function renderSummaryTable(inputs, units) {
        summaryTableBody.innerHTML = "";
        const paramRows = [
            { key: "Resin Content", val: inputs.Resin_Content, unit: "wt%" },
            { key: "Pigment Content", val: inputs.Pigment_Content, unit: "wt%" },
            { key: "Solvent Content", val: inputs.Solvent_Content, unit: "wt%" },
            { key: "Additive Content", val: inputs.Additive_Content, unit: "wt%" },
            { key: "Curing Agent Content", val: inputs.Curing_Agent_Content, unit: "wt%" },
            { key: "Curing Temperature", val: inputs.Curing_Temperature, unit: "°C" },
            { key: "Curing Time", val: inputs.Curing_Time, unit: "minutes" }
        ];

        paramRows.forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${row.key}</strong></td>
                <td>${row.val.toFixed(1)}</td>
                <td><span class="unit">${row.unit}</span></td>
            `;
            summaryTableBody.appendChild(tr);
        });
    }

    function renderChart(predictions, units) {
        const labels = ["Hardness", "Adhesion", "Flexibility", "Gloss", "Drying Time"];
        const values = [
            predictions.Hardness,
            predictions.Adhesion,
            predictions.Flexibility,
            predictions.Gloss,
            predictions.Drying_Time
        ];

        const backgroundColors = [
            "rgba(37, 99, 235, 0.8)",   // Hardness Blue
            "rgba(16, 185, 129, 0.8)",  // Adhesion Green
            "rgba(245, 158, 11, 0.8)",  // Flexibility Amber
            "rgba(147, 51, 234, 0.8)",  // Gloss Purple
            "rgba(239, 68, 68, 0.8)"    // Drying Red
        ];

        const borderColors = [
            "#1d4ed8",
            "#059669",
            "#d97706",
            "#7e22ce",
            "#dc2626"
        ];

        if (predictionChart) {
            predictionChart.destroy();
        }

        const ctx = chartCanvas.getContext("2d");
        predictionChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Predicted Property Magnitude",
                    data: values,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const propKey = context.label.replace(" ", "_");
                                const unit = units[propKey] || "";
                                return `Predicted: ${context.parsed.y} ${unit}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: "#e2e8f0" },
                        ticks: { font: { family: "'Inter', sans-serif", size: 11 } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: "'Inter', sans-serif", size: 11, weight: "bold" } }
                    }
                }
            }
        });
    }

    // --------------------------------------------------------------------------
    // 6. Reset Functionality
    // --------------------------------------------------------------------------
    resetBtn.addEventListener("click", () => {
        // Restore Default Inputs
        resinInput.value = "48.0";
        pigmentInput.value = "24.0";
        solventInput.value = "20.0";
        additiveInput.value = "2.5";
        curingAgentInput.value = "5.5";
        tempInput.value = "105.0";
        timeInput.value = "45.0";
        enforceStrict100.checked = false;

        // Reset UI State
        updateCompositionBar();
        hideAlert();
        resultsContent.style.display = "none";
        loadingState.style.display = "none";
        emptyState.style.display = "block";
        predictionStatusBadge.className = "badge badge-neutral";
        predictionStatusBadge.textContent = "Awaiting Input";

        if (predictionChart) {
            predictionChart.destroy();
            predictionChart = null;
        }
        currentPredictionData = null;
    });

    // --------------------------------------------------------------------------
    // 7. Prediction History via LocalStorage
    // --------------------------------------------------------------------------
    function savePredictionToHistory(inputs, predictions) {
        const history = JSON.parse(localStorage.getItem("coating_predictions_history") || "[]");
        const entry = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            formulation: `${inputs.Resin_Content}/${inputs.Pigment_Content}/${inputs.Solvent_Content}/${inputs.Additive_Content}/${inputs.Curing_Agent_Content}`,
            bake: `${inputs.Curing_Temperature}°C / ${inputs.Curing_Time}m`,
            hardness: predictions.Hardness,
            adhesion: predictions.Adhesion,
            flexibility: predictions.Flexibility,
            gloss: predictions.Gloss,
            drying: predictions.Drying_Time
        };

        history.unshift(entry); // Prepend
        // Keep max 10 entries
        if (history.length > 10) history.pop();
        localStorage.setItem("coating_predictions_history", JSON.stringify(history));
        renderHistoryTable();
    }

    function renderHistoryTable() {
        const history = JSON.parse(localStorage.getItem("coating_predictions_history") || "[]");
        if (history.length === 0) {
            historyTableBody.innerHTML = `<tr><td colspan="8" class="text-center empty-history">No predictions recorded in this browser session.</td></tr>`;
            return;
        }

        historyTableBody.innerHTML = "";
        history.forEach(item => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><small>${item.timestamp}</small></td>
                <td><code>${item.formulation}</code></td>
                <td><small>${item.bake}</small></td>
                <td><strong>${item.hardness.toFixed(1)}</strong></td>
                <td><strong>${item.adhesion.toFixed(1)}</strong></td>
                <td><strong>${item.flexibility.toFixed(1)}</strong></td>
                <td><strong>${item.gloss.toFixed(1)}</strong></td>
                <td><strong>${item.drying.toFixed(1)}m</strong></td>
            `;
            historyTableBody.appendChild(tr);
        });
    }

    clearHistoryBtn.addEventListener("click", () => {
        localStorage.removeItem("coating_predictions_history");
        renderHistoryTable();
    });

    // --------------------------------------------------------------------------
    // 8. Download Results as CSV
    // --------------------------------------------------------------------------
    downloadBtn.addEventListener("click", () => {
        if (!currentPredictionData) return;

        const inputs = currentPredictionData.input_summary;
        const preds = currentPredictionData.predictions;
        const units = currentPredictionData.units;

        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "Category,Parameter,Value,Unit\n";

        // Formulation Inputs
        csvContent += `Input,Resin_Content,${inputs.Resin_Content},wt%\n`;
        csvContent += `Input,Pigment_Content,${inputs.Pigment_Content},wt%\n`;
        csvContent += `Input,Solvent_Content,${inputs.Solvent_Content},wt%\n`;
        csvContent += `Input,Additive_Content,${inputs.Additive_Content},wt%\n`;
        csvContent += `Input,Curing_Agent_Content,${inputs.Curing_Agent_Content},wt%\n`;
        csvContent += `Input,Curing_Temperature,${inputs.Curing_Temperature},deg_C\n`;
        csvContent += `Input,Curing_Time,${inputs.Curing_Time},minutes\n`;

        // Predicted Outputs
        csvContent += `AI_Predicted,Hardness,${preds.Hardness},${units.Hardness || ""}\n`;
        csvContent += `AI_Predicted,Adhesion,${preds.Adhesion},${units.Adhesion || ""}\n`;
        csvContent += `AI_Predicted,Flexibility,${preds.Flexibility},${units.Flexibility || ""}\n`;
        csvContent += `AI_Predicted,Gloss,${preds.Gloss},${units.Gloss || ""}\n`;
        csvContent += `AI_Predicted,Drying_Time,${preds.Drying_Time},${units.Drying_Time || ""}\n`;

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `coating_prediction_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // --------------------------------------------------------------------------
    // Initialization
    // --------------------------------------------------------------------------
    updateCompositionBar();
    renderHistoryTable();
    checkBackendHealth();
    loadModelInfo();
});
