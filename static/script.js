const form = document.getElementById("mind-form");
const thought = document.getElementById("thought");
const output = document.getElementById("output");
const outputSection = document.getElementById("output-section");
const button = form.querySelector("button");

function renderResult(result) {
    output.innerHTML = `
        <div class="result-content">
            <h2>Analysis Results</h2>
            <div class="result-grid">
                <div class="result-card emotion-card">
                    <div class="result-label">Emotion</div>
                    <div class="result-value">${capitalizeFirstLetter(result.emotion)}</div>
                </div>
                <div class="result-card sentiment-card">
                    <div class="result-label">Sentiment</div>
                    <div class="result-value">${capitalizeFirstLetter(result.sentiment)}</div>
                </div>
                <div class="result-card confidence-card">
                    <div class="result-label">Confidence</div>
                    <div class="result-value">${Math.round(result.confidence * 100)}% ${result.confidence < 0.5 ? '<span class="tentative">(Tentative)</span>' : ''}</div>
                </div>
            </div>
            <div class="mind-state-box">
                <div class="mind-state-label">Mind State</div>
                <p class="mind-state-text">${result.mind_state}</p>
            </div>
            ${result.top_emotions && result.top_emotions.length ? `
                <div class="top-emotions">
                    <h3>Top Emotions</h3>
                    <ul>
                        ${result.top_emotions.map(e => `<li>${capitalizeFirstLetter(e.emotion)} — ${Math.round(e.probability * 100)}%</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        </div>
    `;
    outputSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderError() {
    output.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">⚠️</div>
            <h3>Unable to Analyze</h3>
            <p>Something went wrong. Please try again.</p>
        </div>
    `;
}

function renderEmpty() {
    output.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">✨</div>
            <h3>Start Analyzing</h3>
            <p>Enter your thoughts to discover your mental state</p>
        </div>
    `;
}

function capitalizeFirstLetter(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function setLoading(isLoading) {
    button.disabled = isLoading;
    if (isLoading) {
        button.innerHTML = '<span class="loading"></span> Analyzing...';
    } else {
        button.textContent = "Analyze Mind State";
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = thought.value.trim();
    if (!text) {
        renderEmpty();
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            renderError();
            setLoading(false);
            return;
        }

        const result = await response.json();
        renderResult(result);
    } catch (error) {
        console.error("Error:", error);
        renderError();
    } finally {
        setLoading(false);
    }
});

// Initialize year in footer
const year = document.getElementById("year");
if (year) year.textContent = new Date().getFullYear();

// Initialize with empty state
renderEmpty();
