/* Cognitive Mirror — Frontend Application
 * =====================================================================
 * Architecture:
 *   - All DOM built via document.createElement (no innerHTML from data).
 *   - Network errors surfaced to the user via toast notifications.
 *   - Loading states on every user-triggered action.
 *   - Session state server-managed via cookie; rechecked on load.
 *   - Particle background, theme toggling, keyboard shortcuts.
 *   - Premium animated dashboard rendering.
 * ===================================================================== */

/* ================================================================== */
/*  Constants                                                           */
/* ================================================================== */

const API = "/api/v1";
const MAX_CHARS = 1000;

/* ================================================================== */
/*  Theme Management                                                    */
/* ================================================================== */

const THEME_KEY = "cognitive-mirror-theme";

function getSystemTheme() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function getStoredTheme() {
    return localStorage.getItem(THEME_KEY) || "system";
}

function applyTheme(theme) {
    const root = document.documentElement;
    root.classList.remove("theme-light", "theme-dark");
    const resolved = theme === "system" ? getSystemTheme() : theme;
    root.classList.add(`theme-${resolved}`);
    updateThemeToggleIcons(resolved);
}

function cycleTheme() {
    const current = getStoredTheme();
    const next = current === "system" ? "light" : current === "light" ? "dark" : "system";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
    showToast(`Theme: ${next.charAt(0).toUpperCase() + next.slice(1)}`, "success");
}

function updateThemeToggleIcons(resolved) {
    const lightIcon = document.querySelector(".theme-icon-light");
    const darkIcon = document.querySelector(".theme-icon-dark");
    if (!lightIcon || !darkIcon) return;
    if (resolved === "dark") {
        lightIcon.classList.add("hidden");
        darkIcon.classList.remove("hidden");
    } else {
        lightIcon.classList.remove("hidden");
        darkIcon.classList.add("hidden");
    }
}

applyTheme(getStoredTheme());

// Listen for system theme changes
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (getStoredTheme() === "system") applyTheme("system");
});

/* ================================================================== */
/*  Particle Background                                                 */
/* ================================================================== */

(function initParticles() {
    const canvas = document.getElementById("particles-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let particles = [];
    const PARTICLE_COUNT = 50;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createParticles() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                radius: Math.random() * 2 + 1,
                opacity: Math.random() * 0.3 + 0.05,
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const isDark = document.documentElement.classList.contains("theme-dark") ||
            (getStoredTheme() === "system" && getSystemTheme() === "dark");
        const color = isDark ? "255, 255, 255" : "99, 102, 241";

        for (const p of particles) {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${color}, ${p.opacity})`;
            ctx.fill();
        }

        // Draw connections between nearby particles
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(${color}, ${0.04 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(draw);
    }

    resize();
    createParticles();
    draw();

    window.addEventListener("resize", () => {
        resize();
        createParticles();
    });
})();

/* ================================================================== */
/*  DOM References                                                      */
/* ================================================================== */

const authView = document.getElementById("auth-view");
const appView = document.getElementById("app-view");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");
const sidebarCollapseBtn = document.getElementById("sidebar-collapse-btn");
const mobileMenuBtn = document.getElementById("mobile-menu-btn");
const mainInner = document.getElementById("main-inner");

const themeToggleBtn = document.getElementById("theme-toggle");
const logoutSidebarBtn = document.getElementById("logout-sidebar-btn");
const sidebarEmail = document.getElementById("sidebar-email");
const sidebarAvatar = document.getElementById("sidebar-avatar");

const authTabs = document.querySelectorAll(".auth-tab");
const loginForm = document.getElementById("login-form");
const registerForm = document.getElementById("register-form");
const loginError = document.getElementById("login-error");
const registerError = document.getElementById("register-error");
const loginSubmit = loginForm.querySelector(".btn-primary");
const registerSubmit = registerForm.querySelector(".btn-primary");

const navLinks = document.querySelectorAll("[data-nav]");
const mirrorLens = document.getElementById("mirror-lens");
const sherlockLens = document.getElementById("sherlock-lens");
const historyLens = document.getElementById("history-lens");

const mindForm = document.getElementById("mind-form");
const thought = document.getElementById("thought");
const output = document.getElementById("output");
const outputSection = document.getElementById("output-section");
const mindButton = mindForm.querySelector(".btn-primary");
const charCounter = document.getElementById("char-counter");

const entriesList = document.getElementById("entries-list");
const entriesTimeline = document.getElementById("entries-timeline");
const refreshEntriesBtn = document.getElementById("refresh-entries");
const refreshEntriesHistoryBtn = document.getElementById("refresh-entries-history");

const generateInsightsBtn = document.getElementById("generate-insights");
const insightsList = document.getElementById("insights-list");
const sherlockHint = document.getElementById("sherlock-hint");

const shortcutsModal = document.getElementById("shortcuts-modal");
const closeShortcutsBtn = document.getElementById("close-shortcuts");

/* ================================================================== */
/*  Helpers                                                             */
/* ================================================================== */

async function apiCall(path, options = {}) {
    let response;
    try {
        response = await fetch(`${API}${path}`, {
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            ...options,
        });
    } catch (_err) {
        return { ok: false, status: 0, body: { error: "Cannot reach the server. Check your connection." } };
    }
    let body = null;
    try {
        body = await response.json();
    } catch (_e) {
        /* 204, redirect, etc. */
    }
    return { ok: response.ok, status: response.status, body };
}

function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "className") node.className = v;
        else if (k.startsWith("on") && typeof v === "function") {
            node.addEventListener(k.slice(2).toLowerCase(), v);
        } else if (k === "style" && typeof v === "object") {
            Object.assign(node.style, v);
        } else {
            node.setAttribute(k, v);
        }
    }
    for (const child of children) {
        if (typeof child === "string") node.appendChild(document.createTextNode(child));
        else if (child instanceof Node) node.appendChild(child);
    }
    return node;
}

function text(str) {
    return document.createTextNode(str ?? "");
}

function show(node) { if (node) node.classList.remove("hidden"); }
function hide(node) { if (node) node.classList.add("hidden"); }

function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
        month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
    });
}

function setButtonLoading(btn, isLoading, defaultHTML) {
    if (!btn) return;
    btn.disabled = isLoading;
    if (isLoading) {
        btn._originalHTML = btn.innerHTML;
        btn.innerHTML = '<span class="loading"></span>';
    } else {
        btn.innerHTML = btn._originalHTML || defaultHTML;
    }
}

function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
}

/* ================================================================== */
/*  Toast Notifications                                                 */
/* ================================================================== */

function showToast(message, type = "") {
    const container = document.getElementById("toast-container");
    const toast = el("div", { className: `toast ${type}` });
    toast.appendChild(text(message));
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(40px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

/* ================================================================== */
/*  Sidebar Management                                                  */
/* ================================================================== */

function isMobile() {
    return window.innerWidth <= 1024;
}

function openSidebar() {
    sidebar.classList.add("mobile-open");
    sidebarOverlay.classList.add("visible");
    document.body.style.overflow = "hidden";
}

function closeSidebar() {
    sidebar.classList.remove("mobile-open");
    sidebarOverlay.classList.remove("visible");
    document.body.style.overflow = "";
}

function toggleSidebarCollapse() {
    if (isMobile()) {
        sidebar.classList.contains("mobile-open") ? closeSidebar() : openSidebar();
    } else {
        sidebar.classList.toggle("collapsed");
        // Rotate the collapse icon
        const icon = sidebarCollapseBtn.querySelector("svg");
        if (sidebar.classList.contains("collapsed")) {
            icon.style.transform = "rotate(180deg)";
        } else {
            icon.style.transform = "rotate(0deg)";
        }
    }
}

// Persist sidebar collapsed state
(function initSidebarState() {
    const stored = localStorage.getItem("cm-sidebar-collapsed");
    if (stored === "true" && !isMobile()) {
        sidebar.classList.add("collapsed");
        const icon = sidebarCollapseBtn?.querySelector("svg");
        if (icon) icon.style.transform = "rotate(180deg)";
    }
})();

sidebarCollapseBtn.addEventListener("click", () => {
    toggleSidebarCollapse();
    const collapsed = sidebar.classList.contains("collapsed");
    if (!isMobile()) localStorage.setItem("cm-sidebar-collapsed", collapsed.toString());
});

sidebarOverlay.addEventListener("click", closeSidebar);
mobileMenuBtn.addEventListener("click", openSidebar);

// Close sidebar on nav for mobile
document.querySelectorAll("[data-nav]").forEach(link => {
    link.addEventListener("click", () => {
        if (isMobile()) closeSidebar();
    });
});

/* ================================================================== */
/*  Navigation / Lens Switching                                         */
/* ================================================================== */

function switchLens(lens) {
    // Hide all lens panels
    [mirrorLens, sherlockLens, historyLens].forEach(p => hide(p));

    // Show the target
    if (lens === "mirror") show(mirrorLens);
    else if (lens === "sherlock") { show(sherlockLens); loadInsights(); }
    else if (lens === "history") { show(historyLens); loadTimelineEntries(); }

    // Update nav active state
    navLinks.forEach(link => {
        link.classList.toggle("active", link.dataset.nav === lens);
    });

    // Scroll to top
    mainInner.scrollIntoView({ behavior: "smooth", block: "start" });
}

navLinks.forEach(link => {
    link.addEventListener("click", () => {
        const lens = link.dataset.nav;
        if (lens) switchLens(lens);
    });
});

/* ================================================================== */
/*  Character Counter                                                   */
/* ================================================================== */

function updateCharCounter() {
    const remaining = MAX_CHARS - thought.value.length;
    if (charCounter) {
        charCounter.textContent = `${remaining} characters remaining`;
        charCounter.className = remaining < 50
            ? "char-counter char-counter-warn"
            : "char-counter";
    }
}

if (thought && charCounter) {
    thought.addEventListener("input", updateCharCounter);
    updateCharCounter();
}

/* ================================================================== */
/*  Auth Flow                                                           */
/* ================================================================== */

authTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        authTabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        if (tab.dataset.tab === "login") {
            show(loginForm);
            hide(registerForm);
        } else {
            hide(loginForm);
            show(registerForm);
        }
    });
});

async function checkSession() {
    const { ok, body } = await apiCall("/auth/me");
    if (ok && body && body.user) {
        enterApp(body.user);
    } else {
        showAuth();
    }
}

function showAuth() {
    show(authView);
    hide(appView);
    document.body.style.overflow = "";
}

function enterApp(user) {
    hide(authView);
    show(appView);
    sidebarEmail.textContent = user.email;
    if (sidebarAvatar) {
        sidebarAvatar.textContent = user.email.charAt(0).toUpperCase();
    }
    switchLens("mirror");
    loadEntries();
}

loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    loginError.textContent = "";
    const email = document.getElementById("login-email").value.trim();
    const password = document.getElementById("login-password").value;

    if (!email || !password) {
        loginError.textContent = "Email and password are required.";
        return;
    }

    setButtonLoading(loginSubmit, true);

    const { ok, body } = await apiCall("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
    });

    setButtonLoading(loginSubmit, false);

    if (!ok) {
        loginError.textContent = body?.error || "Unable to sign in.";
        return;
    }
    enterApp(body.user);
});

registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    registerError.textContent = "";
    const email = document.getElementById("register-email").value.trim();
    const password = document.getElementById("register-password").value;

    if (!email || !password) {
        registerError.textContent = "Email and password are required.";
        return;
    }
    if (password.length < 8) {
        registerError.textContent = "Password must be at least 8 characters.";
        return;
    }

    setButtonLoading(registerSubmit, true);

    const { ok, body } = await apiCall("/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password }),
    });

    setButtonLoading(registerSubmit, false);

    if (!ok) {
        registerError.textContent = body?.error || "Unable to create account.";
        return;
    }
    enterApp(body.user);
});

logoutSidebarBtn.addEventListener("click", async () => {
    await apiCall("/auth/logout", { method: "POST" });
    loginForm.reset();
    registerForm.reset();
    loginError.textContent = "";
    registerError.textContent = "";
    showAuth();
    if (isMobile()) closeSidebar();
});

/* ================================================================== */
/*  Mirror: Submit Entry                                                */
/* ================================================================== */

/**
 * Natural language descriptions for emotions.
 * Replaces robotic text with thoughtful AI-personality responses.
 */
const EMOTION_DESCRIPTIONS = {
    joy: "Your writing suggests a bright, optimistic emotional state. There are strong signals of happiness, enthusiasm, and positive engagement with your thoughts.",
    sadness: "Your entry carries a reflective, somewhat heavy emotional weight. There are indicators of thoughtful melancholy — the kind that often accompanies deep introspection.",
    anger: "Your words convey a sense of frustration and intensity. This kind of emotional energy often surfaces when something important feels unresolved or unjust.",
    fear: "Your writing suggests underlying anxiety or apprehension. These indicators often point to uncertainty about the future or concern about something beyond your control.",
    love: "Your entry radiates warmth and deep affection. The emotional signature here suggests strong connection, care, and genuine positive regard.",
    surprise: "Your writing suggests unexpected realizations or fresh perspectives. There's a quality of discovery and openness to new experiences.",
    neutral: "Your entry maintains a balanced, measured tone. This often reflects a state of calm observation or thoughtful objectivity.",
};

function getEmotionDescription(emotion) {
    return EMOTION_DESCRIPTIONS[emotion] ||
        `Your writing reveals a ${emotion} emotional signature — each entry adds depth to understanding your cognitive patterns.`;
}

/**
 * Get the appropriate emoji icon for an emotion.
 */
function getEmotionIcon(emotion) {
    const icons = {
        joy: "✨",
        sadness: "🌧️",
        anger: "⚡",
        fear: "🌊",
        love: "💜",
        surprise: "🌟",
        neutral: "🧘",
    };
    return icons[emotion] || "🔮";
}

/**
 * Build the premium dashboard result from prediction data.
 */
function buildMirrorResult(entry) {
    const confidencePercent = Math.round((entry.confidence || 0) * 100);
    const emotion = entry.emotion || "neutral";
    const sentiment = entry.sentiment || "neutral";
    const mindState = entry.mind_state || "";
    const distortions = entry.distortions || [];

    const circumference = 2 * Math.PI * 45; // r=45
    const offset = circumference * (1 - (entry.confidence || 0));

    const wrapper = el("div", { className: "dashboard" });

    /* ---- Emotion Hero Card ---- */
    const hero = el("div", { className: "glass-card emotion-hero stagger-1" });

    // Animated ring
    const ring = el("div", { className: "emotion-hero-ring" });
    ring.innerHTML = `
        <svg viewBox="0 0 100 100">
            <circle class="emotion-hero-ring-bg" cx="50" cy="50" r="45"/>
            <circle class="emotion-hero-ring-fill" cx="50" cy="50" r="45"
                stroke-dasharray="${circumference}"
                stroke-dashoffset="${circumference}"
                data-target-offset="${offset}"/>
        </svg>`;
    const icon = el("div", { className: "emotion-hero-icon" }, [text(getEmotionIcon(emotion))]);
    ring.appendChild(icon);
    hero.appendChild(ring);

    hero.appendChild(el("div", { className: "emotion-hero-label" }, [text("Primary emotion")]));
    hero.appendChild(el("div", { className: "emotion-hero-value" }, [text(capitalize(emotion))]));
    hero.appendChild(el("p", { className: "emotion-hero-description" },
        [text(getEmotionDescription(emotion))]));
    wrapper.appendChild(hero);

    /* ---- Stats Row ---- */
    const statsRow = el("div", { className: "stats-row" });

    // Confidence stat card
    const confCard = el("div", { className: "stat-card confidence stagger-2" });
    confCard.appendChild(el("div", { className: "stat-card-icon" }, [text("📊")]));
    confCard.appendChild(el("div", { className: "stat-card-label" }, [text("Confidence")]));
    confCard.appendChild(el("div", { className: "stat-card-value" }, [text(`${confidencePercent}%`)]));
    const confBar = el("div", { className: "stat-card-bar" });
    const confBarFill = el("div", {
        className: "stat-card-bar-fill",
        style: { width: "0%", transition: "width 1s cubic-bezier(0.34, 1.56, 0.64, 1)" }
    });
    confBarFill.dataset.targetWidth = `${confidencePercent}%`;
    confBar.appendChild(confBarFill);
    confCard.appendChild(confBar);
    statsRow.appendChild(confCard);

    // Sentiment stat card
    const sentClass = `sentiment ${sentiment}`;
    const sentIcon = sentiment === "positive" ? "😊" : sentiment === "negative" ? "😔" : "😐";
    const sentCard = el("div", { className: `stat-card ${sentClass} stagger-2` });
    sentCard.appendChild(el("div", { className: "stat-card-icon" }, [text(sentIcon)]));
    sentCard.appendChild(el("div", { className: "stat-card-label" }, [text("Sentiment")]));
    sentCard.appendChild(el("div", { className: "stat-card-value" }, [text(capitalize(sentiment))]));
    const sentBar = el("div", { className: "stat-card-bar" });
    const sentPercent = sentiment === "positive" ? "75%" : sentiment === "negative" ? "30%" : "55%";
    const sentBarFill = el("div", {
        className: "stat-card-bar-fill",
        style: { width: "0%", transition: "width 1s cubic-bezier(0.34, 1.56, 0.64, 1)" }
    });
    sentBarFill.dataset.targetWidth = sentPercent;
    sentBar.appendChild(sentBarFill);
    sentCard.appendChild(sentBar);
    statsRow.appendChild(sentCard);

    // Emotion stat card
    const emoCard = el("div", { className: "stat-card emotion stagger-2" });
    emoCard.appendChild(el("div", { className: "stat-card-icon" }, [text(getEmotionIcon(emotion))]));
    emoCard.appendChild(el("div", { className: "stat-card-label" }, [text("Intensity")]));
    const intensity = confidencePercent > 75 ? "Strong" : confidencePercent > 45 ? "Moderate" : "Subtle";
    emoCard.appendChild(el("div", { className: "stat-card-value", style: { fontSize: "var(--text-xl)" } },
        [text(intensity)]));
    const emoBar = el("div", { className: "stat-card-bar" });
    const emoBarFill = el("div", {
        className: "stat-card-bar-fill",
        style: { width: "0%", transition: "width 1s cubic-bezier(0.34, 1.56, 0.64, 1)" }
    });
    emoBarFill.dataset.targetWidth = `${confidencePercent}%`;
    emoBar.appendChild(emoBarFill);
    emoCard.appendChild(emoBar);
    statsRow.appendChild(emoCard);

    wrapper.appendChild(statsRow);

    /* ---- Cognitive State Insight Panel ---- */
    if (mindState) {
        const insightPanel = el("div", { className: "insight-panel stagger-3" });
        const header = el("div", { className: "insight-panel-header" });
        const panelIcon = el("div", { className: "insight-panel-icon" }, [text("🧠")]);
        const headerText = el("div");
        headerText.appendChild(el("div", { className: "insight-panel-label" }, [text("Cognitive insight")]));
        headerText.appendChild(el("div", { className: "insight-panel-title" }, [text("What the Mirror notices")]));
        header.appendChild(panelIcon);
        header.appendChild(headerText);
        insightPanel.appendChild(header);
        insightPanel.appendChild(el("div", { className: "insight-panel-body" }, [text(mindState)]));
        wrapper.appendChild(insightPanel);
    }

    /* ---- Distortions Section ---- */
    if (distortions && distortions.length > 0) {
        const distSection = el("div", { className: "distortions-section stagger-4" });
        const distHeader = el("div", { className: "distortions-header" });
        distHeader.appendChild(el("span", { className: "distortions-label" }, [text("Cognitive distortions detected")]));
        distHeader.appendChild(el("span", { className: "distortions-count" },
            [text(`(${distortions.length})`)]));
        distSection.appendChild(distHeader);

        const distList = el("div", { className: "distortions-list" });
        for (const d of distortions) {
            const item = el("div", { className: "distortion-item" });
            item.appendChild(el("span", { className: "distortion-label-pill" }, [text(d.label)]));
            item.appendChild(el("p", { className: "distortion-sentence" }, [text(`"${d.sentence}"`)]));
            item.appendChild(el("p", { className: "distortion-explanation" }, [text(d.explanation)]));
            distList.appendChild(item);
        }
        distSection.appendChild(distList);
        wrapper.appendChild(distSection);
    }

    clearNode(output);
    output.appendChild(wrapper);

    // Animate the ring and bars after render
    requestAnimationFrame(() => {
        // Animate ring
        const ringFill = output.querySelector(".emotion-hero-ring-fill");
        if (ringFill) {
            const targetOffset = parseFloat(ringFill.dataset.targetOffset);
            ringFill.style.strokeDashoffset = targetOffset;
        }

        // Animate bars
        const bars = output.querySelectorAll(".stat-card-bar-fill");
        bars.forEach((bar, i) => {
            setTimeout(() => {
                bar.style.width = bar.dataset.targetWidth || "0%";
            }, i * 150);
        });
    });

    outputSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function buildMirrorError(message) {
    clearNode(output);
    const state = el("div", { className: "empty-state" });
    state.appendChild(el("div", { className: "empty-icon" },
        [text("⚠️")]));
    state.appendChild(el("h3", {}, [text("Unable to process")]));
    state.appendChild(el("p", {}, [text(message || "Something went wrong. Please try again.")]));
    output.appendChild(state);
}

mindForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const textVal = thought.value.trim();
    if (!textVal) return;

    setButtonLoading(mindButton, true);

    const { ok, body } = await apiCall("/entries", {
        method: "POST",
        body: JSON.stringify({ text: textVal }),
    });

    setButtonLoading(mindButton, false);

    if (!ok) {
        buildMirrorError(body?.error);
        return;
    }

    buildMirrorResult(body.entry);
    thought.value = "";
    updateCharCounter();
    loadEntries();
    loadTimelineEntries();
});

/* ================================================================== */
/*  Entry History (Mirror page)                                         */
/* ================================================================== */

function buildEntryCard(entry) {
    const emotion = entry.emotion || "neutral";

    const container = el("div", { className: "timeline-entry" });
    const dot = el("div", { className: `timeline-dot emotion-${emotion}` });
    container.appendChild(dot);

    const card = el("div", { className: "entry-card" });
    const header = el("div", { className: "entry-card-header" });
    header.appendChild(el("span", { className: "entry-date" }, [text(formatDate(entry.created_at))]));
    const meta = el("div", { className: "entry-meta" });
    meta.appendChild(el("span", { className: "entry-tag" },
        [text(`${capitalize(emotion)} · ${capitalize(entry.sentiment || "neutral")}`)]));
    header.appendChild(meta);
    card.appendChild(header);

    const excerpt = entry.text.length > 160
        ? entry.text.slice(0, 160) + "..."
        : entry.text;
    card.appendChild(el("p", { className: "entry-excerpt" }, [text(excerpt)]));

    if (entry.distortions && entry.distortions.length > 0) {
        const tags = el("div", { className: "entry-distortion-tags" });
        for (const d of entry.distortions) {
            tags.appendChild(el("span", { className: "tag-chip" }, [text(d.label)]));
        }
        card.appendChild(tags);
    }

    container.appendChild(card);
    return container;
}

async function loadEntries() {
    if (!entriesList) return;
    const { ok, body } = await apiCall("/entries?per_page=5");
    clearNode(entriesList);

    if (!ok || !body || !body.entries || body.entries.length === 0) {
        entriesList.appendChild(el("p", {
            className: "text-muted text-center",
            style: { padding: "var(--space-xl) 0" }
        }, [text("No entries yet. Your reflections will appear here.")]));
        return;
    }

    for (const entry of body.entries) {
        entriesList.appendChild(buildEntryCard(entry));
    }
}

async function loadTimelineEntries() {
    if (!entriesTimeline) return;
    const { ok, body } = await apiCall("/entries?per_page=20");
    clearNode(entriesTimeline);

    if (!ok || !body || !body.entries || body.entries.length === 0) {
        entriesTimeline.appendChild(el("p", {
            className: "text-muted text-center",
            style: { padding: "var(--space-xl) 0" }
        }, [text("No entries yet.")]));
        return;
    }

    for (const entry of body.entries) {
        entriesTimeline.appendChild(buildEntryCard(entry));
    }
}

if (refreshEntriesBtn) refreshEntriesBtn.addEventListener("click", loadEntries);
if (refreshEntriesHistoryBtn) refreshEntriesHistoryBtn.addEventListener("click", loadTimelineEntries);

/* ================================================================== */
/*  Sherlock Lens                                                       */
/* ================================================================== */

function buildInsightCard(insight) {
    const card = el("div", { className: "insight-card" });
    card.appendChild(el("span", { className: "insight-type" }, [text(insight.insight_type)]));
    card.appendChild(el("p", { className: "insight-deduction" }, [text(insight.deduction)]));

    if (insight.evidence && insight.evidence.length > 0) {
        const evidenceSection = el("div", { className: "insight-evidence" });
        evidenceSection.appendChild(el("span", { className: "insight-evidence-label" }, [text("Evidence")]));
        const ul = el("ul");
        for (const ev of insight.evidence) {
            const evExcerpt = ev.excerpt.length >= 140
                ? ev.excerpt.slice(0, 140) + "..."
                : ev.excerpt;
            ul.appendChild(el("li", {}, [text(`"${evExcerpt}"`)]));
        }
        evidenceSection.appendChild(ul);
        card.appendChild(evidenceSection);
    }

    return card;
}

function buildInsightsEmpty() {
    const state = el("div", { className: "empty-state" });
    state.appendChild(el("div", { className: "empty-icon" }, [text("🔍")]));
    state.appendChild(el("h3", {}, [text("Not enough data yet")]));
    state.appendChild(el("p", {}, [
        text("Keep journaling. Patterns need at least 4 entries before the Sherlock Lens can draw reliable deductions across your history.")
    ]));
    return state;
}

function renderInsights(insights) {
    clearNode(insightsList);

    if (!insights || insights.length === 0) {
        insightsList.appendChild(buildInsightsEmpty());
        return;
    }

    for (const insight of insights) {
        insightsList.appendChild(buildInsightCard(insight));
    }
}

async function loadInsights() {
    if (!insightsList) return;
    const { ok, body } = await apiCall("/insights");
    if (ok && body) {
        renderInsights(body.insights);
    }
}

if (generateInsightsBtn) {
    generateInsightsBtn.addEventListener("click", async () => {
        setButtonLoading(generateInsightsBtn, true);
        if (sherlockHint) sherlockHint.textContent = "";
        clearNode(insightsList);

        const { ok, body } = await apiCall("/insights/generate", { method: "POST" });

        setButtonLoading(generateInsightsBtn, false);

        if (!ok) {
            if (sherlockHint) sherlockHint.textContent = body?.error || "Unable to generate insights right now.";
            return;
        }

        renderInsights(body.insights);
    });
}

/* ================================================================== */
/*  Theme Toggle                                                        */
/* ================================================================== */

if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", cycleTheme);
}

/* ================================================================== */
/*  Keyboard Shortcuts                                                  */
/* ================================================================== */

document.addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;

    // Cmd/Ctrl + Enter: Submit the mind form
    if (mod && e.key === "Enter") {
        e.preventDefault();
        if (mindForm && !mindForm.closest(".hidden")) {
            mindForm.dispatchEvent(new Event("submit"));
        }
    }

    // Cmd/Ctrl + T: Toggle theme
    if (mod && e.key === "t") {
        e.preventDefault();
        cycleTheme();
    }

    // Cmd/Ctrl + B: Toggle sidebar
    if (mod && e.key === "b") {
        e.preventDefault();
        toggleSidebarCollapse();
    }

    // ? : Show keyboard shortcuts
    if (e.key === "?" && !mod && document.activeElement === document.body) {
        e.preventDefault();
        show(shortcutsModal);
    }

    // Esc: Close shortcuts modal
    if (e.key === "Escape" && !shortcutsModal.classList.contains("hidden")) {
        hide(shortcutsModal);
    }

    // Esc: Close mobile sidebar
    if (e.key === "Escape" && isMobile() && sidebar.classList.contains("mobile-open")) {
        closeSidebar();
    }
});

if (closeShortcutsBtn) {
    closeShortcutsBtn.addEventListener("click", () => hide(shortcutsModal));
}

if (shortcutsModal) {
    shortcutsModal.addEventListener("click", (e) => {
        if (e.target === shortcutsModal) hide(shortcutsModal);
    });
}

/* ================================================================== */
/*  Init                                                               */
/* ================================================================== */

const year = document.getElementById("year");
if (year) year.textContent = new Date().getFullYear();

checkSession();
