/**
 * script.js -- Frontend Logic for Email Classifier
 *
 * Handles three flows:
 *   Tab 1 (Paste Email):  Classify a single pasted email
 *   Tab 2 (Scan Inbox):
 *     - Google Sign-In:   OAuth 2.0 -> Gmail API scan
 *     - IMAP fallback:    Manual credentials -> IMAP scan
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const API_URL = "";

// *** IMPORTANT: Replace this with your Google Cloud OAuth Client ID ***
// Get one at: https://console.cloud.google.com/apis/credentials
const GOOGLE_CLIENT_ID = "538253304384-vctkb5kpe2adll3i5as2tc9oljchrdog.apps.googleusercontent.com";

// Gmail API scope (read-only)
const GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly";

// ---------------------------------------------------------------------------
// DOM Elements -- Tab 1 (Paste Email)
// ---------------------------------------------------------------------------

const emailInput = document.getElementById("email-input");
const classifyBtn = document.getElementById("classify-btn");
const loadingEl = document.getElementById("loading");
const resultCard = document.getElementById("result-card");
const resultBadge = document.getElementById("result-badge");
const resultIcon = document.getElementById("result-icon");
const resultLabel = document.getElementById("result-label");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceValue = document.getElementById("confidence-value");
const flaggedWordsEl = document.getElementById("flagged-words");
const highlightedPreview = document.getElementById("highlighted-preview");
const highlightedText = document.getElementById("highlighted-text");
const modelSourceEl = document.getElementById("model-source");

// ---------------------------------------------------------------------------
// DOM Elements -- AI Status
// ---------------------------------------------------------------------------

const aiStatusBadge = document.getElementById("ai-status");
const aiStatusDot = document.querySelector(".ai-status-dot");
const aiStatusText = document.querySelector(".ai-status-text");

// ---------------------------------------------------------------------------
// DOM Elements -- Tab 2 (Scan Inbox)
// ---------------------------------------------------------------------------

// Google Sign-In elements
const googleSignInSection = document.getElementById("google-signin-section");
const googleProfileSection = document.getElementById("google-profile-section");
const googleSignInBtn = document.getElementById("google-signin-btn");
const googleAvatar = document.getElementById("google-avatar");
const googleName = document.getElementById("google-name");
const googleEmail = document.getElementById("google-email");
const googleScanCount = document.getElementById("google-scan-count");

// IMAP elements
const scanEmailInput = document.getElementById("scan-email");
const scanPasswordInput = document.getElementById("scan-password");
const scanProviderSelect = document.getElementById("scan-provider");
const scanCountSelect = document.getElementById("scan-count");
const customHostGroup = document.getElementById("custom-host-group");
const customHostInput = document.getElementById("scan-custom-host");
const scanBtn = document.getElementById("scan-btn");
const testConnBtn = document.getElementById("test-conn-btn");
const connStatus = document.getElementById("conn-status");

// Shared scan elements
const scanLoading = document.getElementById("scan-loading");
const scanLoadingText = document.getElementById("scan-loading-text");
const scanResults = document.getElementById("scan-results");
const emailCardsList = document.getElementById("email-cards-list");
const statTotal = document.getElementById("stat-total");
const statHam = document.getElementById("stat-ham");
const statSpam = document.getElementById("stat-spam");
const statPhishing = document.getElementById("stat-phishing");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let scanResultsData = [];
let currentFilter = "all";
let googleAccessToken = null;
let tokenClient = null;

// ---------------------------------------------------------------------------
// Label Configuration
// ---------------------------------------------------------------------------

const LABEL_CONFIG = {
    ham: { icon: "\u2705", displayName: "Legitimate (Ham)", cssClass: "ham" },
    spam: { icon: "\u26A0\uFE0F", displayName: "Spam", cssClass: "spam" },
    phishing: { icon: "\uD83D\uDEA8", displayName: "Phishing", cssClass: "phishing" },
};

// ==========================================================================
// TAB NAVIGATION
// ==========================================================================

function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tabId);
    });
    document.querySelectorAll(".tab-content").forEach((panel) => {
        panel.classList.toggle("active", panel.id === "tab-" + tabId);
    });
}

// ==========================================================================
// TAB 1: PASTE EMAIL
// ==========================================================================

async function classifyEmail() {
    const text = emailInput.value.trim();
    if (!text) { shakeElement(emailInput); return; }

    setLoading(true, loadingEl, classifyBtn);
    hideResults();

    try {
        const response = await fetch(API_URL + "/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email_text: text }),
        });
        if (!response.ok) {
            const err = await response.json().catch(function() { return {}; });
            throw new Error(err.error || "Server error (" + response.status + ")");
        }
        const data = await response.json();
        displayResult(data, text);
    } catch (error) {
        showError(error.message || "Failed to connect to the API. Is the Flask server running?");
    } finally {
        setLoading(false, loadingEl, classifyBtn);
    }
}

function displayResult(data, originalText) {
    const config = LABEL_CONFIG[data.label] || LABEL_CONFIG.ham;
    resultBadge.className = "result-badge " + config.cssClass;
    resultIcon.textContent = config.icon;
    resultLabel.textContent = config.displayName;

    const pct = (data.confidence * 100).toFixed(1);
    confidenceBar.className = "confidence-bar " + config.cssClass;
    confidenceBar.style.width = pct + "%";
    confidenceValue.textContent = pct + "%";

    flaggedWordsEl.innerHTML = "";
    if (data.flagged_words && data.flagged_words.length > 0) {
        data.flagged_words.forEach(function(word) {
            var tag = document.createElement("span");
            tag.className = "flagged-word-tag";
            tag.textContent = word;
            flaggedWordsEl.appendChild(tag);
        });
    } else {
        flaggedWordsEl.innerHTML = '<span style="color:var(--text-muted);font-size:0.85rem">No suspicious words flagged</span>';
    }
    highlightedText.innerHTML = highlightWords(originalText, data.flagged_words || []);

    // Show model source indicator
    if (data.model_used) {
        var modelNames = {
            "ai": "\uD83E\uDD16 Classified by AI Model (BART-Large-MNLI)",
            "sklearn": "\uD83D\uDCCA Classified by Trained ML Model",
            "rule_based": "\uD83D\uDD27 Classified by Rule Engine (Fallback)",
        };
        var modelClasses = {
            "ai": "model-ai",
            "sklearn": "model-sklearn",
            "rule_based": "model-rules",
        };
        modelSourceEl.textContent = modelNames[data.model_used] || data.model_used;
        modelSourceEl.className = "model-source " + (modelClasses[data.model_used] || "");
        modelSourceEl.classList.remove("hidden");
    } else {
        modelSourceEl.classList.add("hidden");
    }

    resultCard.classList.remove("hidden");
    highlightedPreview.classList.remove("hidden");
}

function hideResults() {
    resultCard.classList.add("hidden");
    highlightedPreview.classList.add("hidden");
}

function clearAll() {
    emailInput.value = "";
    hideResults();
    loadingEl.classList.add("hidden");
}

function showError(message) {
    resultBadge.className = "result-badge phishing";
    resultIcon.textContent = "\u274C";
    resultLabel.textContent = "Error";
    confidenceBar.style.width = "0%";
    confidenceValue.textContent = "\u2014";
    flaggedWordsEl.innerHTML = '<span style="color:var(--color-phishing);font-size:0.85rem">' + escapeHtml(message) + '</span>';
    highlightedPreview.classList.add("hidden");
    resultCard.classList.remove("hidden");
}

// ==========================================================================
// TAB 2: GOOGLE SIGN-IN (OAuth 2.0 + Gmail API)
// ==========================================================================

/**
 * Initialize the Google Identity Services token client.
 * Called once when the page loads.
 */
function initGoogleAuth() {
    if (typeof google === "undefined" || !google.accounts || !google.accounts.oauth2) {
        // GIS library not loaded yet; retry after a short delay
        setTimeout(initGoogleAuth, 500);
        return;
    }

    if (GOOGLE_CLIENT_ID === "YOUR_CLIENT_ID_HERE.apps.googleusercontent.com") {
        console.warn("[Google Auth] Client ID not configured. Replace GOOGLE_CLIENT_ID in script.js.");
        return;
    }

    tokenClient = google.accounts.oauth2.initTokenClient({
        client_id: GOOGLE_CLIENT_ID,
        scope: GMAIL_SCOPE + " profile email",
        callback: handleGoogleTokenResponse,
    });
}

/**
 * Triggered when user clicks the "Sign in with Google" button.
 */
function initiateGoogleSignIn() {
    if (GOOGLE_CLIENT_ID === "YOUR_CLIENT_ID_HERE.apps.googleusercontent.com") {
        showConnStatus(
            "Google Client ID not configured. Open script.js and replace GOOGLE_CLIENT_ID with your OAuth Client ID from Google Cloud Console.",
            false
        );
        return;
    }

    if (!tokenClient) {
        initGoogleAuth();
        setTimeout(function() {
            if (tokenClient) tokenClient.requestAccessToken();
        }, 600);
        return;
    }

    tokenClient.requestAccessToken();
}

/**
 * Callback after Google OAuth token is received.
 */
function handleGoogleTokenResponse(response) {
    if (response.error) {
        console.error("Google auth error:", response);
        showConnStatus("Google sign-in failed: " + (response.error_description || response.error), false);
        return;
    }

    googleAccessToken = response.access_token;

    // Fetch user profile info
    fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
        headers: { Authorization: "Bearer " + googleAccessToken },
    })
        .then(function(r) { return r.json(); })
        .then(function(profile) {
            showGoogleProfile(profile);
            // Auto-start scanning
            googleScanInbox();
        })
        .catch(function(err) {
            console.error("Profile fetch error:", err);
            // Still scan even if profile fetch fails
            showGoogleProfile({ name: "Google User", email: "signed in", picture: "" });
            googleScanInbox();
        });
}

/**
 * Show the signed-in user's profile card.
 */
function showGoogleProfile(profile) {
    googleName.textContent = profile.name || "Google User";
    googleEmail.textContent = profile.email || "";
    if (profile.picture) {
        googleAvatar.src = profile.picture;
        googleAvatar.style.display = "block";
    } else {
        googleAvatar.style.display = "none";
    }

    googleSignInSection.classList.add("hidden");
    googleProfileSection.classList.remove("hidden");
}

/**
 * Sign out of Google.
 */
function googleSignOut() {
    if (googleAccessToken) {
        google.accounts.oauth2.revoke(googleAccessToken, function() {
            console.log("Google token revoked");
        });
    }
    googleAccessToken = null;

    googleSignInSection.classList.remove("hidden");
    googleProfileSection.classList.add("hidden");
    scanResults.classList.add("hidden");
    scanResultsData = [];
}

/**
 * Scan Gmail inbox using the OAuth access token.
 */
async function googleScanInbox() {
    if (!googleAccessToken) {
        showConnStatus("Please sign in with Google first.", false);
        return;
    }

    var count = parseInt(googleScanCount.value, 10) || 25;

    // Show loading
    scanResults.classList.add("hidden");
    scanLoading.classList.remove("hidden");
    scanLoadingText.textContent = "Fetching emails from Gmail...";
    googleSignInBtn.disabled = true;

    try {
        var progressTimer = setTimeout(function() {
            scanLoadingText.textContent = "Classifying " + count + " emails...";
        }, 1500);

        var response = await fetch(API_URL + "/scan-inbox/google", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                access_token: googleAccessToken,
                count: count,
            }),
        });

        clearTimeout(progressTimer);

        if (!response.ok) {
            var errData = await response.json().catch(function() { return {}; });
            if (response.status === 401) {
                // Token expired, prompt re-sign-in
                googleSignOut();
                throw new Error("Session expired. Please sign in again.");
            }
            throw new Error(errData.error || "Server error (" + response.status + ")");
        }

        var result = await response.json();
        scanResultsData = result.results || [];

        renderScanSummary(result.summary, result.total);
        renderScanResults(scanResultsData);
        scanResults.classList.remove("hidden");
    } catch (error) {
        showConnStatus(error.message || "Gmail scan failed.", false);
    } finally {
        scanLoading.classList.add("hidden");
        googleSignInBtn.disabled = false;
    }
}

// ==========================================================================
// TAB 2: IMAP FALLBACK
// ==========================================================================

function onProviderChange() {
    customHostGroup.classList.toggle("hidden", scanProviderSelect.value !== "custom");
}

function togglePasswordVisibility() {
    var input = scanPasswordInput;
    var btn = input.parentElement.querySelector(".toggle-password");
    if (input.type === "password") {
        input.type = "text";
        btn.textContent = "\uD83D\uDE48";
        btn.setAttribute("aria-label", "Hide password");
    } else {
        input.type = "password";
        btn.textContent = "\uD83D\uDC41\uFE0F";
        btn.setAttribute("aria-label", "Show password");
    }
}

function getScanFormData() {
    return {
        email: scanEmailInput.value.trim(),
        password: scanPasswordInput.value,
        provider: scanProviderSelect.value,
        count: parseInt(scanCountSelect.value, 10),
        custom_host: customHostInput.value.trim(),
    };
}

function validateScanForm(data) {
    if (!data.email) { shakeElement(scanEmailInput); showConnStatus("Please enter your email address.", false); return false; }
    if (!data.password) { shakeElement(scanPasswordInput); showConnStatus("Please enter your password.", false); return false; }
    if (data.provider === "custom" && !data.custom_host) { shakeElement(customHostInput); showConnStatus("Please enter the IMAP server host.", false); return false; }
    return true;
}

async function testConnection() {
    var data = getScanFormData();
    if (!validateScanForm(data)) return;
    testConnBtn.disabled = true;
    testConnBtn.textContent = "Testing...";
    connStatus.classList.add("hidden");
    try {
        var resp = await fetch(API_URL + "/scan-inbox/test-connection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        var result = await resp.json();
        showConnStatus(result.message || (result.success ? "Connected!" : "Failed"), result.success);
    } catch (error) {
        showConnStatus("Cannot reach the API server. Is the Flask server running?", false);
    } finally {
        testConnBtn.disabled = false;
        testConnBtn.textContent = "Test Connection";
    }
}

function showConnStatus(message, success) {
    connStatus.textContent = message;
    connStatus.className = "conn-status " + (success ? "success" : "error");
    connStatus.classList.remove("hidden");
}

async function scanInbox() {
    var data = getScanFormData();
    if (!validateScanForm(data)) return;

    connStatus.classList.add("hidden");
    scanResults.classList.add("hidden");
    scanLoading.classList.remove("hidden");
    scanLoadingText.textContent = "Connecting to inbox...";
    scanBtn.disabled = true;
    testConnBtn.disabled = true;

    try {
        var progressTimer = setTimeout(function() {
            scanLoadingText.textContent = "Fetching and classifying " + data.count + " emails...";
        }, 1500);

        var response = await fetch(API_URL + "/scan-inbox", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        clearTimeout(progressTimer);

        if (!response.ok) {
            var errData = await response.json().catch(function() { return {}; });
            throw new Error(errData.error || "Server error (" + response.status + ")");
        }

        var result = await response.json();
        scanResultsData = result.results || [];
        renderScanSummary(result.summary, result.total);
        renderScanResults(scanResultsData);
        scanResults.classList.remove("hidden");
    } catch (error) {
        showConnStatus(error.message || "Scan failed.", false);
    } finally {
        scanLoading.classList.add("hidden");
        scanBtn.disabled = false;
        testConnBtn.disabled = false;
    }
}

// ==========================================================================
// SCAN RESULTS (shared between Google and IMAP)
// ==========================================================================

function renderScanSummary(summary, total) {
    animateCount(statTotal, total);
    animateCount(statHam, summary.ham || 0);
    animateCount(statSpam, summary.spam || 0);
    animateCount(statPhishing, summary.phishing || 0);
}

function animateCount(element, target) {
    var duration = 600;
    var startTime = performance.now();
    function update(currentTime) {
        var elapsed = currentTime - startTime;
        var progress = Math.min(elapsed / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3);
        element.textContent = Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function renderScanResults(results) {
    emailCardsList.innerHTML = "";
    if (results.length === 0) {
        emailCardsList.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-muted)"><p style="font-size:2rem;margin-bottom:0.5rem">No emails found' +
            (currentFilter !== "all" ? " matching this filter" : "") + '.</p></div>';
        return;
    }
    results.forEach(function(emailData, index) {
        emailCardsList.appendChild(createEmailCard(emailData, index));
    });
}

function createEmailCard(emailData, index) {
    var config = LABEL_CONFIG[emailData.label] || LABEL_CONFIG.ham;
    var pct = (emailData.confidence * 100).toFixed(1);
    var dateStr = "";
    try {
        var d = new Date(emailData.date);
        dateStr = d.toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    } catch(e) { dateStr = emailData.date || ""; }

    var card = document.createElement("div");
    card.className = "email-card label-" + emailData.label;
    card.style.animationDelay = (index * 0.04) + "s";
    card.id = "email-card-" + emailData.id;

    var flaggedHtml = "";
    if (emailData.flagged_words && emailData.flagged_words.length > 0) {
        flaggedHtml = '<div class="card-body-section"><div class="card-body-section-title">Flagged Words</div><div class="card-flagged-words">' +
            emailData.flagged_words.map(function(w) { return '<span class="flagged-word-tag">' + escapeHtml(w) + '</span>'; }).join("") +
            '</div></div>';
    }

    card.innerHTML =
        '<div class="email-card-header" onclick="toggleEmailCard(' + emailData.id + ')">' +
            '<span class="card-label-badge ' + config.cssClass + '">' + config.icon + ' ' + emailData.label + '</span>' +
            '<div class="card-info">' +
                '<div class="card-subject">' + escapeHtml(emailData.subject || "(No Subject)") + '</div>' +
                '<div class="card-meta"><span>From: ' + escapeHtml(truncate(emailData.sender, 40)) + '</span><span>' + dateStr + '</span></div>' +
            '</div>' +
            '<span class="card-confidence">' + pct + '%</span>' +
            '<span class="card-expand-icon">\u25BC</span>' +
        '</div>' +
        '<div class="email-card-body">' +
            flaggedHtml +
            '<div class="card-body-section"><div class="card-body-section-title">Email Body</div>' +
                '<div class="card-body-text">' + highlightWords(emailData.body || emailData.snippet || "(No content)", emailData.flagged_words || []) + '</div>' +
            '</div>' +
        '</div>';

    return card;
}

function toggleEmailCard(id) {
    var card = document.getElementById("email-card-" + id);
    if (card) card.classList.toggle("expanded");
}

function filterResults(label) {
    currentFilter = label;
    document.querySelectorAll(".filter-btn").forEach(function(btn) {
        btn.classList.toggle("active", btn.dataset.filter === label);
    });
    var filtered = label === "all" ? scanResultsData : scanResultsData.filter(function(e) { return e.label === label; });
    renderScanResults(filtered);
}

// ==========================================================================
// SHARED UTILITIES
// ==========================================================================

function setLoading(isLoading, spinnerEl, buttonEl) {
    spinnerEl.classList.toggle("hidden", !isLoading);
    if (buttonEl) buttonEl.disabled = isLoading;
}

function highlightWords(text, words) {
    if (!words || words.length === 0) return escapeHtml(text);
    var html = escapeHtml(text);
    var sorted = words.slice().sort(function(a, b) { return b.length - a.length; });
    sorted.forEach(function(word) {
        var escaped = escapeRegex(escapeHtml(word));
        var regex = new RegExp("(" + escaped + ")", "gi");
        html = html.replace(regex, '<span class="highlight-word">$1</span>');
    });
    return html;
}

function shakeElement(el) {
    el.style.animation = "none";
    el.offsetHeight;
    el.style.animation = "shake 0.4s ease";
    setTimeout(function() { el.style.animation = ""; }, 400);
}

function escapeHtml(text) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function truncate(str, maxLen) {
    if (!str) return "";
    return str.length > maxLen ? str.substring(0, maxLen) + "\u2026" : str;
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

emailInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) classifyEmail();
});

[scanEmailInput, scanPasswordInput, customHostInput].forEach(function(input) {
    input.addEventListener("keydown", function(e) {
        if (e.key === "Enter") scanInbox();
    });
});

// Inject shake animation
var shakeStyle = document.createElement("style");
shakeStyle.textContent = "@keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-6px)}40%{transform:translateX(6px)}60%{transform:translateX(-4px)}80%{transform:translateX(4px)}}";
document.head.appendChild(shakeStyle);

// ---------------------------------------------------------------------------
// AI Model Health Check
// ---------------------------------------------------------------------------

async function checkAIHealth() {
    try {
        var response = await fetch(API_URL + "/health");
        if (!response.ok) throw new Error("Server not reachable");
        var data = await response.json();

        if (data.ai_model && data.ai_model.loaded) {
            aiStatusBadge.className = "ai-status-badge online";
            aiStatusBadge.title = "AI Model: " + data.ai_model.model_name;
            aiStatusText.textContent = "AI Model Active";
        } else if (data.model_loaded) {
            aiStatusBadge.className = "ai-status-badge online";
            aiStatusBadge.title = "ML Model loaded (sklearn)";
            aiStatusText.textContent = "ML Model Active";
        } else {
            aiStatusBadge.className = "ai-status-badge fallback";
            aiStatusBadge.title = "Using rule-based fallback classifier";
            aiStatusText.textContent = "Rule Engine";
        }
    } catch (error) {
        aiStatusBadge.className = "ai-status-badge offline";
        aiStatusBadge.title = "Cannot connect to API server";
        aiStatusText.textContent = "Offline";
    }
}

// Initialize Google Auth and check AI health when page loads
window.addEventListener("load", function() {
    initGoogleAuth();
    checkAIHealth();
    // Re-check periodically (every 30 seconds)
    setInterval(checkAIHealth, 30000);
});
