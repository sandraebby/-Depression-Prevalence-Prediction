🧠 Depression Prevalence Predictor — Streamlit App

A friendly, interactive app that estimates country-level depression prevalence using a neural network that learns country-specific patterns (entity embeddings) and other mental-health indicators. Built with PyTorch and Streamlit — made to be useful, fun, and easy to share.

🎯 What this app does (quick elevator pitch)

Pick a country, slide in a few mental-health indicators (like anxiety and bipolar prevalence), and the app returns a clear, human-friendly estimate of depressive disorder prevalence for that country — with a helpful indicator (low / moderate / high). Perfect for exploration, education, and quick what-if checks.

✨ Highlights / Why it’s cool

Entity embeddings: the model learns a compact vector for each country so predictions reflect both global trends and country-specific patterns.

Minimal, beautiful UI: sliders and a clean sidebar for fast experimentation.

Reproducible: uses the exact preprocessing (LabelEncoder, StandardScaler) from training so predictions are consistent.

Lightweight deployment: works on Streamlit Community Cloud (CPU-only) — no GPU required.

🩺 Interpreting predictions

Output: Predicted share of population with depressive disorders (as a percentage).

The app also gives a simple interpretation:

Low → typically below ~2% (example threshold — interpret cautiously)

Moderate → intermediate range

High → above ~5% (example threshold)

These thresholds are heuristics for quick interpretation — treat them as guides, not clinical cutoffs.
