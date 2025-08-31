# Shopify-Merchant-Sales-Demo
Chat GPT wrapper that takes a shopper prompt and recommends clothing based on a merchant's dummy data from Shopify

# 👗 AI Style Picker (Gradio)

Describe your personal style and get two matching items from a product CSV. The app cleans your catalog, calls an LLM to pick the best two items (with a keyword fallback if the LLM isn’t available), and displays images, prices, and short “why” blurbs.

---

## ✨ Features

- **One-prompt picker:** Users describe their style; the app returns exactly **two** items.
- **LLM + fallback:** Uses OpenAI Chat Completions; if it fails or rate-limits, falls back to a **keyword ranker**.
- **CSV cleaning & dedupe:** Strips HTML, formats prices, builds variant labels, picks best image (variant image preferred), and **deduplicates variants**.
- **Variant-aware output:** Shows *Display Title*, optional *Variant*, *SKU*, and *Variant ID* if present.
- **Fast, simple UI:** Built with `gradio.Blocks`.

---

## 🧰 Tech Stack

- **Python** (3.9+)
- **Gradio** UI
- **OpenAI** (>= 1.0 client)
- **pandas** for CSV cleaning

---

## 📂 CSV Schema

**Required columns**
- `Title`
- `Body (HTML)` — long description; HTML is stripped automatically
- `Variant Price`

**
