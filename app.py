# ai_style_picker_patched.py
# Patched version of AI Style Picker with Pylance fixes and safer OpenAI handling

import os
import sys
import re
import json
import traceback
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit(
        "❌ ERROR: OPENAI_API_KEY is missing.\n"
        "Please set it in your .env file like this:\n"
        "OPENAI_API_KEY=your_api_key_here\n"
        "Then restart the program."
    )

from openai import OpenAI

CSV_PATH = os.environ.get("PRODUCT_CSV_PATH", "product_template.csv")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
MAX_DESC_CHARS = int(os.environ.get("MAX_DESC_CHARS", 220))

def strip_html(html: str) -> str:
    if not isinstance(html, str) or html.strip() == "":
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = (text.replace("&nbsp;", " ")
                .replace("&amp;", "&")
                .replace("&quot;", '"')
                .replace("&lt;", "<")
                .replace("&gt;", ">"))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # --- Required base columns ---
    base_needed = ["Title", "Body (HTML)", "Variant Price"]
    for col in base_needed:
        if col not in df.columns:
            raise ValueError(f"CSV is missing column: {col}")

    # --- Optional columns we'll handle gracefully ---
    optional_cols = [
        "Image Src",
        "Variant Image",
        "Option1 Name", "Option1 Value",
        "Option2 Name", "Option2 Value",
        "Option3 Name", "Option3 Value",
        "Variant SKU", "Variant ID",
    ]
    for c in optional_cols:
        if c not in df.columns:
            df[c] = ""

    # Sanitize: convert common text columns to strings WITHOUT turning NaN into "nan"
    text_cols = [
        "Image Src", "Variant Image",
        "Option1 Name", "Option1 Value",
        "Option2 Name", "Option2 Value",
        "Option3 Name", "Option3 Value",
        "Variant SKU", "Variant ID",
    ]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # Clean description
    df["Clean Description"] = df["Body (HTML)"].apply(strip_html)

    # Price display
    def fmt_price(v):
        try:
            return f"${float(v):.2f}"
        except Exception:
            return str(v)
    df["Price Display"] = df["Variant Price"].apply(fmt_price)

    # Build variant label like "Color: Black / Size: M"
    def build_variant_label(r):
        parts = []
        for i in (1, 2, 3):
            name = (r.get(f"Option{i} Name", "") or "").strip()
            val  = (r.get(f"Option{i} Value", "") or "").strip()

            # Skip Shopify "Default Title"
            if not val or val.lower() == "default title":
                continue

            # Some exports use name "Title" — don't show "Title: ..."
            safe_name = name if (name and name.lower() != "title") else ""
            parts.append(f"{safe_name}: {val}" if safe_name else val)
        return " / ".join(parts)

    df["Variant Label"] = df.apply(build_variant_label, axis=1)

    # Display Title: include variant label if present
    def display_title(r):
        base = str(r["Title"]).strip()
        v = str(r["Variant Label"]).strip()
        return f"{base} ({v})" if v else base
    df["Display Title"] = df.apply(display_title, axis=1)

    # Prefer variant image, else product image; never return literal "nan"
    def pick_image(r):
        vi = (r.get("Variant Image", "") or "").strip()
        if vi:
            return vi
        base_img = (r.get("Image Src", "") or "").strip()
        return base_img
    df["Image URL"] = df.apply(pick_image, axis=1)

    # Deduplicate to one row per variant (if applicable)
    if (df["Variant ID"] != "").any():
        key_cols = ["Variant ID"]
    elif (df["Variant SKU"] != "").any():
        key_cols = ["Variant SKU"]
    else:
        key_cols = ["Title", "Option1 Value", "Option2 Value", "Option3 Value"]

    df = (
        df.sort_values(by=["Variant Image", "Image Src"], ascending=False, na_position="last")
          .groupby(key_cols, as_index=False)
          .agg({
              "Title": "first",
              "Body (HTML)": "first",
              "Variant Price": "first",
              "Image URL": "first",
              "Clean Description": "first",
              "Price Display": "first",
              "Variant Label": "first",
              "Display Title": "first",
              "Option1 Name": "first", "Option1 Value": "first",
              "Option2 Name": "first", "Option2 Value": "first",
              "Option3 Name": "first", "Option3 Value": "first",
              "Variant SKU": "first", "Variant ID": "first",
          })
    )

    # LLM snippet per-variant (include variant label in description to help the model)
    def make_snip(r):
        desc = str(r["Clean Description"] or "")
        if r["Variant Label"]:
            desc = f"{desc} (Variant: {r['Variant Label']})"
        return {
            "title": str(r["Display Title"]).strip(),
            "description": desc[:MAX_DESC_CHARS],
            "price": r["Price Display"],
            "image_url": (r["Image URL"] or "").strip(),
        }

    df["LLM Snippet"] = df.apply(make_snip, axis=1)

    # Fallback search text should include variant terms
    df["Search Text"] = (df["Clean Description"].fillna("") + " " + df["Variant Label"].fillna("")).str.strip()

    return df


# --- OpenAI v1 client (no legacy branch) ---

def llm_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in a .env file or environment variable.")
    return OpenAI(api_key=api_key)

def choose_with_llm(user_style: str, products: list, model: str = MODEL_NAME) -> list:
    """
    Ask the LLM to pick exactly 2 items that match the user style.
    Returns a list of 2 dicts: {title, description, price, image_url, why}
    """
    catalog_json = json.dumps(products, ensure_ascii=False)

    system = (
        "You are a concise fashion assistant. Always respond in valid JSON."
        " Choose exactly 2 items that best match the user's personal style."
        " Each item must include: title, price, image_url, and a one-sentence 'why' "
        " explaining how it complements the user's style. Keep 'why' under 30 words."
    )
    user = (
        f"The user's personal style: {user_style}\n\n"
        f"Here is the product catalog as a JSON array. Each element has keys: "
        f"['title','description','price','image_url']:\n{catalog_json}\n\n"
        "Return JSON only in this schema:\n"
        '{ "items": [ {"title": str, "price": str, "image_url": str, "why": str},'
        '           {"title": str, "price": str, "image_url": str, "why": str} ] }'
    )

    client = llm_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.5,
    )

    # make content always a string to satisfy Pylance and json.loads
    content: str = resp.choices[0].message.content or ""

    # Parse JSON safely
    try:
        data = json.loads(content)
        items = data.get("items", [])
        if not (isinstance(items, list) and len(items) == 2):
            raise ValueError("Model did not return exactly 2 items")
        normalized = []
        for it in items:
            normalized.append({
                "title": str(it.get("title", "")).strip(),
                "price": str(it.get("price", "")).strip(),
                "image_url": str(it.get("image_url", "")).strip(),
                "why": str(it.get("why", "")).strip(),
            })
        return normalized
    except Exception:
        # Fallback: empty => let caller use heuristic
        return []

def run_picker(user_style: str, df: pd.DataFrame):
    try:
        # 0) Blank input: clear outputs
        if not user_style or not user_style.strip():
            return None, None, None, None, None, None

        # 1) Require base fields; allow either image column
        required_base = ["LLM Snippet", "Clean Description", "Title", "Price Display"]
        missing = [c for c in required_base if c not in df.columns]
        if missing:
            msg = f"Your CSV is missing required columns: {', '.join(missing)}"
            print("DEBUG:", msg)
            placeholder = ("", "N/A — ", msg)
            return *placeholder, *placeholder

        if not (("Image URL" in df.columns) or ("Image Src" in df.columns)):
            msg = "Your CSV needs an image column: add 'Image URL' or 'Image Src'."
            print("DEBUG:", msg)
            placeholder = ("", "N/A — ", msg)
            return *placeholder, *placeholder

        if df.empty:
            msg = "Your product CSV has no rows."
            print("DEBUG:", msg)
            placeholder = ("", "N/A — ", msg)
            return *placeholder, *placeholder

        # Debug snapshot
        try:
            print(f"DEBUG: df rows={len(df)}; sample row={df.iloc[0].to_dict()}")
        except Exception:
            print(f"DEBUG: df rows={len(df)}; sample row=<unavailable>")

        # 2) Prepare LLM input: keep dicts (don't stringify)
        products_col = df["LLM Snippet"].tolist()
        products = [p if isinstance(p, dict) else {} for p in products_col]

        used_fallback = False
        picks = None

        # 3) Try LLM path (safe to fail)
        try:
            picks = choose_with_llm(user_style, products)
            print(f"DEBUG: choose_with_llm returned {len(picks) if picks else 0} items")
        except Exception:
            import traceback
            print("choose_with_llm failed:\n", traceback.format_exc())
            picks = None

        # 4) Fallback keyword match (variant-aware if fields exist)
        if not picks:
            used_fallback = True
            print("DEBUG: using keyword fallback")
            lower = user_style.lower()
            df = df.copy()

            text_col = "Search Text" if "Search Text" in df.columns else "Clean Description"
            df["__score"] = df[text_col].fillna("").astype(str).str.lower().apply(
                lambda t: sum(1 for w in lower.split() if w in t)
            )

            top2 = df.sort_values("__score", ascending=False).head(2)
            picks = []
            for _, r in top2.iterrows():
                title_val = r.get("Display Title") or r.get("Title", "N/A")
                image_val = (r.get("Image URL") or r.get("Image Src") or "").strip()
                picks.append({
                    "title": title_val,
                    "price": r.get("Price Display", ""),
                    "image_url": image_val,
                    "why": "Selected by keyword match fallback based on your style description.",
                    # optional variant metadata
                    "variant_label": r.get("Variant Label", ""),
                    "sku": r.get("Variant SKU", ""),
                    "variant_id": r.get("Variant ID", ""),
                })

        # 5) Ensure two results
        while len(picks) < 2:
            picks.append({
                "title": "N/A",
                "price": "",
                "image_url": "",
                "why": "No item found.",
                "variant_label": "",
                "sku": "",
                "variant_id": "",
            })

        # 6) Format outputs (HTML img + Markdown info + plain why)
        def to_outputs(item: dict):
            title = item.get("title", "N/A")
            price = item.get("price", "")

            # Clean/sanitize image URL
            url = (item.get("image_url") or "").strip()
            if url.lower() in ("nan", "none", "null"):
                url = ""
            if not url:
                print("DEBUG: empty image_url for item:", item)

            img_html = (
                f'<img src="{url}" alt="{title}" style="max-width:100%;border-radius:12px;" />'
                if url else ""
            )

            # Build info Markdown with optional variant metadata
            bits = []
            vl = (item.get("variant_label") or "").strip()
            if vl:
                bits.append(f"*Variant:* {vl}")
            sku = (item.get("sku") or "").strip()
            if sku:
                bits.append(f"*SKU:* {sku}")
            vid = (item.get("variant_id") or "").strip()
            if vid:
                bits.append(f"*ID:* {vid}")

            info_md = f"{title} — {price}"
            if bits:
                info_md += "\n\n" + "\n".join(bits)

            why = item.get("why", "")
            if used_fallback:
                why = (why + "\n\n_Fallback ranking used (LLM unavailable or quota exceeded)._").strip()

            return img_html, info_md, why

        o1_img, o1_info, o1_why = to_outputs(picks[0])
        o2_img, o2_info, o2_why = to_outputs(picks[1])
        return o1_img, o1_info, o1_why, o2_img, o2_info, o2_why

    except Exception:
        import traceback
        err = traceback.format_exc()
        print("run_picker failed:\n", err)
        msg = "Something went wrong. Check the VS Code terminal for the traceback."
        placeholder = ("", "N/A — ", msg)
        return *placeholder, *placeholder

def build_app():
    load_dotenv()
    df = load_and_clean(CSV_PATH)
    with gr.Blocks(title="AI Style Picker") as demo:
        gr.Markdown("## 👗 AI Style Picker\nDescribe your personal style and get two matching items.\n")
        with gr.Row():
            style = gr.Textbox(label="Describe your personal style",
                               placeholder="e.g., minimalist, neutral colors, streetwear influences, relaxed fit")
        submit = gr.Button("Find my match")
        with gr.Row():
            img1 = gr.HTML(label="Item 1")
            info1 = gr.Markdown()
            why1 = gr.Markdown()
        with gr.Row():
            img2 = gr.HTML(label="Item 2")
            info2 = gr.Markdown()
            why2 = gr.Markdown()
        def on_click(user_style):
            return run_picker(user_style, df)
        submit.click(on_click, inputs=[style], outputs=[img1, info1, why1, img2, info2, why2])
    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.queue()
    demo.launch()
    