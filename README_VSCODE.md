# VS Code Wiring — AI Style Picker

## One-time setup
1) Put these files in your project folder:
   - `ai_style_picker.py`
   - `product_template.csv`
   - `requirements.txt`
   - `.vscode/launch.json`, `.vscode/tasks.json`, `.vscode/settings.json`
2) Copy `.env.example` to `.env` and paste your OpenAI key:
   ```
   OPENAI_API_KEY=sk-...your key...
   ```

## Create the environment & install deps
- In VS Code, press **Ctrl+Shift+P** → `Tasks: Run Task` → **Install requirements**  
  (This will create `.venv` and install everything.)

## Run / Debug
- Press **F5** (or choose the `Run AI Style Picker` launch config).
- Your Gradio app opens in the terminal with a local URL.

## Notes (Windows)
- You do **not** need Conda for this setup; it uses a local `.venv`.
- If Python isn’t found, install Python 3.10+ from python.org and reopen VS Code.
