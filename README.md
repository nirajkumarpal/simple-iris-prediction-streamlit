# Simple Iris Prediction — Streamlit

Small demo app to predict Iris species from four features using a RandomForest model and Streamlit.

Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Notes
- `app.py` will load `model.pkl` if present; if not it will train from `data/iris.csv` and save `model.pkl`.
- Consider adding `model.pkl` to `.gitignore` if you prefer training on deploy.
