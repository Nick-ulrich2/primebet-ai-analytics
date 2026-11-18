# Primebet AI Analytics

Projet DME Systems : segmentation joueurs (K-Means / churn) + recommandations marketing par IA, présentés via un dashboard Streamlit.

## Structure
- `data/raw/` : datasets Kaggle (non suivis par Git)
- `data/processed/` : données Primebet collectées + dataset unifié
- `src/collector/` : scraping public + simulateur calibré
- `src/ml_engine/` : segmentation / churn
- `src/ai_strategist/` : recommandations marketing
- `app.py` : dashboard Streamlit

## Démarrage
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Mettre la clé Gemini dans `.env` : `GEMINI_API_KEY=...` (jamais commitée).
