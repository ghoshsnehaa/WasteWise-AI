# WasteWise AI ♻️

An AI-Powered Waste Identification and Responsible Disposal Assistant aligned with UN SDG 12: Responsible Consumption and Production (Target 12.5).

## Overview
WasteWise AI helps users identify common waste materials using computer vision and provides accurate, grounded disposal guidance powered by Retrieval-Augmented Generation (RAG) and IBM Granite LLMs.

## Project Structure
- `data/`: Contains dataset metadata, split manifests, visual samples, and curated knowledge base. *(Note: The raw RealWaste image dataset is excluded from the repository via `.gitignore`).*
  - `data/knowledge_base/`: Structured recycling guidelines and verification sources for RAG.
  - `data/dataset_splits.json`: Stratified Train/Val/Test split manifest.
  - `data/class_to_idx.json`: Class mapping dictionary for 9 waste categories.
  - `data/dataset_samples.png`: Representative sample grid visualization.
- `models/`: Trained model weights checkpoint (`resnet18_best.pth`) and evaluation results.
- `src/`: Core Python modules (CV classifier, RAG retriever, LLM integration).
- `web/`: Frontend interface for image upload and guidance presentation.
- `tests/`: Automated test suite.

## RealWaste Categories Supported
1. Cardboard
2. Food Organics
3. Glass
4. Metal
5. Miscellaneous Trash
6. Paper
7. Plastic
8. Textile Trash
9. Vegetation

## Run the Web App

### Setup & Launch Instructions
1. Create and activate a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open the local Streamlit URL displayed in your terminal (typically `http://localhost:8501`).

### System Features & Responsible AI Guardrails
* **Computer Vision Model**: Uses a trained ResNet18 transfer learning classifier (75.46% test accuracy across 9 RealWaste categories).
* **Grounded Knowledge Base**: Waste disposal and segregation guidance is retrieved directly from a verified knowledge base aligned with **MoEFCC SWM Rules 2026** and **CPCB Plastic Waste Rules**.
* **Uncertainty Gate**: Low-confidence predictions (< 0.70 threshold) intentionally suppress disposal recommendations to prevent improper waste segregation.
* **IBM Granite LLM Integration**: Operates in a safe mock mode by default unless live `watsonx.ai` API credentials are set in environment variables (`WATSONX_APIKEY` and `WATSONX_PROJECT_ID`).
* **Local Municipal Priority**: Users are explicitly reminded that local Urban Local Body (ULB) and municipal authority rules take precedence.

