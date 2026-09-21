"""
WasteWise AI - Web Frontend Application (Streamlit)

Identify. Understand. Dispose Responsibly.
An AI-powered waste identification and responsible disposal assistant aligned with UN SDG 12.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure project root is in Python sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import WasteWisePipeline

# Page Configuration
st.set_page_config(
    page_title="WasteWise AI — Identify. Understand. Dispose Responsibly.",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Modern, Premium Sustainability Aesthetic
st.markdown("""
<style>
    /* Global Styles & Color Palette */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Card Container Styling */
    .main-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
    }
    
    .header-title {
        color: #064e3b;
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 4px;
    }
    
    .header-tagline {
        color: #047857;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    .badge-confidence-high {
        background-color: #d1fae5;
        color: #065f46;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.875rem;
        display: inline-block;
    }
    
    .badge-confidence-low {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.875rem;
        display: inline-block;
    }

    .badge-demo {
        background-color: #e0f2fe;
        color: #0369a1;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        display: inline-block;
    }
    
    .source-box {
        background-color: #f1f5f9;
        border-left: 4px solid #0d9488;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.875rem;
        color: #334155;
        margin-top: 16px;
    }

    .warning-card {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .info-bullet {
        margin-bottom: 8px;
        line-height: 1.5;
    }

    /* Hide Streamlit default footer/header clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """
    Initializes and caches the WasteWise AI pipeline instance.
    """
    return WasteWisePipeline()


def main():
    # 1. Header & Branding Section
    st.markdown("""
    <div style="text-align: center; padding-top: 10px; padding-bottom: 20px;">
        <div class="header-title">♻️ WasteWise AI</div>
        <div class="header-tagline">Identify. Understand. Dispose Responsibly.</div>
        <p style="color: #64748b; max-width: 600px; margin: 0 auto; font-size: 0.95rem;">
            An AI-powered decision support assistant for waste segregation grounded in 
            <strong>India Solid Waste Management Rules</strong> and <strong>Plastic Waste Management Rules (MoEFCC / CPCB)</strong>.
            Aligned with <strong>UN SDG 12: Responsible Consumption and Production</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load Pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Error initializing backend pipeline: {e}")
        return

    # 2. Image Upload Card
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("1. Upload Waste Image")
    st.caption("Supported image formats: JPG, JPEG, PNG, WEBP")

    uploaded_file = st.file_uploader(
        "Choose an image file...",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of an everyday waste item."
    )

    if uploaded_file is not None:
        try:
            # Display Image Preview
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image Preview", use_column_width=True)

            # Analyze Button
            if st.button("🔍 Analyze Waste Item", type="primary", use_container_width=True):
                # 3. Loading State
                with st.spinner("Analyzing your waste... (Running ResNet18 inference & RAG grounding)"):
                    # Save temporarily to disk for pipeline processing
                    temp_dir = Path(tempfile.gettempdir()) / "wastewise_uploads"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = temp_dir / uploaded_file.name

                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Run Pipeline
                    result = pipeline.analyze_image(temp_path)

                    # Cleanup temp file
                    if temp_path.exists():
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass

                # 4. Display Analysis Results
                st.markdown("---")
                st.subheader("2. Waste Identification & Analysis")

                if result.get("status", "").startswith("error"):
                    st.error(result.get("user_message", "An unexpected pipeline error occurred."))
                
                # CASE A: Low Confidence Result (Responsible AI Guardrail)
                elif not result.get("is_reliable", False):
                    st.markdown(f"""
                    <div class="warning-card">
                        <h4 style="color: #b45309; margin-top: 0;">⚠️ Uncertainty Guardrail Active</h4>
                        <p style="color: #92400e; font-size: 1rem; font-weight: 600;">
                            We're not confident enough to classify this item (Confidence: {result.get('confidence_percentage', 0.0)}%).
                        </p>
                        <p style="color: #78350f; font-size: 0.9rem;">
                            Please upload a clearer image showing the item by itself in good lighting.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.info(
                        "🛡️ **Responsible AI Notice**: WasteWise AI intentionally suppresses disposal recommendations "
                        "when prediction confidence is below threshold to prevent improper waste segregation and environmental contamination."
                    )

                # CASE B: High Confidence Result
                else:
                    cat_name = result.get("display_name", "Unknown")
                    conf_pct = result.get("confidence_percentage", 0.0)
                    granite_mode = "Live IBM Granite" if result.get("is_live_service") else "Demo/Mock Mode"

                    # Metrics Row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Material", cat_name)
                    with col2:
                        st.metric("Confidence Score", f"{conf_pct}%")
                    with col3:
                        st.metric("Status", "HIGH")

                    st.markdown(f'<span class="badge-demo">AI Explanation: {granite_mode}</span>', unsafe_allow_html=True)

                    # Responsible Disposal Guidance Card
                    st.markdown("### 📋 Responsible Disposal Guidance")

                    # Display LLM/Granite Guidance
                    guidance_text = result.get("granite_response", "")
                    if guidance_text:
                        st.markdown(guidance_text)

                    # Detailed Facts Breakdown from RAG Knowledge Base
                    kb_entry = result.get("retrieved_knowledge", {})
                    if kb_entry:
                        guidance_data = kb_entry.get("guidance", {})
                        meta_data = kb_entry.get("verification_metadata", {})

                        with st.expander("🔍 View Detailed Recycling & Preparation Steps"):
                            st.markdown(f"**General Disposal Type**: `{kb_entry.get('general_disposal_category')}`")
                            
                            if guidance_data.get("do_s"):
                                st.markdown("**Recommended Practices (Do's)**:")
                                for do in guidance_data.get("do_s", []):
                                    st.markdown(f"- ✅ {do}")
                            
                            if guidance_data.get("dont_s"):
                                st.markdown("**Things to Avoid (Don'ts)**:")
                                for dont in guidance_data.get("dont_s", []):
                                    st.markdown(f"- ❌ {dont}")

                            if guidance_data.get("preparation_steps"):
                                st.markdown("**Preparation Steps**:")
                                for step in guidance_data.get("preparation_steps", []):
                                    st.markdown(f"1. {step}")

                            if guidance_data.get("environmental_impact_note"):
                                st.success(f"🌱 **Why It Matters**: {guidance_data.get('environmental_impact_note')}")

                        # Source Transparency Box
                        source_name = meta_data.get("source_name", "MoEFCC & CPCB Waste Management Rules")
                        source_url = meta_data.get("source_url", "https://cpcb.nic.in/")

                        st.markdown(f"""
                        <div class="source-box">
                            <strong>📌 Knowledge Source:</strong> {source_name}<br/>
                            <a href="{source_url}" target="_blank" style="color: #0f766e; text-decoration: underline;">Verify Source Reference ({source_url})</a><br/>
                            <em style="color: #64748b; font-size: 0.8rem;">
                                Note: Waste-management practices, bin color-coding, and scrap collection pathways may vary by local Urban Local Body (ULB) / Municipality.
                            </em>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while reading the image: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # 5. Responsible AI Guardrails & Project Transparency Section
    with st.expander("🛡️ Why WasteWise AI is Careful (Responsible AI Principles)"):
        st.markdown("""
        WasteWise AI adheres to strict **Responsible AI & Sustainability** principles:

        - **Transparent Confidence**: Displays explicit prediction confidence percentages rather than hiding model uncertainty.
        - **Uncertainty Gate**: Automatically suppresses disposal recommendations when confidence is low to avoid giving wrong advice.
        - **Grounded Knowledge (RAG)**: Disposal instructions are retrieved strictly from verified Indian regulatory frameworks (**MoEFCC & CPCB Rules**), preventing AI hallucinations.
        - **Local-Rule Awareness**: Acknowledges that final disposal pathways depend on local Urban Local Body (ULB) / Municipal guidelines.
        - **Privacy Protection**: Images are processed locally during session inference; no user images or personal data are stored.
        - **Decision Support**: Built as an awareness and decision-support tool, not a replacement for municipal authorities.
        """)


if __name__ == "__main__":
    main()
