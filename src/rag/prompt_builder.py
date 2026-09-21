"""
WasteWise AI - RAG Context & Prompt Builder Module

Combines vision model predictions, confidence evaluation results, and retrieved
knowledge base information into a structured prompt context for IBM Granite LLM.
Enforces Responsible AI rules by handling low-confidence uncertainty gracefully.
"""

from typing import Dict, Any, Optional
from src.utils.confidence import evaluate_confidence
from src.rag.retriever import fetch_guidance

SYSTEM_INSTRUCTION = """You are WasteWise AI, an empathetic, responsible, and educational sustainability assistant.
Your goal is to explain waste classification results and provide clear disposal guidance.

STRICT RESPONSIBLE AI RULES:
1. Ground your answer ONLY in the provided Knowledge Base facts. Do NOT invent disposal rules or extra steps.
2. Emphasize that material identification is decision-support, not universal municipal law (local rules vary).
3. Explain the environmental impact to encourage responsible user behavior.
4. Keep the output friendly, structured, and easy for students and households to understand.
"""


def build_rag_prompt(
    predicted_category: str,
    confidence_score: float,
    confidence_result: Optional[Dict[str, Any]] = None,
    knowledge_entry: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Constructs a structured prompt & response package combining prediction, confidence, and RAG facts.

    Args:
        predicted_category: Category ID (e.g. 'plastic', 'glass', 'food_organics')
        confidence_score: Model confidence score (0.0 to 1.0)
        confidence_result: Optional pre-computed output from evaluate_confidence()
        knowledge_entry: Optional pre-fetched category data from retriever
        threshold: Optional confidence threshold float override

    Returns:
        Structured dictionary containing status, user messages, retrieved facts, and formatted LLM prompt.
    """
    # 1. Evaluate confidence if not pre-computed
    if confidence_result is None:
        kwargs = {} if threshold is None else {"threshold": threshold}
        confidence_result = evaluate_confidence(confidence_score, **kwargs)

    # 2. Retrieve knowledge entry if not pre-fetched
    if knowledge_entry is None and isinstance(predicted_category, str):
        knowledge_entry = fetch_guidance(predicted_category)

    # 3. Handle Unknown Category / Missing Knowledge
    if confidence_result.get("status") == "invalid_input" or not knowledge_entry:
        return {
            "status": "unknown_category",
            "is_reliable": False,
            "category": predicted_category,
            "confidence_score": confidence_score,
            "confidence_details": confidence_result,
            "retrieved_facts": None,
            "user_message": (
                f"We could not find verified disposal guidance for '{predicted_category}'. "
                "Please consult your local municipal waste management authority."
            ),
            "system_instruction": SYSTEM_INSTRUCTION,
            "llm_prompt": None
        }

    # 4. Handle Low Confidence / Uncertainty (Responsible AI Guardrail)
    if not confidence_result.get("is_reliable", False):
        return {
            "status": "low_confidence_uncertainty",
            "is_reliable": False,
            "category": predicted_category,
            "confidence_score": confidence_score,
            "confidence_details": confidence_result,
            "retrieved_facts": knowledge_entry,
            "user_message": (
                f"The AI is uncertain about this item (confidence: {confidence_score*100:.1f}%). "
                "To prevent improper disposal, please capture a clearer photo in better light or verify the item manually."
            ),
            "system_instruction": SYSTEM_INSTRUCTION,
            "llm_prompt": None  # Do NOT generate confident LLM advice when model is uncertain
        }

    # 5. Handle High Confidence -> Construct IBM Granite Prompt
    guidance = knowledge_entry.get("guidance", {})
    metadata = knowledge_entry.get("verification_metadata", {})

    prompt_text = f"""[SYSTEM INSTRUCTION]
{SYSTEM_INSTRUCTION}

[MODEL PREDICTION]
Identified Material: {knowledge_entry.get('display_name', predicted_category)}
Prediction Confidence: {confidence_score*100:.1f}%
General Disposal Type: {knowledge_entry.get('general_disposal_category', 'N/A')}

[RETRIEVED FACTUAL KNOWLEDGE BASE]
Summary: {guidance.get('summary', '')}
Do's: {', '.join(guidance.get('do_s', []))}
Don'ts: {', '.join(guidance.get('dont_s', []))}
Preparation Steps: {', '.join(guidance.get('preparation_steps', []))}
Environmental Impact: {guidance.get('environmental_impact_note', '')}
Verification Source: {metadata.get('source_name', 'Verified Recycling Guidelines')}

[USER PROMPT / TASK]
Using ONLY the facts provided above, generate a brief, friendly, and structured 3-point recommendation for disposing of this {knowledge_entry.get('display_name')} item.
"""

    return {
        "status": "success",
        "is_reliable": True,
        "category": predicted_category,
        "display_name": knowledge_entry.get("display_name"),
        "confidence_score": confidence_score,
        "confidence_details": confidence_result,
        "retrieved_facts": knowledge_entry,
        "system_instruction": SYSTEM_INSTRUCTION,
        "llm_prompt": prompt_text,
        "user_message": f"Identified as {knowledge_entry.get('display_name')} with {confidence_score*100:.1f}% confidence."
    }
