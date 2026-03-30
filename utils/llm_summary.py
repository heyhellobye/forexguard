"""
LLM-generated risk summaries using Google Gemini.

Setup:
  pip install google-generativeai
  set GEMINI_API_KEY=your-key-here   (Windows CMD)

Get a free API key at: https://aistudio.google.com/app/apikey

Falls back to template summary if key is missing or API fails.
"""

import logging
import os
from typing import Optional

log = logging.getLogger("forexguard.llm")


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(alert: dict) -> str:
    uid      = alert.get("user_id", "unknown")
    score    = alert.get("ensemble_score", 0)
    severity = alert.get("severity", "UNKNOWN")
    flags    = alert.get("flags", [])
    features = alert.get("top_features", {})
    ms       = alert.get("model_scores", {})
    disagree = alert.get("ensemble_disagreement", 0)

    feat_lines  = "\n".join(
        f"  - {k}: {v:+.3f} (z-score vs population)"
        for k, v in features.items()
    )
    flag_lines  = "\n".join(f"  - {f}" for f in flags)
    model_lines = "\n".join(f"  - {k}: {v:.3f}" for k, v in ms.items())

    return f"""You are a senior AML (Anti-Money Laundering) compliance analyst at a forex brokerage.
You have received an automated anomaly detection alert for a trader account.
Write a concise, professional risk summary for the compliance team.

ALERT DATA
──────────
User ID        : {uid}
Ensemble Score : {score:.4f} (range 0-1, higher = more suspicious)
Severity       : {severity}
Model Agreement: {'LOW — models disagree, manual review recommended' if disagree > 0.25 else 'HIGH — models agree'}

Model Scores:
{model_lines}

Top Anomalous Feature Signals (z-scores vs full user population):
{feat_lines}

Triggered Compliance Flags:
{flag_lines}

INSTRUCTIONS
────────────
Write a 3-4 sentence risk narrative that:
1. Describes what suspicious behaviour was detected, in plain English
2. Explains why this pattern is concerning from a compliance/AML perspective
3. States what specific activity the compliance team should investigate
4. Suggests the most likely fraud/abuse category (e.g. structuring, account takeover, bonus abuse, wash trading, etc.)

Keep the tone professional and factual. Do not use bullet points. Write in flowing prose.
Do NOT repeat the raw numbers — translate them into human-readable observations.
"""


# ── Gemini caller ─────────────────────────────────────────────────────────────
def generate_llm_summary(alert: dict,
                          api_key: Optional[str] = None,
                          model: str = "gemini-1.5-flash",
                          max_tokens: int = 300) -> str:
    """
    Generate a Gemini risk summary for an alert dict.
    Returns the summary string, or falls back to the template summary.

    model options:
      gemini-1.5-flash  — fast, free tier available  (recommended)
      gemini-1.5-pro    — higher quality, lower rate limit
      gemini-2.0-flash  — latest, fastest
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        log.debug("[llm] GEMINI_API_KEY not set — using template summary")
        return alert.get("summary", "")

    try:
        import google.generativeai as genai

        genai.configure(api_key=key)

        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,       # low temp = more factual, consistent
            ),
        )

        prompt   = _build_prompt(alert)
        response = gemini_model.generate_content(prompt)
        summary  = response.text.strip()

        log.info(f"[llm] Gemini summary generated for {alert.get('user_id')} "
                 f"({len(summary)} chars)")
        return summary

    except ImportError:
        log.warning("[llm] google-generativeai not installed. "
                    "Run: pip install google-generativeai")
        return alert.get("summary", "")
    except Exception as e:
        log.warning(f"[llm] Gemini API call failed: {e} — using template summary")
        return alert.get("summary", "")


# ── Batch enrichment ──────────────────────────────────────────────────────────
def enrich_alerts_with_llm(alerts: list[dict],
                            api_key: Optional[str] = None,
                            only_severity: tuple = ("HIGH", "CRITICAL"),
                            model: str = "gemini-1.5-flash") -> list[dict]:
    """
    Enrich a list of alert dicts with Gemini-generated summaries.
    Only generates for HIGH/CRITICAL by default to save API calls.
    Returns the same list with 'llm_summary' added to each dict.
    """
    key = api_key or os.getenv("GEMINI_API_KEY", "")

    for alert in alerts:
        sev = alert.get("severity", "LOW")
        if sev in only_severity and key:
            alert["llm_summary"] = generate_llm_summary(
                alert, api_key=key, model=model)
        else:
            alert["llm_summary"] = alert.get("summary", "")

    return alerts
