# Secrets & API Keys — README

## No External API Keys Required

This system is designed to run **100% offline**. No external API keys,
tokens, or credentials are required for any functionality.

### Components That Do NOT Require API Keys

| Component | Details |
|-----------|---------|
| NER (M1) | Uses local spaCy models (`en_core_web_sm`) |
| Embeddings (M4/M6_feature) | Uses local `sentence-transformers/all-MiniLM-L6-v2` or TF-IDF fallback |
| LLM (M4) | Uses local HuggingFace models (flan-t5-large) or template-based fallback |
| Graph Analytics (M3) | Pure NetworkX — no external dependencies |
| Dashboard (M6) | Streamlit — runs locally |
| Evidence Signing | Ephemeral RSA keys generated at export time |

### If You Find Credentials in the Codebase

If you encounter any hardcoded API keys, tokens, passwords, or credentials
in this repository:

1. **Do NOT use them** — they may be test/expired values
2. **Report them** to the security lead immediately
3. **Remove them** from the codebase and commit history:
   ```bash
   # Remove from current code
   git rm <file-with-credentials>

   # Remove from history (if needed)
   git filter-repo --invert-paths --path <file-with-credentials>
   ```
4. **Rotate the credential** at its source (API dashboard, etc.)

### Environment Variables

The only environment variable used is:

```bash
CRIMINAL_USE_REAL_MODULES=0|1  # Toggle between mock and real backend modules
```

This is NOT a secret — it's a feature flag for development vs. production mode.
