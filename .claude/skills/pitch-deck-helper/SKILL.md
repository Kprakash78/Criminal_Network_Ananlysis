---
name: pitch-deck-helper
description: Use when asked to draft, structure, or review hackathon pitch/demo content for this project — slide outlines, the one-line pitch, or judge Q&A prep. Not for the code itself.
---

# Pitch Deck Helper

This project gets judged on a live demo plus a short pitch. Keep all pitch content
consistent with what's actually built — never pitch a feature that isn't working yet.

## The core pitch (memorize this, keep it to one line)

"The hash can't be faked or quietly edited after the fact — anyone can verify a credential
against a public ledger without trusting us."

## Slide structure (keep to 5–6 slides for a hackathon demo)

1. **Problem** — credentials are easy to forge, hard to verify without calling the issuer.
2. **Solution** — one sentence + a simple diagram (issuer → hash → chain, verifier → hash →
   check).
3. **Live demo** — issue a real credential, then verify a tampered one live. This slide is
   really "stop talking, switch to the browser."
4. **How it works** — the DESIGN.md diagram, simplified. No Solidity code on screen.
5. **What's next** — revoke, issuer dashboard, mention only if actually built or clearly
   marked as future work.

## When reviewing pitch content

- Cut jargon a non-technical judge won't parse ("bytes32", "gas", "mapping") — say "a
  unique fingerprint of the file" instead of "a SHA-256 hash," unless the judge asks for
  technical depth.
- Every claim on a slide should be demoable. If it's not built yet, label it "planned,"
  don't imply it works.
- Anticipate the obvious judge question — "why not just use a database?" — and have the
  one-sentence answer ready: a database requires trusting whoever runs it; a public chain
  doesn't.
