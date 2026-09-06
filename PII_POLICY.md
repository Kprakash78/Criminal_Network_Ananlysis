# PII & Data Handling Policy

## ⚠️ CRITICAL: Do NOT Use Real Data in This Repository

This repository contains **synthetic (fabricated) data only**. All FIRs, CDRs,
transaction records, person names, phone numbers, and account numbers are
artificially generated for development and demonstration purposes.

## Rules

1. **NEVER commit real FIRs** (First Information Reports) to this repository.
2. **NEVER commit real CDRs** (Call Detail Records) to this repository.
3. **NEVER commit real financial transaction data** to this repository.
4. **NEVER commit real PII** (names, phone numbers, Aadhaar numbers, addresses)
   of any actual person to this repository.

## For Deployments

When deploying this system with real case data:

- Store all case data in a **separate, access-controlled database** — never in Git
- Use **encryption at rest** for all stored PII
- Implement **role-based access control** (RBAC) for data access
- Enable **audit logging** for all data access events
- Ensure compliance with **IT Act 2000** and relevant data protection regulations
- Follow **chain of custody** procedures for evidence handling

## Handling Accidental Commits

If real data is accidentally committed:

1. **Immediately** notify the team lead
2. Use `git filter-branch` or `git filter-repo` to remove the data from history
3. Force-push the cleaned history
4. Rotate any exposed credentials or identifiers
5. Document the incident in the security log

## Contact

For questions about data handling, contact the project security lead.
