---
name: smart-contract-helper
description: Use when writing, editing, testing, or debugging Solidity contracts in contracts/, or when setting up Hardhat config, writing deploy scripts, or debugging a failed transaction against the deployed contract.
---

# Smart Contract Helper

Conventions and checklist for this repo's Solidity work. Read `docs/DESIGN.md` first for
the contract's intended structure before editing.

## Before writing any contract code

1. Confirm the function signature matches what's in `docs/DESIGN.md`. If the request
   requires a signature change, update DESIGN.md in the same session — don't let it drift.
2. Check `contracts/hardhat.config.js` for the target network before writing a deploy
   script — this repo deploys to testnet only, never mainnet.

## Style rules

- One contract per file. File name matches contract name exactly.
- NatSpec (`/// @notice`, `/// @param`, `/// @return`) on every external/public function.
- Use `bytes32` for hashes, never `string`, for gas efficiency.
- Custom errors (`error NotIssuer();`) instead of `require(x, "string")` — cheaper and
  clearer in this Solidity version.
- Mark functions `view`/`pure` whenever they don't modify state — this is what makes
  `verifyCredential` free to call.

## Testing checklist (Hardhat + Chai)

For every new external function, write tests for:
- The happy path.
- The unauthorized-caller path (e.g. someone other than the issuer calling `revoke`).
- Double-issuing the same hash — decide and test the intended behavior (overwrite? revert?).

Run with `npx hardhat test` from `contracts/`. Don't consider a contract change done until
tests pass.

## Debugging a failed transaction

1. Check the revert reason first — Hardhat prints it; don't guess.
2. Confirm the wallet is on the correct network (chain ID matches `.env`).
3. Confirm the account has testnet gas — link the faucet in the README if it's empty.
4. If the hash doesn't match on verify, suspect file re-encoding before suspecting the
   contract — see "Known risks" in DESIGN.md.

## Deploying

- Deploy scripts live in `scripts/`, not `contracts/`.
- After every deploy, write the new contract address into `backend/.env` and
  `frontend/.env` — the demo breaks silently if these fall out of sync.
