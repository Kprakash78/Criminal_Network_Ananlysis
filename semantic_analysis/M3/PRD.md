# M5 — LLM / RAG — PRD

## 1. What M5 owns
Two jobs, in sequence:
1. **Query understanding**: turn a natural-language user query into a structured
   representation (objects, actions, attributes, speech-intent, context) that M4 can
   actually search against.
2. **RAG answering**: take M4's top-K retrieved segments as evidence, and produce a
   grounded, timestamp-aware natural-language answer with an explainability breakdown
   ("94% match — motorcycle detected, person detected, transcript contains repair
   instructions, segment 00:21–00:43").

M5 never invents facts not present in the retrieved evidence — every claim in the
answer must trace back to a field in the retrieved segment record.

## 2. Why this exists in the architecture
Plain vector search returns "similar" segments but can't explain *why*, can't combine
multiple weak signals into one confident answer, and can't handle queries with real
compositional structure ("X while Y", "person in red vs blue"). M5 is the layer that
(a) makes the search precise by decomposing the query before it even hits FAISS, and
(b) makes the result trustworthy by grounding the final answer in cited evidence
instead of letting the LLM free-associate from the video title.

## 3. MVP scope vs optional
**MVP:**
- Structured query parsing: LLM call with a fixed Pydantic schema (objects, actions,
  persons, speech_intent, context) — see §4 for schema, §5 for prompt strategy.
- Pass structured query + raw query to M4; receive top-K candidate segments back.
- Grounded answer generation: LLM call that receives the retrieved segments (formatted
  per §7) and produces: which video/segment is the best match, a natural-language
  answer, and the explainability bullet list, using only what's in the retrieved
  evidence.
- Failure handling: if M4 returns nothing above a relevance threshold, say so plainly
  instead of hallucinating a match.

**Optional / only if time remains, in priority order:**
1. LLM-based reranking of M4's top-K before generation (only worth it if M4's raw
   ranking is visibly noisy in testing — don't add complexity pre-emptively).
2. Query rewriting for ambiguous/short queries (e.g. "motorcycle guy" → expand before
   parsing).
3. Attribute-binding-aware scoring: when the structured query has attributes bound to
   specific objects (e.g. `{"object": "motorcycle", "attributes": ["blue"]}` vs a
   separate `{"object": "person", "attributes": ["red"]}`), verify the retrieved
   segment's `objects[].attributes` actually match the *same* object before counting it
   as a strong match, rather than just checking "red" and "blue" both appear somewhere
   in the segment. This is the concrete fix for CLIP/embedding-level bag-of-words
   confusion — see §8. Worth doing if M3 supplies attributes; skip if M3 doesn't reach
   its optional attribute-extraction tier.

## 4. Structured output schema (Pydantic)
```python
from pydantic import BaseModel
from typing import Optional

class AttributedObject(BaseModel):
    object: str
    attributes: list[str] = []

class StructuredQuery(BaseModel):
    objects: list[AttributedObject] = []
    actions: list[str] = []
    persons: list[str] = []
    speech_intent: list[str] = []
    context: list[str] = []
    raw_query: str
```
Note this upgrades the PS's original flat `"objects": ["motorcycle"]` example to
attribute-bound objects — small change, but it's what makes "person in red fixing a
blue motorcycle" resolvable at all downstream. If time is extremely short, the flat
version from the problem statement is an acceptable MVP fallback; add attributes as the
first optional upgrade.

## 5. Prompt engineering strategy
- Use the LLM's native structured-output / tool-use mode (function calling with the
  Pydantic schema as the tool signature) rather than asking for free-text JSON and
  regex-parsing it. This eliminates most parsing failures outright.
- System prompt should explicitly instruct: "extract only what's stated or clearly
  implied, do not invent objects/actions not suggested by the query, leave lists empty
  rather than guessing."
- One-shot or few-shot examples in the prompt (2–3 examples covering: object+action
  query, attribute-binding query, speech-intent query) meaningfully improve reliability
  over zero-shot — cheap to add, worth doing.

## 6. RAG architecture
```
user query
  -> structured query parsing (LLM call #1, function-calling)
  -> sent to M4 (structured query + raw query)
  <- top-K segments (with per-segment scores) from M4
  -> [optional: LLM rerank]
  -> context formatting (see §7)
  -> answer generation (LLM call #2, evidence-grounded)
  -> final answer + explanation + timestamps
```
Two LLM calls total for the MVP path (parse, then generate). Do not add a third
"critic"/self-correction call under this deadline unless call #2 is visibly
hallucinating in testing.

## 7. Context formatting for the generation call
Feed the LLM a compact, structured representation of each retrieved segment — not raw
JSON dumped verbatim (wastes tokens, harder for the model to reason over). Example
per-segment context block:
```
Segment: video_017, 00:21–00:43
Visual: person repairing motorcycle
Transcript: "First remove the side panel, then loosen the bolt..."
Objects detected: person (0.96), motorcycle (0.91)
OCR: none
Scores: visual=0.91, transcript=0.93, object_match=1.0, final=0.92
```
Cap to top 3–5 segments in context — more than that both wastes tokens and dilutes the
model's ability to pick the actual best match. Token budget: at this scale (few
sentences per segment, few segments), staying under a few thousand tokens for context
is trivial — don't over-engineer truncation logic for a hackathon corpus.

## 8. Hallucination prevention / evidence-grounded answers
- System prompt for the generation call: "Only state facts present in the segments
  below. If evidence is weak or absent, say so. Every claim must reference a specific
  segment's data (object detected, transcript content, OCR text, or score)."
- The explainability bullet list is not just UI decoration — it's a forcing function:
  requiring the model to cite which signals fired (object detected / transcript match /
  OCR match) makes fabrication much harder than free-text answering.
- Attribute-binding check (§3 optional item): if implemented, this is also a
  hallucination guard — it stops M5 from confidently claiming "person in red fixing
  blue motorcycle" when the retrieved segment actually has a red motorcycle and a
  differently-colored person, which is exactly the failure mode CLIP-style embeddings
  are documented to produce.

## 9. Failure handling when retrieval finds nothing
If M4's top result is below a relevance threshold (tune during testing, start ~0.5 on
whatever combined score M4 produces), respond: "No strong match found for this query,"
optionally with the best-available weak match shown separately and clearly labeled as
low-confidence. Never silently present a weak match as if it were strong.

## 10. Success criteria for the demo
- 5+ hand-picked demo queries covering: simple object query, action+speech-intent
  query ("explaining while repairing"), and one attribute-binding query, all return
  correct top match with correct timestamp range.
- The explainability bullets visibly correspond to real evidence, checkable live by a
  judge who watches the cited timestamp.
- At least one deliberately-unanswerable query demoed to show the "no strong match"
  path works (avoids looking like the system always confidently says something).

## 11. Integration contract (what M5 promises/expects)
- Input from user: raw natural-language query string (via M6's API).
- Output to M4: `StructuredQuery` JSON (§4) + raw query string.
- Input from M4: list of segment records with `scores` populated, matching
  `COMMON_DATA_CONTRACT.md`.
- Output to M6/frontend: final answer text, cited segment(s) with video_id + timestamp
  range, explainability bullet list, confidence indicator.
