# M5 — LLM / RAG — Backend / Implementation Guide

## 1. Concepts you need (in order)
1. **Function calling / structured output**: getting an LLM to return JSON matching a
   fixed schema reliably, instead of free text you regex out.
2. **RAG basics**: retrieve-then-generate. You're not building the retrieval (M4 does
   that) — you're the query-side and generation-side bookends.
3. **Prompt grounding**: constraining a generation call to only use provided context,
   not the model's general knowledge, to avoid hallucination.
4. That's genuinely most of it — you do not need LangChain/LlamaIndex abstractions for
   a two-call pipeline this small. Direct API calls are simpler to write, debug, and
   explain to judges.

## 2. Libraries / API
```bash
pip install anthropic     # or openai, pick one and standardize across the team
pip install pydantic
```
Use whichever LLM API the team has credits/keys for. Claude and GPT-4-class models both
support structured/function-calling output adequate for this. Don't spend time
evaluating multiple providers — pick one on day 1 and move on.

## 3. Query parsing call (structured output)
```python
from pydantic import BaseModel
from anthropic import Anthropic

client = Anthropic()

class AttributedObject(BaseModel):
    object: str
    attributes: list[str] = []

class StructuredQuery(BaseModel):
    objects: list[AttributedObject] = []
    actions: list[str] = []
    persons: list[str] = []
    speech_intent: list[str] = []
    context: list[str] = []

PARSE_SYSTEM_PROMPT = """You extract structured search intent from a natural-language
video search query. Extract only what is stated or clearly implied. Do not invent
objects, actions, or attributes that aren't suggested by the query text. If a category
has nothing to extract, leave its list empty.

Examples:
Query: "Find videos where someone is repairing a motorcycle while explaining the process."
-> objects: [{"object": "motorcycle"}], actions: ["repairing"], persons: ["person"],
   speech_intent: ["explaining process"], context: ["repair"]

Query: "person in red fixing a blue motorcycle"
-> objects: [{"object": "person", "attributes": ["red"]}, {"object": "motorcycle", "attributes": ["blue"]}],
   actions: ["fixing"], context: []
"""

def parse_query(user_query: str) -> StructuredQuery:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=PARSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_query}],
        tools=[{
            "name": "extract_query",
            "description": "Extract structured search intent",
            "input_schema": StructuredQuery.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "extract_query"},
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return StructuredQuery(**tool_use.input)
```
Forcing `tool_choice` to the extraction tool removes almost all of the "model answered
in prose instead of JSON" failure class.

## 4. Sending to M4 and receiving candidates
M5 calls into M4's retrieval function/endpoint with `(structured_query, raw_query)` and
gets back a list of segment dicts per `COMMON_DATA_CONTRACT.md` with `scores`
populated. Treat this as a plain function call or a local FastAPI internal call for the
MVP — no message queue, no async job system needed at this scale.

## 5. Context formatting + generation call
```python
def format_segment(seg: dict) -> str:
    ts = f"{int(seg['start_ts'])//60:02d}:{int(seg['start_ts'])%60:02d}"
    te = f"{int(seg['end_ts'])//60:02d}:{int(seg['end_ts'])%60:02d}"
    objs = ", ".join(f"{o['label']} ({o['confidence']})" for o in seg.get("objects", []))
    return (
        f"Segment: {seg['video_id']}, {ts}-{te}\n"
        f"Visual: {seg.get('visual', {}).get('description', 'n/a')}\n"
        f"Transcript: \"{seg.get('audio', {}).get('transcript', 'n/a')}\"\n"
        f"Objects detected: {objs or 'none'}\n"
        f"OCR: {', '.join(seg.get('ocr', [])) or 'none'}\n"
        f"Scores: {seg.get('scores', {})}\n"
    )

GENERATE_SYSTEM_PROMPT = """You answer video search questions using ONLY the evidence
segments provided below. Never state a fact not present in the evidence. If the
evidence is weak (say, all scores below 0.5) or absent, say plainly that no strong
match was found instead of guessing.

For your answer, provide:
1. The best-matching video and approximate timestamp range (mm:ss-mm:ss).
2. A one-sentence natural-language answer to the user's question.
3. An explainability bullet list citing which specific signals support the match
   (e.g. "Motorcycle detected", "Transcript contains repair instructions").
"""

def generate_answer(user_query: str, segments: list[dict]) -> str:
    context = "\n---\n".join(format_segment(s) for s in segments[:5])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=GENERATE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"User question: {user_query}\n\nEvidence:\n{context}",
        }],
    )
    return response.content[0].text
```

## 6. Token management
At hackathon corpus scale (top 3–5 segments, a couple sentences each), you will not hit
context limits. Don't build truncation/summarization logic pre-emptively — it's
wasted engineering time. If a transcript field is unusually long, truncate it to ~300
chars in `format_segment` and move on.

## 7. Is an LLM reranker worth it under this deadline?
Default answer: **no, not for MVP.** M4's combined score (visual + transcript + object
+ OCR match) is already a reasonable ranking signal. Add an LLM reranking pass only if,
during testing, you observe M4's top-1 result is visibly wrong more than a small
minority of the time on your demo query set. If you do add it: one extra LLM call that
receives the top-10 candidates and re-orders/filters to top-3, same grounding
constraints as §5.

## 8. Testing methodology
1. **Schema test**: assert `parse_query()` output is always a valid `StructuredQuery`
   across 10+ varied example queries (including edge cases: very short query, query
   with no clear object, attribute-binding query).
2. **Grounding test**: manually verify, for 5+ demo queries, that every sentence in the
   generated answer traces to an actual field in the retrieved segment — flag any
   generated claim that isn't supported.
3. **No-match test**: run a query with no plausible match in the corpus (e.g. asking
   about content that doesn't exist in any demo video) and confirm the "no strong
   match" path triggers instead of a hallucinated answer.
4. **End-to-end demo rehearsal**: run the exact 5 queries you plan to show judges,
   timed, at least once the day before the deadline — not for the first time live.

## 9. Example queries and expected structured output
| Query | Expected objects | Expected actions | Expected speech_intent |
|---|---|---|---|
| "Find videos where a chef is using an air fryer" | air fryer | using | [] |
| "person wearing a helmet working on a motorcycle" | person(attr: helmet), motorcycle | working on | [] |
| "someone repairing a motorcycle while explaining the process" | motorcycle | repairing | explaining process |
| "person in red fixing a blue motorcycle" | person(attr: red), motorcycle(attr: blue) | fixing | [] |

## 10. Common failure modes + debugging checklist
| Symptom | Likely cause | Fix |
|---|---|---|
| Parsing call returns prose, not JSON | `tool_choice` not forced | Force tool_choice as in §3 |
| Generated answer states something not in evidence | System prompt not strict enough, or too much context diluting it | Tighten grounding instruction; cap segments to top 3 |
| Always picks segment 1 regardless of query | Context formatting doesn't clearly differentiate segments, or scores not actually varying | Print `scores` per segment before generation call; verify M4 is returning meaningfully different scores |
| "No match" triggers too often | Relevance threshold set too high relative to M4's actual score distribution | Print score distribution across a test query set, recalibrate threshold empirically, don't guess a number |
| Attribute-binding queries silently wrong (wrong object gets the attribute) | M3 attributes not populated, or M5 not checking binding, just token overlap | Confirm M3's optional attribute tier shipped; if not, don't claim attribute-binding as a feature in the demo |

## 11. Integration contract recap
See PRD §11. M5 sits between the user-facing API (M6) and retrieval (M4); it never
touches FAISS directly and never fabricates evidence not returned by M4.
