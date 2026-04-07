# Data Collection — Research Evolver

## Sources

### ChatGPT Export (`chatgpt_data.txt`)
- **Source:** `~/Downloads/chatgptexport/` — full ChatGPT conversation export (97MB zip, 4 shards)
- **Coverage:** 392 conversations across ~March 2024 onwards
- **Extracted:** 20,113 unique turns → 6.7 MB
- **Content:** Creative writing, songs, tech Q&A, reasoning, casual conversation, code help

### Marble Conversations (`marble_data.txt`)
- **Source:** OpenClaw workspace memory files + session transcripts
  - `memory/2026-04-*.md` — daily session notes
  - `memory/2026-04-04-starfire-math-voice-kg.md` — raw Star session transcripts
  - `memory/2026-04-04-whatsapp-*.md` — WhatsApp session logs
  - `SOUL.md`, `USER.md`, `IDENTITY.md` — personality + context
- **Extracted:** 934 turns → 89 KB
- **Content:** Zach ↔ Marble conversations about Star, Starfire, coding, AI research

### Nova Export (`nova_data.txt`)
- **Source:** `~/Downloads/chat-Nova-1774216688296.md`
- **Status:** Mostly OpenClaw config/tool noise, not useful conversation data
- **Not used in combined dataset**

## Combined Dataset (`research_evolver_data.txt`)
- **Total:** 21,046 lines, 6.89 MB
- **Deduplication:** Per-line, first 120 chars as key
- **Format:** `user: ...` / `assistant: ...` alternating turns
- **Cleaned:** Markdown stripped, URLs removed, whitespace collapsed

## Processing Scripts
- `extract_chatgpt.py` — parses ChatGPT export shards, extracts message turns
- `extract_nova.py` — parses Nova markdown export
- `collect_marble_data.py` — extracts from workspace memory files
- `merge_data.py` — merges sources, deduplicates, writes combined dataset

## Notes for Training
- Very diverse: creative, technical, casual, reasoning
- Includes profanity and raw language (ChatGPT creative sessions)
- CircuitLM tokenizer vocab is small (64-348 tokens) — will heavily compress this data
- Consider filtering to target vocab range after tokenization
