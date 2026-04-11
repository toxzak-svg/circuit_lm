# Data Collection — Research Evolver

## Sources

### Marble Conversations v2 (`marble_conversations.txt`)
- **Source:** OpenClaw workspace memory files + session transcripts
  - `memory/2026-04-*.md` — all daily session notes (2026-04-02 through 2026-04-11)
  - `projects/star/data/personal_chats/` — Star personal chat exports
  - SOUL.md, USER.md, IDENTITY.md, AGENTS.md (personality context)
- **Extracted:** 263 turns (37 user, 226 assistant) → 24.3 KB
- **Content:** Zach ↔ Marble conversations about Star, Starfire, coding, AI research, CrumbStore, math engine, WhatsApp integration, Kaggle experiments
- **Format:** `user: ...` / `assistant: ...` alternating turns
- **Notably includes:** Late-night conversations about AI consciousness, emergence, architecture theory, and Starfire build decisions

### Existing marble_data.txt (91 KB)
- Previous extraction via collect_marble_data.py script
- Overlaps significantly with marble_conversations.txt
- Consider merging/deduplicating in future run

### ChatGPT Export (`chatgpt_data.txt`)
- **Source:** `~/Downloads/chatgptexport/` — full ChatGPT conversation export (97MB zip, 4 shards)
- **Coverage:** 392 conversations across ~March 2024 onwards
- **Extracted:** 20,113 unique turns → 6.7 MB
- **Content:** Creative writing, songs, tech Q&A, reasoning, casual conversation, code help

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
- `do_extract.py` — extracts from memory/*.md using timestamp + user/assistant patterns

- `extract_marble_conversations.py` — (backup extraction script, see do_extract.py)

## Notes for Training
- Very diverse: creative, technical, casual, reasoning
- Includes profanity and raw language (ChatGPT creative sessions)
- CircuitLM tokenizer vocab is small (64-348 tokens) — will heavily compress this data
- Consider filtering to target vocab range after tokenization
