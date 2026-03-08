# Layered Context Architecture

## Visual Overview

```
User Query: "How should I handle token expiration?"
                    ↓
        ┌───────────────────────────┐
        │   LayeredContext.compile()  │
        │      (Orchestrator)         │
        └───────────────────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    │    Gather from all tiers:     │
    └───────────────┬───────────────┘
                    ↓
    ╔═══════════════════════════════════════╗
    ║  TIER 1: Structured State             ║
    ║  ─────────────────────────────────    ║
    ║  Always included (if non-empty)       ║
    ║                                       ║
    ║  ## Symbols                           ║
    ║  - generateToken() in src/auth.js     ║
    ║  - verifyToken() in src/auth.js       ║
    ║                                       ║
    ║  ## Constraints                       ║
    ║  - Must support OAuth2                ║
    ║                                       ║
    ║  ## Decisions                         ║
    ║  - Use JWT: better security           ║
    ║                                       ║
    ║  ## Open Loops                        ║
    ║  - [pending] Token rotation           ║
    ║                                       ║
    ║  ## Test Failures                     ║
    ║  - test_expiry: Expected error        ║
    ║                                       ║
    ║  Cost: ~150 tokens (compact!)         ║
    ╚═══════════════════════════════════════╝
                    ↓
    ╔═══════════════════════════════════════╗
    ║  TIER 2: Episodic Memory              ║
    ║  ─────────────────────────────────    ║
    ║  Retrieved by relevance to query      ║
    ║                                       ║
    ║  Query: "token expiration"            ║
    ║         ↓                             ║
    ║  [Hybrid Search: Vector + BM25]       ║
    ║         ↓                             ║
    ║  Top 3 chunks:                        ║
    ║  1. [Score: 3.46] verifyToken code    ║
    ║  2. [Score: 3.20] JWT docs            ║
    ║  3. [Score: 1.32] refresh tokens      ║
    ║                                       ║
    ║  Cost: ~200 tokens (only relevant!)   ║
    ╚═══════════════════════════════════════╝
                    ↓
    ╔═══════════════════════════════════════╗
    ║  TIER 0: Hot Working Set              ║
    ║  ─────────────────────────────────    ║
    ║  Recent conversation (last N tokens)  ║
    ║                                       ║
    ║  user: "How do JWT tokens work?"      ║
    ║  assistant: "JWT consists of..."      ║
    ║  user: "How about refresh tokens?"    ║
    ║  assistant: "Refresh tokens..."       ║
    ║  user: "Handle expiration?"           ║
    ║                                       ║
    ║  Cost: ~1800 tokens (recent only!)    ║
    ╚═══════════════════════════════════════╝
                    ↓
    ┌───────────────────────────────┐
    │  Compile into single prompt:  │
    │  • Tier 1 state                │
    │  • Tier 2 retrieved chunks     │
    │  • Tier 0 recent messages      │
    │  • Current query               │
    │                                │
    │  Total: ~2150 tokens           │
    │  (vs 50K+ for full history!)   │
    └───────────────────────────────┘
                    ↓
              Send to LLM
```

## Data Flow: Message Lifecycle

```
1. New message arrives
   context.addMessage('user', 'Long message...')
                ↓
   ┌────────────────────────────────┐
   │  TIER 0: Hot Working Set       │
   │  Add to message queue          │
   │  Count tokens                  │
   └────────────────────────────────┘
                ↓
   Current tokens > max?
        NO → Keep in Tier 0
        YES → Evict oldest ↓
   ┌────────────────────────────────┐
   │  Evicted messages              │
   └────────────────────────────────┘
         ↓                    ↓
   ┌──────────────┐    ┌──────────────┐
   │  TIER 2      │    │  TIER 3      │
   │  Add to      │    │  Archive     │
   │  retrieval   │    │  for audit   │
   │  index       │    │  trail       │
   └──────────────┘    └──────────────┘

2. Structured state update
   context.updateState({ symbol: {...} })
                ↓
   ┌────────────────────────────────┐
   │  TIER 1: Structured State      │
   │  Update symbols map            │
   │  Compact representation        │
   └────────────────────────────────┘
                ↓
   ┌────────────────────────────────┐
   │  TIER 3: Archive update        │
   │  Record state change           │
   └────────────────────────────────┘

3. Knowledge chunk added
   context.addToMemory('docs...')
                ↓
   ┌────────────────────────────────┐
   │  TIER 2: Episodic Memory       │
   │  • Tokenize content            │
   │  • Build term frequency vector │
   │  • Index for retrieval         │
   └────────────────────────────────┘
                ↓
   ┌────────────────────────────────┐
   │  TIER 3: Archive chunk         │
   └────────────────────────────────┘

4. Query compilation
   context.compile('How does X work?')
                ↓
   ┌────────────────────────────────┐
   │  Gather Tier 1 (always)        │
   │  ✓ All symbols                 │
   │  ✓ All constraints             │
   │  ✓ All decisions               │
   │  ✓ All open loops              │
   │  ✓ All test failures           │
   └────────────────────────────────┘
                ↓
   ┌────────────────────────────────┐
   │  Retrieve from Tier 2          │
   │  • Vector similarity           │
   │  • BM25 keyword match          │
   │  • Hybrid score = 0.5×V + 0.5×B│
   │  • Sort and take top K         │
   └────────────────────────────────┘
                ↓
   ┌────────────────────────────────┐
   │  Get recent from Tier 0        │
   │  Last N messages in order      │
   └────────────────────────────────┘
                ↓
   ┌────────────────────────────────┐
   │  Format as structured prompt   │
   │  # Context State               │
   │  # Relevant Context            │
   │  # Recent Conversation         │
   │  # Current Query               │
   └────────────────────────────────┘
                ↓
            Return prompt
```

## Retrieval Algorithm (Tier 2)

```javascript
retrieve(query, k=10) {
  // 1. Tokenize query
  tokens = tokenize(query)
  queryTF = termFrequency(tokens)

  // 2. Score all chunks
  for each chunk in memory:
    // Vector similarity (cosine)
    vectorScore = cosineSim(queryTF, chunk.termFreq)

    // BM25 sparse matching
    bm25Score = 0
    for term in queryTokens:
      if term in chunk:
        idf = log((N - df(term) + 0.5) / (df(term) + 0.5) + 1)
        tf = chunk.termFreq[term]
        bm25Score += idf * (tf * (k1+1)) / (tf + k1*(1-b+b*docLen/avgLen))

    // Hybrid score
    score = 0.5 * vectorScore + 0.5 * bm25Score

  // 3. Sort by score, return top K
  return topK(chunks, k)
}
```

### Why Hybrid?

**Vector alone fails for exact matches:**
- Query: "JWT token"
- Chunk: "Use JWT for authentication"
- Vector: Moderate match (depends on shared terms)
- BM25: **Strong match** ("JWT" appears)

**BM25 alone fails for semantic similarity:**
- Query: "authentication credentials"
- Chunk: "Use tokens to verify identity"
- Vector: **Good match** (semantic similarity)
- BM25: Weak match (few shared terms)

**Hybrid gets both!**

## Token Economics

### Naive Approach
```
Keep everything in context:
- 100 messages × 500 tokens = 50,000 tokens
- Cost: High
- Latency: High
- Relevance: Mixed (90% irrelevant)
```

### Layered Approach
```
Tier 0: Recent 10 messages = 5,000 tokens
Tier 1: Structured state = 200 tokens
Tier 2: Top 5 retrieved = 1,000 tokens
────────────────────────────────────
Total in context: 6,200 tokens

Tier 3: 90 archived messages = 0 tokens (not in context)

Savings: 87.6% reduction
Quality: Higher (more relevant)
```

## Implementation Details

### Tier 0: Hot Working Set
- **Data structure**: Array (FIFO queue)
- **Index**: None (sequential access)
- **Eviction**: Oldest first when token limit reached
- **Token counting**: Rough estimate (chars/4)

### Tier 1: Structured State
- **Data structures**:
  - `symbols`: Map<name, {signature, file, type}>
  - `constraints`: Set<string>
  - `decisions`: Array<{decision, rationale, timestamp}>
  - `openLoops`: Map<id, {description, status}>
  - `testFailures`: Map<name, {error, file, line}>
- **Serialization**: JSON (compact)
- **Token cost**: ~10-20 tokens per item

### Tier 2: Episodic Memory
- **Data structure**: Array of chunks
- **Indexes**:
  - Inverted index: term → [chunk IDs]
  - Term frequency vectors per chunk
- **Retrieval**: O(n) scan with scoring (could optimize with ANN)
- **Storage**: In-memory (could persist)

### Tier 3: Cold Archive
- **Data structure**: Append-only array
- **Index**: timestamp → entry
- **Access patterns**: Rare (export, audit, recovery)
- **Storage**: In-memory (designed for cheap storage)

## Extension Points

### Custom Retrieval
```javascript
class SemanticMemory extends EpisodicMemory {
  async retrieve(query, k) {
    // Use real embeddings
    const queryEmbed = await getEmbedding(query);
    const chunks = this.chunks.map(c => ({
      chunk: c,
      score: cosineSim(queryEmbed, c.embedding)
    }));
    return topK(chunks, k);
  }
}
```

### Persistent Storage
```javascript
class PersistentArchive extends ColdArchive {
  async archive(type, data, metadata) {
    const id = super.archive(type, data, metadata);
    await db.insert('archive', { id, type, data, metadata });
    return id;
  }

  async getById(id) {
    return await db.query('SELECT * FROM archive WHERE id = ?', [id]);
  }
}
```

### State Compression
```javascript
class CompressedState extends StructuredState {
  serialize() {
    // Only keep most recent/important items
    return {
      symbols: this.getMostRecentSymbols(20),
      constraints: Array.from(this.state.constraints).slice(0, 10),
      decisions: this.state.decisions.slice(-5),
      openLoops: this.getActiveLoops(),
      testFailures: this.state.testFailures
    };
  }
}
```

## Comparison to Other Approaches

| Approach | Context Size | Relevance | Cost | Complexity |
|----------|--------------|-----------|------|------------|
| **Naive (all messages)** | 50K tokens | Mixed | High | Low |
| **Sliding window** | 16K tokens | Recent only | Medium | Low |
| **Pure RAG** | 10K tokens | Query-based | Medium | Medium |
| **Layered (this)** | 6K tokens | Multi-modal | Low | Medium |

**Key advantages:**
- Lower token usage than naive or pure RAG
- Better relevance than sliding window
- Structured state provides persistent context
- Hybrid retrieval catches both semantic and exact matches
- Full audit trail in cold archive

## Metrics to Measure

If implementing in production:

1. **Context Efficiency**: Tokens in context / Total tokens available
2. **Retrieval Precision**: Relevant chunks / Retrieved chunks
3. **Retrieval Recall**: Retrieved relevant / Total relevant
4. **State Compression**: Structured tokens / Equivalent text tokens
5. **Eviction Impact**: Task success before/after eviction

## Next Steps

1. **Better embeddings**: Replace term frequency with sentence transformers
2. **Persistent storage**: SQLite for Tier 2/3
3. **Incremental updates**: Don't rebuild entire index on each add
4. **Batch retrieval**: Retrieve once per conversation turn, not per query
5. **Adaptive thresholds**: Adjust tier sizes based on workload
6. **Graph retrieval**: Link related symbols/chunks
7. **Temporal decay**: Weight recent items higher in retrieval
