# RAG Lab - Web Content Retriever

A learning lab for retrieval-augmented generation (RAG) over website content: token-based chunking, embeddings, similarity search, multi-query selection and token-budget filling. Along the way: token budgets and cost control, scraping, tool calling and small Gradio UIs, with the reason for each decision recorded.

NOTE: It is built without any in-built framework for now, adding LangChain and a vector database is a possible later step. 

The task used to try it is a **company brochure generator**: given a company's website, it writes a short Markdown brochure for prospective customers, investors and recruits.

## What I built and learned

| Topic | Where |
|---|---|
| LLM calls, streaming, JSON output (link selection) | `brochure_generation.py`, `llm_client.py` |
| Token budgeting with `tiktoken`, measured on real sites | `tokens.py`, `config.py` |
| RAG from scratch: chunking, embeddings, similarity, multi-query selection, budget filling | `rag.py` |
| Freezing fetched pages so experiments repeat exactly | `page_cache.py` |
| Scraping and cleaning page text | `scraper.py` |
| Tool calling and an agent loop, with Gradio UIs | `SalesBrochureWithUI/` |

## How it works

**Baseline** (the default):

```
URL -> collect links -> LLM picks the relevant ones (JSON) -> fetch each page
    -> keep the first MAX_PAGE_TOKENS tokens of each page -> one prompt -> LLM writes the brochure
```

**RAG path** (`RAG_ENABLED=true`):

```
URL -> same pages, uncut, saved in cache/<site>.json
    -> split every page into overlapping token chunks -> embed chunks and five topic queries
    -> score every chunk against every topic -> fill RAG_BUDGET_TOKENS by taking turns between topics
    -> put the chosen chunks back in reading order -> one prompt -> LLM writes the brochure
```

The five topic queries (`prompts.py`): what the company does, products and technology, customers, team culture and careers, financials and investors.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file:

```dotenv
OPENAI_API_KEY=your-openai-api-key
RAG_ENABLED=true        # optional; leave out for the baseline
```

```bash
python main.py https://example.com            # prints the brochure
python main.py https://example.com --stream   # streams it (baseline path only)
```

On the RAG path the fetched pages are saved in `cache/<domain>.json` and reused. Delete a site's file to fetch it again (the link selection is an LLM call, so it can pick different pages each time). `cache/` should stay out of git.

**Gradio apps** (run from the project root):

```bash
python -m SalesBrochureWithUI.stream_brochure_with_ui         # URL form, streams the brochure
python -m SalesBrochureWithUI.conversational_AI_with_ui       # chat that calls the brochure generator as a tool
PYTHONPATH=. gradio SalesBrochureWithUI/conversational_AI_with_ui.py --demo-name view    # auto-reload
```

`python path/to/file.py` only puts that file's folder on `sys.path`, so `from config import ...` fails. `-m` or `PYTHONPATH=.` from the root fixes it. The `gradio` command starts the app in a subprocess, so setting `PYTHONPATH` is what makes the import work there.

## Configuration (`config.py`)

| Setting | Value | Meaning |
|---|---|---|
| `API_MODEL` | `gpt-4o-mini` | Chat model; also selects the `tiktoken` encoding |
| `MAX_LINKS_PER_PAGE` | 10 | Linked pages fetched besides the landing page |
| `MAX_PAGE_TOKENS` | 2,500 | Tokens kept per page in the baseline |
| `CHUNK_TOKENS_SIZE` / `CHUNK_OVERLAP_TOKENS_SIZE` | 350 / 50 | Chunk size and overlap |
| `EMBEDDING_MODEL` / `EMBEDDING_BATCH_SIZE` | `text-embedding-3-small` / 100 | Embeddings and texts per request |
| `RAG_BUDGET_TOKENS` | 8,000 | Chunk tokens sent to the model on the RAG path |
| `REPEATED_LINE_MIN_SHARE` | 0.35 | Lines repeated across more than this share of pages are removed as menu text |
| `RAG_ENABLED` | `false` | From `.env`: switches to the RAG path |
| `MAX_CONTENT_TOKENS` | 25,000 | Defined but not enforced anywhere |

## What I learned

**Tokens, not characters.** Characters per token varied from about 1.1 (Hugging Face's pricing page) to about 4 (ordinary prose), so a character cap cannot predict cost or fit. Limits are in tokens, measured with `tiktoken`. The context window (200k) is not the reason to limit input; cost, latency and noise are.

**Look at what the scraped text really is.** Cleaning the page text mattered more than any retrieval setting:
- `nav` and `footer` removal is safe (4 to 23% fewer tokens). Removing `header`, or anything hidden or `aria-hidden`, deletes real content such as headings, customer stories and testimonials.
- 82 to 84% of Hugging Face's pricing and enterprise pages were text art (`✳ ⟡ · ◎`). Dropping symbol-only lines took the whole site from 35.6k to about 11k tokens.
- Menus built from plain `div`s are removed by deleting lines that repeat across 35% or more of a site's pages.
- `requests` guesses ISO-8859-1 when a server sends no charset, which garbled UTF-8 text. Passing the raw bytes to BeautifulSoup fixes it.

**Chunking by tokens, not characters.** The same number of characters can be 300 tokens (prose) or 1,000 (price tables, symbols), so character-sized chunks would vary widely in what they cost the budget and the embedding model. Token-sized chunks are even, fit the embedding input limit and make the `fill_budget` arithmetic exact. Chunks are cut on the token list (350 tokens, 50 overlap) and each slice is decoded back to text. The cost is that a cut can fall mid-sentence; the overlap softens this, and splitting on paragraph boundaries is the alternative I did not try. Each chunk keeps its page type, URL and position so the selection can be put back in reading order. Plain NumPy was enough for similarity: OpenAI embeddings have length 1, so the dot product is the cosine similarity.

**One query is the wrong shape for a brochure.** A query mixing products, customers and careers blurs into one point between the topics, so scores were low and flat (0.04 to 0.27) and generic chunks ranked well. Five topic queries scored separately work better than one. Queries should describe content, not give instructions.

**Fill the budget by taking turns.** Scores are not comparable between topics, so sorting all chunks by score could let one topic take the whole budget. `fill_budget` gives each topic its best unchosen chunk in turn until the budget is used.

**Tool calling.** The model chooses tools from their name and description; tool results go back as `tool` messages matched by `tool_call_id`, after one assistant message that carries all the `tool_calls`; streamed tool calls arrive in fragments that are merged by `index`. A Gradio chat function must `yield` its output, because a `return` value is discarded.

## Evaluation (to be implemented)

Not built yet. The RAG path produces a brochure, but nothing measures whether its selection is better than the baseline's. The plan:

1. Freeze the pages of a few sites (done, see `page_cache.py`).
2. Write a short list of must-have facts per site, grouped by topic (the slogan, a few customer names, a careers line, a product).
3. Build the baseline context and the RAG context at the **same token budget**, so only the selection differs.
4. Count how many facts each context contains, overall and per topic, and read the generated brochures too.

Things to remember when it is built: the fact list is chosen by hand, so about half of it will sit near the top of pages, which favours the baseline; the count measures what reaches the model, not how good the brochure is; and the RAG prompt adds a heading above every chunk, so compare total prompt tokens as well as the budget. A hybrid (the first lines of each page plus retrieved chunks) is an idea to try once the comparison exists.

## Guardrails (to be implemented)

Only **resource** guardrails exist so far: a token cap per page on the baseline and `RAG_BUDGET_TOKENS` on the RAG path (`MAX_CONTENT_TOKENS` is defined but not enforced), and links such as `mailto:` and `#anchors` are filtered out before link selection. There are no safety or quality guardrails. The plan, by kind:

| Kind | What to add | Why it is needed here |
|---|---|---|
| **Input** | Allow only `http` and `https` URLs and refuse localhost and private addresses; add a timeout to every request | In the chat, the URL comes from a user and goes straight to `requests.get`, so the server can be made to fetch any address; today a request can also hang |
| **Untrusted text** | Tell the model that page text is data and not instructions, and separate it clearly in the prompt | Scraped text is put into the prompt, so a page could say "ignore your instructions"; in the chat the brochure also returns to a model that has tools |
| **Quality** | Tell the model to use only the provided text; if the fetched pages total very few tokens, say so instead of writing a brochure | Thin input (google.com, a maintenance page) lets the model fill gaps from what it already knows |
| **Structured output** | Validate the link JSON: required keys present, and only URLs that were in the link list | The model can return a missing key or an invented URL, which today crashes the fetch loop or wastes a request |
| **Loops and cost** | Cap the chat tool loop's iterations, validate the tool arguments, set `max_tokens` on the calls | The loop is `while True` and `json.loads` of the arguments is unguarded |

Things to remember when it is built: each guardrail should be tested with a bad input (a localhost URL, a page that contains "ignore previous instructions", the Patagonia maintenance page, a link JSON with a missing key), and every refusal should say why instead of failing silently.

## Known issues and not done

- The scraper does not run JavaScript. Script-rendered or blocked sites give almost nothing (google.com's landing page is about 33 tokens; patagonia.com returned a maintenance page). With thin input the model may fill gaps from what it already knows.
- Link selection is an LLM call and varies between runs; once it returned ten `product` pages for Salesforce and nothing about the company. Links are not made absolute or deduplicated, so `stripe.com` and `stripe.com/` were fetched as two identical pages.
- On the RAG path with `RAG_ENABLED=true`, the scraper skips the per-page cap, so `--stream` and the form UI (which do not use RAG) send uncapped pages.
- No near-duplicate removal (the same logo wall on several pages), no per-topic score threshold (a topic with nothing on the site still takes chunks), and the heading above every chunk adds tokens that the budget does not count.
- No guardrails beyond the token caps (see above).
