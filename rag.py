from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from tokens import tokens_to_text, text_to_tokens, get_tokens_count
from llm_client import client
import numpy as np


def chunk_text(text, chunk_token_size, overlap_token_size):
    """
    Chunks the given text into smaller pieces based on the specified token size and overlap size.
    Chunk based on tokens rather than characters as tokens per character vary a lot between pages and using characters may lead to inconsistent chunk sizes.

    Args:
        text (str): The text to be chunked.
        chunk_token_size (int): The maximum number of tokens per chunk.
        overlap_token_size (int): The number of overlapping tokens between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """

    if overlap_token_size >= chunk_token_size:
        raise ValueError("Overlap size must be smaller than chunk size.")

    text_tokens = text_to_tokens(text)
    start = 0

    chunks = []
    len_tokens = len(text_tokens)

    while start < len_tokens:
        # Determine the end index for the current chunk based on the chunk size.
        end = start + chunk_token_size
        tokens = text_tokens[start:end]
        chunk = tokens_to_text(tokens)
        chunks.append(chunk)

        # Move the start index forward by the chunk size minus the overlap size to get overlapping content from the previous chunk.
        start += chunk_token_size - overlap_token_size

        # If the remaining tokens are less than or equal to the overlap size ( which means it is only overlapping content from the previous chunk), 
        # stop creating new chunks.
        if len_tokens - start <= overlap_token_size:
            break
    return chunks


def chunk_pages(pages, chunk_token_size, overlap_token_size):
    """
    Chunks the content of multiple pages into smaller pieces based on the specified token size and overlap size.

    Args:
        pages (list): A list of dictionaries, each containing "url", "type", and "content" keys.
        chunk_token_size (int): The maximum number of tokens per chunk.
        overlap_token_size (int): The number of overlapping tokens between consecutive chunks.

    Returns:
        list: A list of dictionaries, each containing "url", "type", "chunk content", and "index" keys for each chunk.
    """
    all_chunks = []
    for page in pages:
        chunks_of_page = chunk_text(page["content"], chunk_token_size, overlap_token_size)
        for i, chunk_of_page in enumerate(chunks_of_page):
            chunk = {"url": page["url"], "type": page["type"], "content": chunk_of_page, "index": i}
            all_chunks.append(chunk)

    return all_chunks


def embed_texts(texts):
    start = 0
    vectors = []
    while start < len(texts):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend([item.embedding for item in response.data])
        start += EMBEDDING_BATCH_SIZE

    return np.array(vectors, dtype = np.float32)


def rank_chunks(chunks, similarities):
    """
    Returns copies of the chunks sorted from most to least similar to the query.

    Args:
        chunks (list): Chunk dicts, in the same order as the similarity scores.
        similarities (np.ndarray): One score per chunk, shape (len(chunks),).

    Returns:
        list: Chunk dicts with an added "score" key, best match first.
    """
    order = np.argsort(similarities)[::-1]   # argsort is ascending, so reverse it
    return [{**chunks[i], "score": float(similarities[i])} for i in order]


def union_top_k(chunks, similarities, topic_names, k):
    """
    Takes the top k chunks for every topic and merges them into one de-duplicated list.

    Args:
        chunks (list): Chunk dicts, in the same order as the rows of `similarities`.
        similarities (np.ndarray): Shape (n_chunks, n_topics), one column per topic.
        topic_names (list): Topic names, in the same order as the columns.
        k (int): How many chunks to take from each topic.

    Returns:
        list: One record per unique chunk:
              {"chunk": <chunk dict>, "topics": [names that picked it], "best_score": float}
    """
    union = {}
    for j, topic in enumerate(topic_names):
        for chunk in rank_chunks(chunks, similarities[:, j])[:k]:
            key = (chunk["url"], chunk["index"])   # unique per chunk: index restarts on each page
            if key not in union:
                original = {name: value for name, value in chunk.items() if name != "score"}
                union[key] = {"chunk": original, "topics": [], "best_score": chunk["score"]}
            union[key]["topics"].append(topic)
            union[key]["best_score"] = max(union[key]["best_score"], chunk["score"])
    return list(union.values())


def order_chunks(union, chunks):
    """
    Puts the selected chunks back into reading order: pages in the order they were
    fetched, and chunks within a page by their index.

    Args:
        union (list): Records returned by union_top_k.
        chunks (list): The full chunk list, used only to learn the page order.

    Returns:
        list: Chunk dicts (without topics or scores) in reading order.
    """
    page_position = {}
    for chunk in chunks:
        page_position.setdefault(chunk["url"], len(page_position))   # first appearance = page order

    selected = [record["chunk"] for record in union]
    return sorted(selected, key=lambda c: (page_position[c["url"]], c["index"]))


def total_chunk_tokens(chunks):
    return sum(get_tokens_count(chunk["content"]) for chunk in chunks)


def print_union_summary(union, k, n_topics, hub_min_topics=3):
    print(f"Union size: {len(union)} (max possible {k * n_topics})")
    for record in union:
        if len(record["topics"]) >= hub_min_topics:
            c = record["chunk"]
            print(f"hub | {c['type']:<12} | #{c['index']:<3} | {record['topics']} | {c['content'][:80]!r}")


def print_chunks(ranked, label):
    print(f"--- {label} ---")
    for rank, chunk in enumerate(ranked, start=1):
            print(f"rank {rank:>3} | score {chunk['score']:.3f} | #{chunk['index']:<3} | {chunk['type']:<12} | {chunk['content'][:100]!r}")