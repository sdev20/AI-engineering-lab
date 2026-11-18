import json
from config import CHUNK_OVERLAP_TOKENS_SIZE, MAX_CONTENT_TOKENS, MAX_LINKS_PER_PAGE, API_MODEL, CHUNK_TOKENS_SIZE, RAG_ENABLED
from page_cache import get_pages
from prompts import get_brochure_query, get_brochure_system_prompt, get_brochure_user_prompt, get_relevant_links_system_prompt, get_relevant_links_user_prompt, topic_queries
from llm_client import client
from rag import chunk_pages, embed_texts, order_chunks, print_chunks, print_union_summary, rank_chunks, total_chunk_tokens, union_top_k
from scraper import fetch_website_content, fetch_website_links
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



model = API_MODEL


def fetch_relevant_links(links, website_url):
    """
    Fetches relevant links for brochure generation from the given list of links.

    Args:
        links (list): The list of links to process.
        website_url (str): The URL of the website to scrape.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_relevant_links_system_prompt()},
            {"role": "user", "content": get_relevant_links_user_prompt(links, website_url)}
        ],
        response_format={"type": "json_object"} # Specify the response format as JSON
    )

    relevant_links = response.choices[0].message.content
    return json.loads(relevant_links)  # Parse the JSON string into a Python dictionary

def fetch_content_and_relevant_links(website_url):
    """
    Fetches the content of the given website URL and relevant links for brochure generation.

    Args:
        website_url (str): The URL of the website to scrape.

    Returns:
        list: A list of pages with their content and URLs.
    """

    # fetch the list of links from the website
    website_links = fetch_website_links(website_url) 
    # fetch the relevant links for brochure generation
    relevant_links = fetch_relevant_links(website_links, website_url)

    pages = [{"type": "landing", "content": fetch_website_content(website_url), "url": website_url}]

    for link in relevant_links['links'][:MAX_LINKS_PER_PAGE]:
       pages.append({"type": link['type'], "content": fetch_website_content(link["url"]), "url": link["url"]})
    return pages


def create_brochure(website_url):
    if RAG_ENABLED:
        return create_brochure_with_rag(website_url)
    else:
        return create_brochure_with_no_rag(website_url)
    

def create_brochure_with_no_rag(website_url):
    """
    Creates a brochure for the given website URL.

    Args:
        website_url (str): The URL of the website to scrape.
    """
    pages = fetch_content_and_relevant_links(website_url)
    company_content = format_pages_for_brochure(pages)

    system_prompt = get_brochure_system_prompt() 
    user_prompt = get_brochure_user_prompt(company_content)

    brochure_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

    result = brochure_response.choices[0].message.content
    return result


def create_brochure_with_rag(website_url):
    """
    Creates a brochure for the given website URL using RAG (Retrieval-Augmented Generation).

    Args:
        website_url (str): The URL of the website to scrape.
    """
    pages = get_pages(website_url, fetch_content_and_relevant_links)
    chunks = chunk_pages(pages, CHUNK_TOKENS_SIZE, CHUNK_OVERLAP_TOKENS_SIZE)
    chunks_embeddings = embed_texts([c['content'] for c in chunks])
  
    print(f"Chunks embeddings shape: {chunks_embeddings.shape}")

    #query_embeddings = embed_texts([get_brochure_query()])[0]

    # similarities = chunks_embeddings @ query_embeddings

    # ranked = rank_chunks(chunks, similarities)
    # print_chunks(ranked, "All chunks ranked by similarity to the brochure query")
    # print_chunks(ranked[:10], "top 10")
    # print_chunks(ranked[-10:], "bottom 10")


    topic_query_embeddings = embed_texts(list(topic_queries.values()))
    print(f"Topic query embeddings shape: {topic_query_embeddings.shape}")

    similarities = chunks_embeddings @ topic_query_embeddings .T
    print(f"Similarities shape: {similarities.shape}")

    topic_names = list(topic_queries)
    union = union_top_k(chunks, similarities, topic_names, k=10)
    print_union_summary(union, len(topic_names), 10)
    print(f"Union of top 10 chunks for each topic: {len(union)} chunks")
    ordered = order_chunks(union, chunks)
    print(total_chunk_tokens(ordered), "tokens in ordered union of top 10 chunks for each topic")


    # company_content = format_pages_for_brochure(pages)

    # system_prompt = get_brochure_system_prompt() 
    # user_prompt = get_brochure_user_prompt(company_content)

    # brochure_response = client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt}
    #         ],
    #     )

    # result = brochure_response.choices[0].message.content
    #return result
    return ""



def stream_brochure(website_url):
    """
    Creates a brochure for the given website URL.

    Args:
        website_url (str): The URL of the website to scrape.
    """
    pages = fetch_content_and_relevant_links(website_url)
    company_content = format_pages_for_brochure(pages)
    brochure_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_brochure_system_prompt()},
                    {"role": "user", "content": get_brochure_user_prompt(company_content)}
                ],
                stream=True
            )

    result = ""
    for chunk in brochure_response:
        if chunk.choices[0].delta.content is not None:
            delta = chunk.choices[0].delta.content
            result += delta
            print(delta, end="", flush=True)

    return result


def stream_brochure_for_ui_chat(website_url):
    """
    Creates a brochure for the given website URL.

    Args:
        website_url (str): The URL of the website to scrape.
    """
    pages = fetch_content_and_relevant_links(website_url)
    company_content = format_pages_for_brochure(pages)
    brochure_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_brochure_system_prompt()},
                    {"role": "user", "content": get_brochure_user_prompt(company_content)}
                ],
                stream=True
            )

    result = ""
    for chunk in brochure_response:
        if chunk.choices[0].delta.content is not None:
            delta = chunk.choices[0].delta.content
            result += delta
            yield result # if called from gradio, this will stream the result to the UI in real-time
            

def format_pages_for_brochure(pages):
    formatted_content = f"## Landing Page:\n\n{pages[0]['content']}\n## Relevant Links:\n"
    for page in pages[1:]:
        formatted_content += f"\n\n### Link: {page['type']}\n"
        formatted_content += page['content']
    return formatted_content


def format_chunks_for_brochure(chunks):
    formatted_content = ""
    for chunk in chunks:
        formatted_content += f"\n\n### Link: {chunk['type']} | URL: {chunk['url']} | Chunk Index: {chunk['index']}\n"
        formatted_content += chunk['content']
    return formatted_content