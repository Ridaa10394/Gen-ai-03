from groq import Groq

CHAT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided document context.

Rules:
- Only answer using information from the context below.
- If the answer is not in the context, reply exactly: "I cannot find this in the document."
- Be concise and cite the page number when possible, e.g., (page 3).
- Do not invent facts or use outside knowledge.
"""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c in chunks)


def answer(client: Groq, query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot find this in the document."
    context = format_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
