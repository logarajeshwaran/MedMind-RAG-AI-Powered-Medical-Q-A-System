from openai import OpenAI
from src.reteriver import retrieve
from src.config import OPENAI_API_KEY,LLM,MAX_TOKENS



def build_prompt(query,context_chunks):

    context = """\n\n -- \n\n  """.join(chunk['text'] for chunk in context_chunks)

    prompt = f""" You are a knowledgeable medical assistant you have to use the CONTEXT below to answer the question
    If the context doesnot contain enough information say " I dont have enough information"

    ====Context====
    {context}

    ===Question====
    {query}

    ===Instriction===
    1. Be Clear and accurate
    2. Do not hallucinate or add information outside the context
    3. If relevant , Mention which part of the context supports your answer


    ===Answer====
"""
    return prompt


def ask_query(query):

    chunk = retrieve(query)

    prompt = build_prompt(query,chunk)

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model = LLM,
        max_tokens = MAX_TOKENS,
        messages = [
            {
                "role"    : "system",
                "content" : "You are a helpful and accurate medical assistant"
             },
             {
                 "role" : "user",
                 "content" : prompt
             } 
        ]
    )

    answer = response.choices[0].message.content.strip()

    return {
        "query" : query,
        "answer" : answer,
        "sources" : chunk
    }