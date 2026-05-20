from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import kick_zscaler

from llama_cpp import Llama

embedding_model =  HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
)

texts = ["""Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.""",

"""Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles.
Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.""",

"""Interstellar premiered on October 26, 2014, in Los Angeles.
In the United States, it was first released on film stock, expanding to venues using digital projectors.
The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014.
It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight.
It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""
]

db = FAISS.from_texts(texts=texts, embedding=embedding_model)

question = "Income generated"

prompt_tpl = """Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}
"""

llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)

docs = db.similarity_search(question, k=2)
context = "\n\n".join(doc.page_content for doc in docs)

prompt = prompt_tpl.format_map({"context": context, "question": question})

response = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}])

print(response['choices'][0]['message']['content'])

llm.close()