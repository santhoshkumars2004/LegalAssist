from langchain.prompts import PromptTemplate

template = """Create a final answer to the given questions using the provided document excerpts (given in no particular order) as sources. ALWAYS include a "SOURCES" section in your answer citing only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not have enough information to answer the question and leave the SOURCES section empty. Use only the provided documents and do not attempt to fabricate an answer.

---------

QUESTION: {question}
=========
{ipc_content}
=========
{cpc_content}
=========
{crpc_content}
=========
FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["question", "ipc_content", "cpc_content", "crpc_content"]
)
