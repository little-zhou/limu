"""
@Author: danyzhou
@Updated: 11/11/25 3:17 PM

Advantest Confidential - All Rights Reserved
"""

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes

import os

# 设置 API 密钥
os.environ["GOOGLE_API_KEY"] = "AIzaSyDasOR22yOTEF8Qbr_GhA5VyLnXDUTeMpc"

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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"

# 1. create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. create parser
parser = StrOutputParser()

# 4. create chain
chain = prompt_template | model | parser

# 5. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces"
)

# 6. adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
