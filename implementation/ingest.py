import os
import glob
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


load_dotenv(override=True)


KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)


# Connect Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)


index_name = "insurellm-main"


# Create index if it does not exist
if index_name not in pc.list_indexes().names():

    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


index = pc.Index(index_name)


NAMESPACE = "insurellm"



def fetch_documents():

    documents = []

    for folder in glob.glob(
        str(Path(KNOWLEDGE_BASE) / "*")
    ):

        doc_type = os.path.basename(folder)

        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={
                "encoding": "utf-8"
            },
        )

        for doc in loader.load():

            doc.metadata["doc_type"] = doc_type
            documents.append(doc)


    return documents



def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)



def embed_and_prepare(chunks):

    vectors = []


    for i, doc in enumerate(chunks):

        vector = embeddings.embed_query(
            doc.page_content
        )


        vectors.append({

            "id": f"doc-{i}",

            "values": vector,

            "metadata": {

                "source": doc.metadata.get(
                    "source",
                    ""
                ),

                "text": doc.page_content,

                "doc_type": doc.metadata.get(
                    "doc_type",
                    ""
                )
            }
        })


    return vectors



def insert_vectors(vectors):

    index.upsert(
        vectors=vectors,
        namespace=NAMESPACE
    )



if __name__ == "__main__":


    docs = fetch_documents()

    print(
        f"Loaded {len(docs)} documents"
    )


    chunks = create_chunks(docs)

    print(
        f"Created {len(chunks)} chunks"
    )


    vectors = embed_and_prepare(chunks)


    insert_vectors(vectors)


    print(
        f"🚀 Inserted {len(vectors)} vectors into Pinecone"
    )