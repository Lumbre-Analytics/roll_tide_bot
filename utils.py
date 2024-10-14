import os
from dotenv import load_dotenv
import hashlib
import json

import wikipediaapi
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def download_wikipedia_pages_by_category(
    categories:list,
    env_directory:str =".env",
    write_directory:str ="./docs/"
    ):
    """
    Takes a list of Wikipedia Category Names and downloads all Wikipedia pages within 
    those categories to a local directory. 
    
    Parameters:
    categories (list): List of Wikipedia Categories.
    env_directory (string): Path to .env file which contains the user's Wikipedia User Agent.
    write_directory (string): Path in which to write wikipedia pages.
    """
    
    # Load Wikipedia User Agent from .env  
    load_dotenv(env_directory)
    user_agent = os.environ.get("WIKIPEDIA_USER_AGENT")

    wiki_wiki = wikipediaapi.Wikipedia(user_agent, 'en')

    # Get a list of all pages in our chosen categories
    all_pages = []
    for cat in categories:
        pages = wiki_wiki.page(cat)
        pages = [page for page in pages.categorymembers.keys() if "Category:" not in page]
        all_pages += pages

    wiki_html = wikipediaapi.Wikipedia(
    user_agent=user_agent,
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    # Download html from all pages in list as text files
    for page in all_pages:
        p_html = wiki_html.page(page)
        with open(f'{write_directory}{page}.txt', 'w') as f:
            f.write(p_html.text)

    return None


def stable_hash(doc: Document) -> str:
    """
    Stable hash document based on its metadata.
    """
    return hashlib.sha1(json.dumps(doc.metadata, sort_keys=True).encode()).hexdigest()


def create_vector_db(docs_dir:str, db_dir:str, env_dir:str=".env"):
    """
    Create document embeddings using OpenAI's API and text-embedding-ada-002 embedding model,
    Store embedded documents in local a ChromaDB vector database.

    Parameters:
    docs_dir: (str): Directory of documents to embed.
    db_dir (string): Local directory to host the vectorstore.
    env_dir (string): Path to .env file which contains the user's Wikipedia User Agent.
    """

    # Load Open AI API key from .env file
    load_dotenv(env_dir)

    # Initialize embedding model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Initialize Chroma DB vector database
    docs_vectorstore = Chroma(
        collection_name="docs_store",
        embedding_function=embeddings_model,
        persist_directory=db_dir,
    )

    # Load docs to vectorize
    loader = DirectoryLoader(
        docs_dir,
        glob="*.txt",
        loader_kwargs={"open_encoding": "utf-8"},
        recursive=True,
        show_progress=True,
    )
    docs = loader.load()

    # Split docs into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)

    # Keep track of where splits have been made
    split_ids = list(map(stable_hash, splits))

    docs_vectorstore.add_documents(splits, ids=split_ids)

    return None

   

if __name__=="__main__":
    download_wikipedia_pages_by_category(
        categories= [
            "Category:Alabama_Crimson_Tide_football_seasons",
            "Category:Alabama_Crimson_Tide_football",
            "Category:Alabama_Crimson_Tide_football_games",
            "Category:Alabama_Crimson_Tide_football_bowl_games",
        ]
    )
    
    create_vector_db(docs_dir="docs", db_dir="docs-db")