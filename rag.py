# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = "deepseek-r1:7b", embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        template_temp = """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}
            
            Question:
            {question}
            """

        # if (self.verbosity > 0):
        #     template_temp += "Answer concisely and accurately in {verbosity} sentences or less."

        self.prompt = ChatPromptTemplate.from_template(template_temp)
        
        # self.prompt = ChatPromptTemplate.from_template(
        #     """
        #     You are a helpful assistant answering questions based on the uploaded document.
        #     Context:
        #     {context}
            
        #     Question:
        #     {question}
        #     """
            
        

            
        
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()

        # Add filename to metadata for each doc
        for doc in docs:
            print (f"Adding source to metadata: ", pdf_file_path)
            doc.metadata["source"] = pdf_file_path
            doc.metadata["firstline"] = doc.page_content.split("\n")[0]
    

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def vs_samples(self):
        """
        List the data in the vector store.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        # Extract 'source' from document metadata
        metadata_list = self.vector_store.get()["metadatas"]
        return list({doc["firstline"] for doc in metadata_list if "firstline" in doc})
    
        
    
  


    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2, verbosity: int = 0):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
            "verbosity": verbosity
        }

        if (verbosity > 0):
            self.prompt = ChatPromptTemplate.append("Answer concisely and accurately in {verbosity} sentences or less.")

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None
    