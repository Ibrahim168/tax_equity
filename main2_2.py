import os
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any
from pydantic import BaseModel

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
csv_file_path = "/Users/heem/Desktop/eo3.csv"
pdf_file_path = "/Users/heem/Desktop/eo_master_file.pdf"

class CSVAnalysisAgent(Agent):
    csv_agent: Any = None

    def __init__(self, csv_path):
        super().__init__(
            role="CSV Analysis Expert",
            goal="Analyze and answer questions about CSV data",
            backstory="I am an AI agent specialized in analyzing CSV files and providing insights from the data.",
            allow_delegation=False
        )
        
        self.csv_agent = create_csv_agent(
            llm, 
            csv_path, 
            verbose=True, 
            allow_dangerous_code=True
        )

    def invoke(self, user_query):
        return self.csv_agent.invoke(user_query)

class PDFContextAgent(Agent):
    vector_store: Any = None

    def __init__(self, pdf_path):
        super().__init__(
            role="PDF Context Expert",
            goal="Search and provide relevant context from PDF documentation",
            backstory="I am an AI agent that helps find relevant information from PDF documentation to enhance data analysis.",
            allow_delegation=False
        )
        
        # Load and process PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(texts, embeddings)

    def invoke(self, query):
        # Search for relevant context
        docs = self.vector_store.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Use LLM to synthesize relevant information
        prompt = f"Based on the following context from the documentation, what information is relevant to answer the query: '{query}'?\n\nContext: {context}"
        response = llm.invoke(prompt)
        return response.content

class ManagerAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Query Manager",
            goal="Determine if a query needs additional context from documentation",
            backstory="I am an AI agent that analyzes user queries and determines whether they need additional context from documentation to be answered properly.",
            allow_delegation=True
        )

    def invoke(self, query):
        prompt = f"""Analyze this query and determine if it likely needs additional context from documentation to be answered properly. 
        Query: {query}
        
        Respond with either 'YES' or 'NO' followed by a brief explanation.
        """
        response = llm.invoke(prompt)
        return response.content.strip().upper().startswith('YES')

def main():
    print("Welcome to the Enhanced CSV Chatbot! Type 'quit' to exit.")
    print("You can ask questions about your CSV file and related documentation.")
    
    # Add file existence checks
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_file_path}")
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_file_path}")
    
    csv_agent = CSVAnalysisAgent(csv_file_path)
    pdf_agent = PDFContextAgent(pdf_file_path)
    manager_agent = ManagerAgent()
    
    while True:
        user_query = input("\nEnter your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using the Enhanced CSV Chatbot. Goodbye!")
            break
            
        try:
            # First, let the manager decide if we need context
            needs_context = manager_agent.invoke(user_query)
            
            if needs_context:
                # Get relevant context from PDF
                context = pdf_agent.invoke(user_query)
                # Enhance the original query with context
                enhanced_query = f"""Consider this additional context while answering the question:
                {context}
                
                Now answer this question: {user_query}"""
                response = csv_agent.invoke(enhanced_query)
            else:
                response = csv_agent.invoke(user_query)
                
            print("\nResponse:", response)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try asking your question in a different way.")

if __name__ == "__main__":
    main()
