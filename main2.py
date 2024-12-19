import os
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from typing import Any
from pydantic import BaseModel

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
csv_file_path = "/Users/heem/Desktop/eo3.csv"

class CSVAnalysisAgent(Agent):
    # Declare csv_agent as a field
    csv_agent: Any = None

    def __init__(self, csv_path):
        # First initialize the parent class
        super().__init__(
            role="CSV Analysis Expert",
            goal="Analyze and answer questions about CSV data",
            backstory="I am an AI agent specialized in analyzing CSV files and providing insights from the data.",
            allow_delegation=False
        )
        
        # Then create the csv_agent after parent initialization
        self.csv_agent = create_csv_agent(
            llm, 
            csv_path, 
            verbose=True, 
            allow_dangerous_code=True
        )

    def invoke(self, user_query):
        return self.csv_agent.run(user_query)

def main():
    print("Welcome to the CSV Chatbot! Type 'quit' to exit.")
    print("You can ask questions about your CSV file.")
    
    agent = CSVAnalysisAgent(csv_file_path)
    
    while True:
        user_query = input("\nEnter your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using the CSV Chatbot. Goodbye!")
            break
            
        try:
            response = agent.invoke(user_query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try asking your question in a different way.")

if __name__ == "__main__":
    main()
