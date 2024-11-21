import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import Tool, PDFSearchTool, SerperDevTool
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["SERPER_API_KEY"] = "SERPER_API_KEY"

class DatasetChatbot:
    def __init__(self, dataset_path, pdf_doc_path):
        """Initialize the chatbot with dataset and documentation paths"""
        self.dataset_path = dataset_path
        self.pdf_doc_path = pdf_doc_path
        self.df = pd.read_csv(dataset_path)  # Assuming CSV format, adjust if needed
        
        # Initialize tools using CrewAI's Tool class
        python_repl = PythonAstREPLTool()
        self.python_tool = Tool(
            name="Python REPL",
            description="A Python REPL for executing Python code to analyze data",
            func=python_repl.run
        )
        
        # Use PDFSearchTool directly as it's already a CrewAI tool
        self.pdf_tool = PDFSearchTool(pdf=pdf_doc_path)
        
        # Use SerperDevTool directly as it's already a CrewAI tool
        self.search_tool = SerperDevTool()

    def create_agents(self):
        """Create the specialized agents"""
        
        # Data Analysis Agent
        self.analyst = Agent(
            role='Data Analyst',
            goal='Analyze dataset and provide accurate insights',
            backstory="""You are an expert data analyst skilled at analyzing datasets 
                        and extracting meaningful insights. You have deep knowledge 
                        of pandas and data analysis.""",
            tools=[self.pdf_tool, self.python_tool],  # Using CrewAI tools
            verbose=True,
            allow_delegation=True
        )

        # Response Agent 
        self.communicator = Agent(
            role='AI Communication Specialist',
            goal='Provide clear, user-friendly responses about the dataset',
            backstory="""You are an expert at communicating complex data insights 
                        in simple, understandable terms. You excel at providing 
                        context and explanations.""",
            tools=[self.pdf_tool, self.search_tool],  # Using CrewAI tools
            verbose=True,
            allow_delegation=True
        )

    def create_tasks(self, user_question):
        """Create tasks based on user question"""
        
        # Analysis Task
        analysis_task = Task(
            description=f"""
                Analyze the dataset to answer: {user_question}
                Use the Python tool to analyze the data when needed.
                Use the PDF documentation tool to get additional context about columns.
                Dataset is available as 'self.df' in the Python environment.
                Provide detailed analysis results.
            """,
            agent=self.analyst,
            expected_output="Detailed analysis results with supporting data",
            human_input=False,
            tool_input={"args": {"description": user_question}}
        )

        # Response Formulation Task
        response_task = Task(
            description=f"""
                Using the analysis results and PDF documentation, create a clear,
                user-friendly response to: {user_question}
                Make sure to include relevant context from the documentation when needed.
                The response should be informative yet easy to understand.
            """,
            agent=self.communicator,
            expected_output="Clear, contextual response to the user's question",
            human_input=False,
            tool_input={"query": user_question}
        )

        return [analysis_task, response_task]

    def answer_question(self, question):
        """Process a user question and return an answer"""
        
        # Create crew
        crew = Crew(
            agents=[self.analyst, self.communicator],
            tasks=self.create_tasks(question),
            process=Process.sequential,
            verbose=True,
            memory=True  # Enable memory for context retention
        )

        # Get response
        result = crew.kickoff()
        return result

def main():
    # Initialize the chatbot
    chatbot = DatasetChatbot(
        dataset_path="/Users/heem/Desktop/eo3.csv",
        pdf_doc_path="/Users/heem/Desktop/eo_master_file.pdf"
    )
    chatbot.create_agents()

    # Interactive loop
    print("Dataset Chatbot initialized. Type 'exit' to quit.")
    while True:
        question = input("\nWhat would you like to know about the dataset? ")
        if question.lower() == 'exit':
            break
            
        try:
            answer = chatbot.answer_question(question)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"\nError processing question: {str(e)}")

if __name__ == "__main__":
    main()
