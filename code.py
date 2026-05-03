"""
RAG Chatbot Development Lab - Starter Code
Course: Create Chatbots & NLP Apps
Module 1: RAG Chatbot Development - Foundation
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA


import os

# PROVIDED CODE - DO NOT MODIFY
# Sample knowledge base content (normally loaded from file)
SAMPLE_KB_CONTENT = """
What are our business hours? Our customer service team is available Monday through Friday, 9 AM to 6 PM EST.

How do I reset my password? To reset your password, click the "Forgot Password" link on the login page and follow the email instructions.

What payment methods do you accept? We accept all major credit cards, PayPal, and bank transfers for premium accounts.

How do I track my order? You can track your order by logging into your account and visiting the "Order History" section.

What is your return policy? We offer 30-day returns for unused items in original packaging. Return shipping is free for defective products.

How do I contact customer support? You can reach us via email at support@company.com, phone at 1-800-555-0123, or through our live chat feature.

Can I change my shipping address? You can modify your shipping address up to 2 hours after placing your order through your account dashboard.

Do you offer international shipping? Yes, we ship to over 50 countries. International shipping costs and delivery times vary by location.
"""

# PROVIDED CODE - DO NOT MODIFY
def create_sample_knowledge_base():
    """Create a sample knowledge base file for testing"""
    with open('knowledge_base.txt', 'w') as f:
        f.write(SAMPLE_KB_CONTENT)
    print("Sample knowledge base created: knowledge_base.txt")

# PROVIDED CODE - DO NOT MODIFY
def calculate_sus_score(responses):
    """
    Calculate SUS score from 10-item questionnaire responses
    responses: list of 10 integers (1-5 scale)
    """
    if len(responses) != 10:
        raise ValueError("SUS requires exactly 10 responses")

    score = 0
    for i, response in enumerate(responses):
        if i % 2 == 0:  # Odd-numbered questions (positive)
            score += response - 1
        else:  # Even-numbered questions (negative)
            score += 5 - response

    return score * 2.5

# Activity 1: Knowledge Base Preparation
print("Activity 1: Knowledge Base Preparation")

# Step 1: Create and load knowledge base
create_sample_knowledge_base()

### PRACTICE CHALLENGE 1 ###
# TASK: Modify the text splitter to use 600-character chunks with 100-character overlap
# and compare the retrieval quality for complex multi-step questions
# YOUR CODE HERE

# Step 2: Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Modify this for Practice Challenge 1
    chunk_overlap=50  # Modify this for Practice Challenge 1
)

# Load and split documents
loader = TextLoader('knowledge_base.txt')
documents = loader.load()
texts = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents")
print(f"Created {len(texts)} text chunks")

# Step 3: Create vector database
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector database created successfully")

# Activity 2: RAG Chain Implementation
print("\nActivity 2: RAG Chain Implementation")

# Step 1: Initialize language model
llm = OllamaLLM(
    model="mistral",
    temperature=0.3
)

# Step 2: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

print("RAG chatbot initialized")

### PRACTICE CHALLENGE 2 ###
# TASK: Implement error handling that gracefully responds when the knowledge base
# doesn't contain relevant information for a user's query
# YOUR CODE HERE

def chat_with_error_handling(query):
    """Enhanced chat function with error handling"""
    try:
        result = qa_chain.invoke({"query": query})

        # Basic implementation - enhance for Practice Challenge 2
        return result['result'], result['source_documents']
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}", []

# Step 3: Test the chatbot
test_queries = [
    "What are your business hours?",
    "How do I reset my password?",
    "Can you help me with a refund?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    response, sources = chat_with_error_handling(query)
    print(f"Response: {response}")
    print(f"Sources used: {len(sources)} documents")

# Activity 3: SUS Evaluation Implementation
print("\nActivity 3: SUS Evaluation Implementation")

# PROVIDED CODE - DO NOT MODIFY
SUS_QUESTIONS = [
    "I think that I would like to use this chatbot frequently.",
    "I found the chatbot unnecessarily complex.",
    "I thought the chatbot was easy to use.",
    "I think that I would need the support of a technical person to be able to use this chatbot.",
    "I found the various functions in this chatbot were well integrated.",
    "I thought there was too much inconsistency in this chatbot.",
    "I would imagine that most people would learn to use this chatbot very quickly.",
    "I found the chatbot very cumbersome to use.",
    "I felt very confident using the chatbot.",
    "I needed to learn a lot of things before I could get going with this chatbot."
]

### PRACTICE CHALLENGE 3 ###
# TASK: Design a test scenario where users might encounter the chatbot's limitations
# and measure how these failures impact SUS scores
# YOUR CODE HERE

def create_test_scenarios():
    """Create realistic test scenarios for SUS evaluation"""
    scenarios = [
        {
            "description": "Customer wants to check business hours",
            "query": "What are your business hours?",
            "expected_outcome": "Should provide specific hours and days"
        },
        {
            "description": "Customer needs password reset help",
            "query": "I forgot my password, how do I reset it?",
            "expected_outcome": "Should provide step-by-step instructions"
        },
        {
            "description": "Customer asking about payment methods",
            "query": "What payment options do you accept?",
            "expected_outcome": "Should list accepted payment methods"
        }
    ]
    return scenarios

# Sample SUS evaluation responses (1-5 scale for each question)
sample_responses = [4, 2, 4, 2, 4, 2, 4, 2, 4, 2]  # Sample data
sus_score = calculate_sus_score(sample_responses)
print(f"Sample SUS Score: {sus_score}")

def conduct_sus_evaluation():
    """Framework for conducting SUS evaluation"""
    print("\nSUS Evaluation Framework:")
    print("1. Present test scenarios to users")
    print("2. Allow natural interaction with chatbot")
    print("3. Collect responses to 10 SUS questions")
    print("4. Calculate final SUS score")
    print("5. Interpret results (>70 = good, >80 = excellent)")

conduct_sus_evaluation()

print("\nLab completed! You have successfully:")
print("✓ Implemented RAG chatbot with document retrieval")
print("✓ Created vector database from knowledge base")
print("✓ Built question-answering system with source attribution")
print("✓ Established SUS evaluation framework")
print("✓ Tested system with realistic customer scenarios")

