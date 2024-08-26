from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import Union, List
from pinecone import Pinecone
from operator import itemgetter
import requests
import os
from flask import render_template

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Initialize Pinecone
api_key = os.getenv('PINECONE_API')
pc = Pinecone(api_key=api_key)
index_name = "legal"
index = pc.Index(index_name)

# Function for web search
def web_search(query_text):
    try:
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        subscription_key = os.getenv('BING_API_KEY')
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        params = {"q": query_text, "textDecorations": True, "textFormat": "HTML"}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        if 'webPages' in search_results and search_results['webPages']['value']:
            return search_results['webPages']['value'][0]
        else:
            return None

    except Exception as e:
        return {"error": f"An error occurred during the web search: {e}"}

# Function to format web search results
def format_web_search_results(search_result):
    if not search_result:
        return "No information available."

    name = search_result.get('name', 'No Title')
    url = search_result.get('url', 'No URL')
    snippet = search_result.get('snippet', 'No Snippet')

    formatted_result = f"**Title:** {name}\n**URL:** {url}\n**Snippet:** {snippet}\n"
    return formatted_result

# Function to store results in Pinecone
def store_in_pinecone(query_text, search_results, vector_store, embedding):
    try:
        text_to_embed = query_text + "\n" + format_web_search_results(search_results)
        vector_store.upsert([(query_text, embedding)])
        print(f"Stored search results for '{query_text}' in Pinecone.")
        
    except Exception as e:
        print(f"Error storing data in Pinecone: {e}")

# Function to query Pinecone
def query_pinecone(query_text, vector_store, top_k=5):
    try:
        query_response = vector_store.similarity_search(query_text, k=top_k)
        results = []

        if query_response and isinstance(query_response, list):
            for match in query_response:
                doc_id = match.get('id', 'No ID')
                score = match.get('score', 0.0)
                metadata = match.get('metadata', {})
                results.append({'id': doc_id, 'score': score, 'metadata': metadata})
        else:
            results = web_search(query_text)
            if results:
                text_to_encode = query_text + "\n" + format_web_search_results(results)
                embedding = embeddings.embed(text_to_encode)
                store_in_pinecone(query_text, results, vector_store, embedding)
                return results
            else:
                return {"message": "No information available."}

        return results
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Function for reciprocal rank fusion
def reciprocal_rank_fusion(docs_list):
    fused_results = []
    for docs in docs_list:
        if docs is None or not isinstance(docs, list):
            continue
        fused_results.extend(docs)
    return fused_results

# Function to check if query is law related
def is_law_related(query_text):
    keywords = ["law", "legal", "court", "attorney", "legal advice", "regulation", "legislation"]
    return any(keyword in query_text.lower() for keyword in keywords)

# Initialize PineconeVectorStore and ChatGroq
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Define route for handling messages
@app.route('/ask', methods=['POST'])
def handle_message():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        template = """You are a helpful legal assistant that generates multiple search queries according to Pakistan law based on a single input query.
Your goal is to generate specific and relevant legal search queries to assist with legal research and advice according to Pakistan Law.
Generate multiple legal search queries related to: {question}
Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        
        generate_queries = (
            prompt_rag_fusion
            | ChatGroq(temperature=0.2)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        
        template = """You are a knowledgeable and detail-oriented lawyer specializing in the legal system of Pakistan. 
        Answer the following legal question based on the provided context. Ensure your response is accurate, well-structured, and cites relevant laws, precedents, or legal principles where applicable.
        Context:
        {context}
        Question:
        {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        retrieval_chain_rag_fusion = RunnableParallel(
            passed=RunnablePassthrough(),
            modified=lambda x: {
                "query_results": query_pinecone(x["question"], vector_store, top_k=5),
                "fused_results": reciprocal_rank_fusion([query_pinecone(x["question"], vector_store, top_k=5)])
            }
        )
        
        llm = ChatGroq(temperature=0.5, model="llama3-groq-70b-8192-tool-use-preview")
        
        final_rag_chain = (
            {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        results = final_rag_chain.invoke({"question": question})

        if not results or len(results) == 0:
            if is_law_related(question):
                web_search_results = web_search(question)
                if web_search_results:
                    text_to_encode = question + "\n" + format_web_search_results(web_search_results)
                    store_in_pinecone(query_text=question, search_results=web_search_results, vector_store=vector_store, embedding=embeddings.embed(text_to_encode))
                    formatted_results = format_web_search_results(web_search_results)
                    return jsonify({"results": f"Results from web search:\n{formatted_results}"})
                else:
                    return jsonify({"results": "No information available."})
            else:
                return jsonify({"results": "No information available."})
        else:
            return jsonify({"results": f"Results from Pinecone: {results}"})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

# Define route for the welcome message
@app.route('/start', methods=['GET'])
def start():
    return jsonify({"message": "Welcome! You can ask me any legal question."})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


