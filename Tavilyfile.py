import os
import requests

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  
URI = "https://api.tavily.com/search"

def query_tavily(question: str) -> str:
   
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}","Content-Type": "application/json"}
    payload = { "query": question , "max_results" : 3, "include_answer" : True} # what to send to the Tavily , aka the user question

    try:
        response = requests.post(URI, headers=headers, json=payload)
        print("Status code:", response.status_code)
        response = response.json() 

        answer = response.get("answer")    
        results = response.get("results", []) # getting the list of results from reponse
        output = ""
        if answer:
            output += f"Answer: {answer}\n\n"
        if not results:
            return "I couldn't find any relevant news at the moment."

        for r in results[:3]:  # top 3 results
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            output += f"Title: {title}\nURL: {url}\n\n"

        return output.strip()

    except Exception as e:
        return f"Tavily API error: {str(e)}"
