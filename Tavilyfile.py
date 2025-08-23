import os
import requests
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
URI = "https://api.tavily.com/search"

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set. Please add it to your environment.")

def query_tavily(question: str) -> str:
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": question,
        "max_results": 3,
        "include_answer": True
    }

    try:
        response = requests.post(URI, headers=headers, json=payload)
        print("Status code:", response.status_code)
        response.raise_for_status()  # will raise if not 2xx
        data = response.json()

        answer = data.get("answer")
        results = data.get("results", [])

        output = ""
        if answer:
            output += f"Answer: {answer}\n\n"
        if results:
            for r in results[:3]:
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                output += f"Title: {title}\nURL: {url}\n\n"

        # Always return something, never empty
        return output.strip() or "No relevant information found."

    except Exception as e:
        return f"Tavily API error: {str(e)}"
