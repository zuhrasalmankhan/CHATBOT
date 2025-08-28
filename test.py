from pymongo import MongoClient
from getpass import getpass


client = MongoClient("mongodb://localhost:27017")
db = client["chatbot"]
users_col = db["users"]
chats_col = db["chats"]

def seed_users():
    if users_col.count_documents({}) == 0:
        users_col.insert_many([
            {"username": "amna", "password": "1123"},
            {"username": "zahra", "password": "abcd"}
        ])
        print("Users inserted!")

def login():
    username = input("Enter username: ")
    password = getpass("Enter password: ")

    user = users_col.find_one({"username": username, "password": password})
    if user:
        print(f" Welcome {username}")
        return username
    else:
        print(" Invalid credentials")
        return None

# ---- Save chat ----
def save_message(username, role, content):
    chats_col.insert_one({
        "username": username,
        "role": role,          # "user" or "assistant"
        "content": content
    })

def get_history(username):
    messages = list(chats_col.find({"username": username}))
    return messages

if __name__ == "__main__":
    seed_users()
    user = None
    while not user:
        user = login()

    while True:
        query = input(f"\n{user}: ")
        if query.lower() in ["exit", "quit"]:
            break

        save_message(user, "user", query)
        # inserting llm reposnse

        # Save assistant message
        save_message(user, "assistant", response)
        print(f"AI: {response}")
