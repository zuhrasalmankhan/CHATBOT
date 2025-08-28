from pymongo import MongoClient
from getpass import getpass

# ---- Mongo Setup ----
client = MongoClient("mongodb://localhost:27017")
db = client["chatbot"]
users_col = db["users"]
chats_col = db["chats"]

# ---- Hardcoded users (insert once) ----
def seed_users():
    if users_col.count_documents({}) == 0:
        users_col.insert_many([
            {"username": "amna", "password": "1123"},
            {"username": "zahra", "password": "abcd"}
        ])
        print("Users inserted!")

# ---- Login function ----
def login():
    username = input("Enter username: ")
    password = getpass("Enter password: ")

    user = users_col.find_one({"username": username, "password": password})
    if user:
        print(f"✅ Welcome {username}")
        return username
    else:
        print("❌ Invalid credentials")
        return None

# ---- Save chat ----
def save_message(username, role, content):
    chats_col.insert_one({
        "username": username,
        "role": role,          # "user" or "assistant"
        "content": content
    })

# ---- Get chat history ----
def get_history(username):
    messages = list(chats_col.find({"username": username}))
    return messages

# ---- Demo loop ----
if __name__ == "__main__":
    seed_users()
    user = None
    while not user:
        user = login()

    while True:
        query = input(f"\n{user}: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Save user message
        save_message(user, "user", query)

        # Fake response (replace with your chatbot later)
        if "lahore" in query.lower():
            response = "You live in Pakistan."
        elif "where do i live" in query.lower():
            history = get_history(user)
            found = None
            for msg in history:
                if "lahore" in msg["content"].lower():
                    found = "You mentioned you live in Lahore, Pakistan."
                    break
            response = found if found else "I don’t know, you never told me."
        else:
            response = "Hmm, tell me more."

        # Save assistant message
        save_message(user, "assistant", response)
        print(f"AI: {response}")
