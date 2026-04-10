#test_stream.py

import requests
import json
import time
import uuid
SESSION_ID = str(uuid.uuid4())

URL = "http://127.0.0.1:8000/chat_stream"


def chat():
    print(". HTTP Streaming Client Started")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            response = requests.post(
                URL,
                json={
                    "user_input": user_input,
                    "session_id": SESSION_ID
                },
                stream=True
            )

            print("Bot: ", end="", flush=True)

            buffer = ""  # . word buffer

            for line in response.iter_lines():
                if not line:
                    continue

                decoded = line.decode()

                if decoded.startswith("data: "):
                    content = decoded.replace("data: ", "")

                    # . END STREAM
                    if content == "[DONE]":
                        if buffer:
                            print(buffer, end="", flush=True)
                        print("\n")
                        break

                    # . FINAL METADATA
                    if content.startswith('{"type": "final"'):
                        data = json.loads(content)

                        print("\n. State:", data["state"])
                        print("🎯 Reward:", data["reward"])
                        print(". Agent:", data["agent"])
                        continue

                    # . CLEAN EMPTY TOKENS
                    if not content.strip():
                        continue

                    # . BUFFER LOGIC (fix broken words)
                    buffer += content

                    # print only on word boundary
                    if content.endswith((" ", ".", ",", "?", "!", "\n")):
                        print(buffer, end="", flush=True)
                        buffer = ""

                    # . small delay → smooth typing feel
                    time.sleep(0.01)

        except Exception as e:
            print(". Error:", e)


if __name__ == "__main__":
    chat()