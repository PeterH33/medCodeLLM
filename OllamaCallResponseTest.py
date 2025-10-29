# This initial code was using http.client, just to test things and see what efforts would be needed to stream responses.
# import http.client
# import json

# conn = http.client.HTTPConnection("localhost", 11434)

# prompt = json.dumps({
#     "model": "deepseek-r1:8b",
#     "prompt": "Rephrase Hello World and respond."
# })

# headers = {"Content-Type": "application/json"}
# print("Called ollama")
# conn.request("POST", "/api/generate", body=prompt, headers=headers)
# response = conn.getresponse()

# print("Getting response")
# #This block may take a moment to respond, does not provide live line by line
# for line in response.read().decode().splitlines():
#     data = json.loads(line)
#     if "response" in data:
#         print(data["response"], end="", flush=True)
#     if data.get("done"):
#         break

# conn.close()

# Using requests is much more straightforward and allows for streaming the response of the model from ollama
import requests
import json

url = "http://localhost:11434/api/generate"

prompt = {
    "model": "deepseek-r1:8b",
    "prompt": "Rephrase Hello World and respond."
}

response = requests.post(url, json=prompt, stream=True)

# Stream output clean
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8"))
        if "response" in data:
            print(data["response"], end="", flush=True)
        if data.get("done", False):
            break

# Firehose output, at conclusion there is an interesting field "context" that comes with a fairly large vector
# for chunk in response.iter_content(chunk_size=None):
#     if chunk:
#         print(chunk.decode("utf-8"), end="", flush=True)

print("File end")