import requests

# the token is for huggingface t5-base-finetuned-common_gen
API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/mrm8488/t5-base-finetuned-common_gen"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

search_keyword = input("Enter a search keyword: ")

# you can change the keywords below, to get better auto-completion results
# the model will automatically make the sentence grammatically correct
output = query({
	"inputs": "picture photo type kind surrounding landscape" + search_keyword,
})

# pirnt the output
print(output)
