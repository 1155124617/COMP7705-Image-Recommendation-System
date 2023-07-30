import requests

# the token is for huggingface t5-base-finetuned-common_gen
API_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/mrm8488/t5-base-finetuned-common_gen"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def associate_landscape(original_text):
    output = query({
        "inputs": "picture photo type kind surrounding landscape" + original_text,
    })

    return output[0]['generated_text']

