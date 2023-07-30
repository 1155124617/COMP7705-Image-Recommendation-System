import openai
import json

original_prompt = "Could you please autocomplete the following words or sentences, in order to make them good image search suggestions? example input: cat, output: a cute photo of cat. Now the new input is to be inputed, what are the outputs? Please suggest 10 outputs. Notice: Please ensure the original input is still in the output. Please make the length of outputs in a increasing order to make it look tidy. Example ratio: Please make first 3 with 4 words limit. Then 3 with 6 words limit. Then 3 with 8 words limit, then 3 with 10 words limit... Also, if the input is not english, please try to return in the correspondiong language. Do not show the word counts. Now start, New input:"

# attention: the api key is private and belongs to Jerry, please prevent the leak of it without permission
openai.api_key = ""


def associate_chatgpt(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a search autocomplete assistant."},
            {"role": "user", "content": original_prompt + text},
        ]
    )
    response_content = response['choices'][0]['message']['content']
    response_content = response_content.split('\n')
    return response_content[0]
