import requests

url = f"http://127.0.0.1:8080/completion"

prompt = """
### Instruction:
    You are db administrator now,please give me tips to finetune postgresql installed at server with 32GB memory and 16 cores of cpu

### Response:

"""

req_json = {
    "stream": False,
    "n_predict": 400,
    "temperature": 0,
    "stop": [
        "</s>",
    ],
    "repeat_last_n": 256,
    "repeat_penalty": 1,
    "top_k": 20,
    "top_p": 0.75,
    "tfs_z": 1,
    "typical_p": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "mirostat": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "grammar": "",
    "n_probs": 0,
    "prompt": prompt
}

res = requests.post(url, json=req_json)
result = res.json()["content"]
print(result)

