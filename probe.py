import json
import urllib.request

def probe(url, messages, api_key="EMPTY"):
    payload = {
        "model": "qwen3.6-35b-a3b-fp8",
        "messages": messages,
        "temperature": 0
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read().decode('utf-8'))
            choice = res['choices'][0]['message']
            content = choice.get('content', '')
            reasoning = choice.get('reasoning_content', '')
            return {"content": content, "reasoning": reasoning}
    except Exception as e:
        return {"error": str(e)}

endpoints = ["http://127.0.0.1:8050/v1/chat/completions", "http://127.0.0.1:31080/v1/chat/completions"]
cases = {
    "A": [
        {"role": "assistant", "content": "84729163058472916305", "reasoning_content": "The second number is 39105847263910584726."},
        {"role": "user", "content": "What is the second number?"}
    ],
    "B": [
        {"role": "assistant", "content": "84729163058472916305"},
        {"role": "user", "content": "What is the second number?"}
    ]
}

results = {}
for url in endpoints:
    port = url.split(':')[-1].split('/')[0]
    for case_label, messages in cases.items():
        key = f"{port}-{case_label}"
        results[key] = probe(url, messages)

print(json.dumps(results, indent=2))
