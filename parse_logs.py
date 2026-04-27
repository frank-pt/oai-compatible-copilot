import sys, json, re

with open("/tmp/log_tail.txt", "r") as f:
    content = f.read()

events = re.split(r"=== (.*?) request-preflight ===", content)
results = []
for i in range(1, len(events), 2):
    ts = events[i]
    body = events[i+1]
    
    model_id = re.search(r"modelId:\s*\"([^\"]+)\"", body)
    initiator = re.search(r"requestInitiator:\s*\"([^\"]+)\"", body)
    conv_id = re.search(r"conversationId:\s*\"([^\"]+)\"", body)
    api_mode = re.search(r"apiMode:\s*\"([^\"]+)\"", body)
    
    m_id = model_id.group(1) if model_id else "N/A"
    init = initiator.group(1) if initiator else "N/A"
    c_id = conv_id.group(1) if conv_id else "N/A"
    am = api_mode.group(1) if api_mode else "N/A"
    
    replays = []
    # Match reasoning_content in history/messages
    for match in re.finditer(r"reasoning_content:\s*(?:\"([^\"]*)\"|(null|undefined))", body):
        text = match.group(1)
        lit = match.group(2)
        if lit or not text:
            replays.append("empty")
        else:
            replays.append("non-empty")
            
    results.append((ts, m_id, init, c_id, am, len(replays), str(replays)))

for r in results[-12:]:
    print(" | ".join(map(str, r)))
