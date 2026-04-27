import json
import re
import sys

def parse_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the latest request-preflight block
    # Matches: === 2026-04-27T04:24:46.824Z request-preflight ===
    marker_pattern = r'=== .*? request-preflight ==='
    preflight_headers = list(re.finditer(marker_pattern, content))
    
    if not preflight_headers:
        print("No request-preflight block found.")
        return

    latest_header = preflight_headers[-1]
    start_pos = latest_header.end()
    
    # The JSON starts after the header and ends at the next marker or EOF
    next_marker = re.search(r'=== .*? ===', content[start_pos:])
    if next_marker:
        preflight_json_str = content[start_pos : start_pos + next_marker.start()].strip()
    else:
        preflight_json_str = content[start_pos:].strip()

    try:
        preflight_data = json.loads(preflight_json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding preflight JSON: {e}")
        # Try to find the closing brace index
        last_brace = preflight_json_str.rfind('}')
        if last_brace != -1:
             try:
                 preflight_data = json.loads(preflight_json_str[:last_brace+1])
             except:
                 return
        else:
            return

    # Look for the completion/response block
    # Since grep failed, maybe it's "completion" or similar
    comp_header_match = re.search(r'=== .*? (completion|response) ===', content[start_pos:])
    completion_data = None
    if comp_header_match:
        comp_start = start_pos + comp_header_match.end()
        # Find next marker or EOF
        next_m = re.search(r'=== .*? ===', content[comp_start:])
        if next_m:
            comp_json_str = content[comp_start : comp_start + next_m.start()].strip()
        else:
            comp_json_str = content[comp_start:].strip()
        
        try:
            completion_data = json.loads(comp_json_str)
        except:
            pass

    # Extract info
    model = preflight_data.get('model') or preflight_data.get('modelId')
    api_mode = preflight_data.get('apiMode')
    
    # Original Messages
    orig_messages = preflight_data.get('originalMessages', [])
    last_user_msg = ""
    for msg in reversed(orig_messages):
        if msg.get('role') == 'user':
            last_user_msg = msg.get('text', '')
            break
            
    orig_assistant = [{"text": m.get('text'), "thinking": m.get('thinking')} for m in orig_messages if m.get('role') == 'assistant']
    
    # Restored Messages
    rest_messages = preflight_data.get('restoredMessages', [])
    rest_assistant = [{"text": m.get('text'), "thinking": m.get('thinking')} for m in rest_messages if m.get('role') == 'assistant']
    
    # OpenAI Messages
    openai_messages = preflight_data.get('openaiMessages') or preflight_data.get('requestBody', {}).get('messages', [])
    openai_assistant = [{"content": m.get('content'), "reasoning_content": m.get('reasoning_content')} for m in openai_messages if m.get('role') == 'assistant']
    
    # Completion
    comp_content = None
    comp_reasoning = None
    if completion_data:
        choices = completion_data.get('choices', [])
        if choices:
            msg = choices[0].get('message', {})
            comp_content = msg.get('content')
            comp_reasoning = msg.get('reasoning_content')

    # Booleans
    any_restored_thinking = any(m.get('thinking') is not None and m.get('thinking') != "" for m in rest_assistant)
    any_request_reasoning = any(m.get('reasoning_content') is not None and m.get('reasoning_content') != "" for m in openai_assistant)

    print(f"Model/ApiMode: {model} / {api_mode}")
    print(f"Last User Message: {last_user_msg!r}")
    print(f"Original Assistant: {json.dumps(orig_assistant)}")
    print(f"Restored Assistant: {json.dumps(rest_assistant)}")
    print(f"OpenAI/RequestBody Assistant: {json.dumps(openai_assistant)}")
    print(f"Final Completion: content={json.dumps(comp_content)}, reasoning_content={json.dumps(comp_reasoning)}")
    print(f"Any restored thinking: {any_restored_thinking}")
    print(f"Any request reasoning_content: {any_request_reasoning}")

if __name__ == "__main__":
    parse_log('/tmp/oaicopilot-deepseek-debug.log')
