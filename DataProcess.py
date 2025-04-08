import json
import re
import os


file_dir = "transcripts"
# List to store all conversation data
all_conversation_data = []

def process_transcript(transcript):
    time_text_pairs = []
    for i in range(0, len(transcript), 2):
        if i+1 < len(transcript):
            time_text_pairs.append((transcript[i], transcript[i+1]))
    
    # Now detect speaker changes based on context and language patterns
    conversations = []
    current_speaker = "host"  # Assume starting with host
    current_utterance = ""
    
    # Common speaker change indicators
    speaker_change_patterns = [
        r"\byeah\b.*\bbut\b", 
        r"\babsolutely\b", 
        r"\bI mean\b",
        r"\byou know\b",
        r"\blike\b.*\bI\b"
    ]
    
    for _, text in time_text_pairs:
        change_detected = False
        
        # Look for explicit dialogue indicators or question-response patterns
        if re.search(r"\?", current_utterance) and not re.search(r"\?", text[:20]):
            change_detected = True
        
        # Look for speaker change patterns
        for pattern in speaker_change_patterns:
            if re.search(pattern, text[:30]):  # Check start of new segment
                change_detected = True
                break
        
        # If we detect a likely speaker change
        if change_detected and current_utterance:
            conversations.append({"role": current_speaker, "content": current_utterance.strip()})
            current_speaker = "guest" if current_speaker == "host" else "host"
            current_utterance = text
        else:
            current_utterance += " " + text
    
    # Add the final utterance
    if current_utterance:
        conversations.append({"role": current_speaker, "content": current_utterance.strip()})
    
    return conversations

# Loop through all JSON files in the directory
for filename in os.listdir(file_dir):
    file_path = os.path.join(file_dir, filename)
    if os.path.isfile(file_path) and filename.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
        
        # Process the transcript to get conversational turns
        conversations = process_transcript(raw_data)
        
        # Create prompt-response pairs
        for i in range(len(conversations) - 1):
            if conversations[i]["role"] == "host" and conversations[i + 1]["role"] == "guest":
                host_text = conversations[i]["content"]
                guest_text = conversations[i + 1]["content"]
                
                # Filter out very short utterances
                if len(host_text.split()) >= 10 and len(guest_text.split()) >= 10:
                    all_conversation_data.append({
                        "instruction": "Generate a podcast guest response in the style of Joe Rogan's guests",
                        "input": host_text,
                        "output": guest_text
                    })

# Save processed data to JSONL file
output_file = "fine_tune_data.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in all_conversation_data:
        f.write(json.dumps(item) + "\n")

print(f"Data processing success! File saved as {output_file}")
print(f"Total examples generated: {len(all_conversation_data)}")