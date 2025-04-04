import json
import re
import os

# Path to the folder containing transcript JSON files
file_dir = "transcripts"

# List to store all conversation data
conversation_data = []

# Function to convert timestamps to seconds
def tStamps(time_str):
    match = re.match(r"(\d+):(\d+)", time_str)
    if match:
        minutes, seconds = map(int, match.groups())
        return minutes * 60 + seconds
    return None

# Loop through all JSON files in the directory
for filename in os.listdir(file_dir):
    file_path = os.path.join(file_dir, filename)
    if os.path.isfile(file_path) and filename.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)


        # Extract timestamps and dialogues
        timestamps = [tStamps(raw_data[i]) for i in range(0, len(raw_data), 2)]
        dialogues = [raw_data[i] for i in range(1, len(raw_data), 2)]

        # Calculate the average gap between timestamps
        time_gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps)) if timestamps[i] and timestamps[i - 1]]
        average_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 5  # Default 5 sec

        # Assign speaker roles
        formatted_data = []
        current_speaker = "host"  # Assume the first speaker is always the host

        for i in range(len(dialogues)):
            if i > 0 and timestamps[i] and timestamps[i - 1]:  # Check timestamps
                if timestamps[i] - timestamps[i - 1] > average_gap:
                    current_speaker = "guest" if current_speaker == "host" else "host"

            formatted_data.append({"role": current_speaker, "content": dialogues[i]})

        # Structure for fine-tuning OpenAI API
        for i in range(len(formatted_data) - 1):
            if formatted_data[i]["role"] == "host" and formatted_data[i + 1]["role"] == "guest":
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are an AI that generates engaging podcast scripts like Joe Rogan."},
                        {"role": "user", "content": f"Host: {formatted_data[i]['content']}"},
                        {"role": "assistant", "content": f"Guest: {formatted_data[i + 1]['content']}"}
                    ]
                }
                conversation_data.append(entry)

# Save processed data to JSONL file
output_file = "fine_tune_data.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in conversation_data:
        f.write(json.dumps(item) + "\n")

print(f"Data processing success! File saved as {output_file}")
