import json
input_path="fine_tune_data.jsonl"
output_path="Unsloth_data.jsonl"

with open(input_path,"r",encoding="utf-8") as infile,open(output_path,"w",encoding="utf-8") as outfile:
    for line in infile:
        item=json.loads(line.strip())
        prompt=f"{item['instruction']}\n\n{item['input']}"
        completion=item['output']
        json.dump({"prompt":prompt,"completion":completion},outfile)
        outfile.write("\n")

print(f"converted dataset saved to {output_path}")