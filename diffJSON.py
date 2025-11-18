"""This script verifies whether or not models are complying with providing output as a specifically formated JSON structure. After the introduction of the Ollama structured output system compliance was at 100% or 0% for each model depending on output capabilities."""
import re


filepath = 'results/zeroshot/output009.txt'


# Regex to match the JSON block with any content in the fields
json_pattern = re.compile(
    r'\{\s*'
    # r'"medical_record"\s*:\s*\{\s*'
    r'"original_document"\s*:\s*(?:".*?"|\[.*?\]|\{.*?\}|[^,]+)\s*,\s*'
    r'"diagnostic_codes"\s*:\s*(?:\[.*?\]|".*?"|\{.*?\}|[^,]+)\s*,\s*'
    r'"diagnoses"\s*:\s*(?:\[.*?\]|".*?"|\{.*?\}|[^,]+)\s*,?\s*'
    # r'\}\s*'
    r'\}',
    re.DOTALL
)

# Regex to match blocks between the markers
block_pattern = re.compile(
    r'(</think>|please wait\.\.\.)\s*(.*?)\s*Time to completion',
    re.DOTALL
)

# Regex to remove content inside <think>...</think> tags
think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

blocks = list(block_pattern.finditer(content))

if not blocks:
    print("No blocks found between the specified markers.")
else:
    for b_idx, block_match in enumerate(blocks, start=1):
        block_start_pos = block_match.start(2)
        block_text = block_match.group(2).strip()

        # Remove <think>...</think> content
        cleaned_text = think_pattern.sub("", block_text).strip()

        # Find JSON blocks
        json_matches = list(json_pattern.finditer(cleaned_text))

        # Remove JSON to see if anything else remains
        remaining_text = json_pattern.sub("", cleaned_text).strip()

        line_number = content.count("\n", 0, block_start_pos) + 1

        if len(json_matches) == 1 and remaining_text == "":
            print(f"Block {b_idx} at line {line_number}: ✅ only JSON found")
        else:
            print(f"Block {b_idx} at line {line_number}: ❌ unexpected content or multiple JSONs")
            if len(json_matches) != 1:
                print(f"  - JSON structures found: {len(json_matches)}")
            if remaining_text != "":
                print(f"  - Other content found outside <think> tags:\n{remaining_text}\n")
