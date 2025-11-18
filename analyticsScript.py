
import difflib
import glob
import re
import pandas as pd


targetFile = 'results/rag/output003.txt'


# ======================================================================================
# =================== Diff to compare JSON structure compliance ========================
# ======================================================================================

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

with open(targetFile, "r", encoding="utf-8") as f:
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

print()
# ======================================================================================
# ====================== Diff for field "original_document" ============================
# ======================================================================================

# Load the doctors notes which function as keys
keyFiles = sorted(glob.glob('doctorNotes/*.txt'))
keys = {fn: open(fn).read() for fn in keyFiles}

# Load target outputxxx.txt file
with open(targetFile, 'r') as f:
    targetText = f.read()

modelPattern = r'Starting query using model\s+(.*?)\s+please wait\.\.\.(.*?)Time:'
modelBlocks = re.findall(modelPattern, targetText, flags=re.DOTALL)

docPattern = r'"original_document":\s"(.*?)",'

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

results = []

for modelName, blockText in modelBlocks:

    # find all document blocks inside this model output
    documents = re.findall(docPattern, blockText, flags=re.DOTALL)
    if not documents:
        results.append((modelName, 'Model Failed JSON gen', 0, 0))

    for doc in documents:

        bestKey = "No match"
        bestScore = 0

        for kname, ktext in keys.items():
            score = similarity(doc, ktext)
            if score > bestScore:
                bestScore = score
                bestKey = kname

        errorRate = 1 - bestScore

        # save: Model | Key | Similarity | Error Rate
        results.append((modelName, bestKey, bestScore, errorRate))


print('Model           | Matched Key           | Similarity | Error Rate')
print('--------------------------------------------------')
for model, key, sim, err in results:
    print(f'{model:15} | {key:21} | {sim:.3f}     | {err:.3f}')


# Sending the output to excel files for mapping


df = pd.DataFrame(results, columns=["Model", "MatchedKey", "Similarity", "ErrorRate"])
df.to_excel('resultsOrgRecDiff.xlsx', index=False)


# ======================================================================================
# ====================== Extract and compile performance times =========================
# ======================================================================================

with open(targetFile) as f:
    text = f.read()

# Regex pattern to capture model name and time
pattern = re.compile(
    r"Starting query using model (.+?) please wait\.\.\..*?Time:\s*([-\d.]+)\s*seconds",
    re.DOTALL
)

# Find all matches
matches = pattern.findall(text)

# Print header
print(f"{'Model':<12} | {'Time'}")
print("-" * 20)

data = []

# Print each match with absolute time
for model, time_str in matches:
    time_val = abs(float(time_str))
    print(f"{model:<18} | {time_val:.2f}")
    data.append({"Model": model, "Time": round(time_val, 2)})

df = pd.DataFrame(data)

df.to_excel('resultsTimes.xlsx', index=False)