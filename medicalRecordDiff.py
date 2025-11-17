""" Script for automatic identification of the error rate of LLM transcription of medical notes into fields in output.
The output of the script is difficult to automatically sort by the LLM that created the output due to the possibility of complete failure of the model to generate code vaguely appropriate.
Examine to output001.txt for an example of such failures as there are only 15 possible matches when there should be 25 in each doc. 
Perhaps the solution is to use a consistent header to seperate out each output in the medCodeLLM script.
"""

import difflib
import glob
import re
import os


while True:
    targetFile = input("Enter the filename inside 'results/' (e.g., zeroshot/output001.txt): ").strip()
    fullPath = os.path.join("results", targetFile)

    if os.path.isfile(fullPath):
        break
    else:
        print(f"ERROR: File not found: {fullPath}\nPlease try again.\n")


# Load the doctors notes which function as keys
keyFiles = sorted(glob.glob('doctorNotes/*.txt'))
keys = {fn: open(fn).read() for fn in keyFiles}

# Load target outputxxx.txt file
with open('results/' + targetFile) as f:
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
import pandas as pd

df = pd.DataFrame(results, columns=["Model", "MatchedKey", "Similarity", "ErrorRate"])
df.to_excel('results.xlsx', index=False)