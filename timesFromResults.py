import re
import pandas as pd

targetFile = 'results/zeroshot/output007.txt'
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

df.to_excel('times.xlsx', index=False)