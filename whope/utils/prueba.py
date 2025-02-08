import os

print("Listing files in current directory:", os.listdir("."))

with open("nlm_rules.pip", "r") as f:
    lines = f.readlines()
    for line in lines:
        print(line)
