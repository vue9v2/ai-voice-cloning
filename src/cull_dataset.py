import os
import sys

indir = f'./training/{sys.argv[1]}/'
cap = int(sys.argv[2])

if not os.path.isdir(indir):
	raise Exception(f"Invalid directory: {indir}")

if not os.path.exists(f'{indir}/train.txt'):
	raise Exception(f"Missing dataset: {indir}/train.txt")

with open(f'{indir}/train.txt', 'r', encoding="utf-8") as f:
	lines = f.readlines()

validation = []
training = []

for line in lines:
	split = line.split("|")
	filename = split[0]
	text = split[1]

	if len(text) < cap:
		validation.append(line.strip())
	else:
		training.append(line.strip())

with open(f'{indir}/train_culled.txt', 'w', encoding="utf-8") as f:
	f.write("\n".join(training))

with open(f'{indir}/validation.txt', 'w', encoding="utf-8") as f:
	f.write("\n".join(validation))

print(f"Culled {len(validation)} lines")