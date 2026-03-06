
path = 'data/raw/english/validated.tsv'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific merge issue
content = content.replace('std1089', 'std\n1089')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Fixed TSV file.")
