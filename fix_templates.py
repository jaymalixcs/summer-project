import os
import re

d = 'templates'
# Match any data-chart-json='{{ anything }}' without | safe
pattern = re.compile(r"data-chart-json='{{ ([^|}']+?) }}'")
replacement = lambda m: "data-chart-json='{{ " + m.group(1).strip() + " | safe }}'"

for fname in sorted(os.listdir(d)):
    if not fname.endswith('.html'):
        continue
    path = os.path.join(d, fname)
    content = open(path, encoding='utf-8').read()
    matches = pattern.findall(content)
    if matches:
        print(f"{fname}: needs fixing -> {matches}")
        new_content = pattern.sub(replacement, content)
        open(path, 'w', encoding='utf-8').write(new_content)
    else:
        print(f"{fname}: ok")
