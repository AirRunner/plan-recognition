import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


with open("results.json", 'r') as json_results:
    results = json.load(json_results)

df = pd.DataFrame(results)

fig = plt.figure(figsize=(10, 6))

sns.set_theme(style='darkgrid')
sns.lineplot(data=df)

plt.savefig("models_results.png")
plt.show()
