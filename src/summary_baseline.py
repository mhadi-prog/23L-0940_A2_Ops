import pandas as pd
from models import build_baseline_cnn

model = build_baseline_cnn()
model.summary()

rows = [{"layer": l.name, "type": l.__class__.__name__,
         "output_shape": str(l.output.shape), "params": l.count_params()}
        for l in model.layers]
df = pd.DataFrame(rows)
df.to_csv("reports/baseline_layers.csv", index=False)
print(df.to_string(index=False))
print("Total params:", model.count_params())