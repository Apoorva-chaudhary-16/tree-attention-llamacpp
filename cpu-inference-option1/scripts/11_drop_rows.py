import pandas as pd, sys
tag, exp, model, quant, depth = sys.argv[1:6]
p = f"results/bench_{tag}.csv"; d = pd.read_csv(p)
bad = (d.exp == exp) & (d.model == model) & (d.quant == quant) & (d.sweep == "decode") & (d.depth == int(depth))
print("removing", int(bad.sum()), "rows"); d[~bad].to_csv(p, index=False)
