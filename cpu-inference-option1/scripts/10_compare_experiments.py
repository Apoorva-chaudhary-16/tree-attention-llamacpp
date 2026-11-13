import pandas as pd, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "laptop"
d = pd.read_csv(f"results/bench_{tag}.csv")
d = d[(d.model == "llama-3.2-1b") & (d.quant == "Q4_K_M") & (d.sweep == "decode") & (d.threads == 8) & (d.status == "ok")]
p = d.pivot_table(index="depth", columns="exp", values="ts_mean")
p = p[[c for c in ["baseline", "fa_off", "kv_q8"] if c in p]]
for c in ["fa_off", "kv_q8"]:
    p[c + "_gain_%"] = (p[c] / p["baseline"] - 1) * 100
print(p.round(2))
s = d[d.exp.isin(["baseline", "kv_q8"]) & d.depth.isin([512, 1024])][["exp", "depth", "samples"]]
print(s.to_string(index=False))
