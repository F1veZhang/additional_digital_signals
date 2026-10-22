"""Frozen per-target channel policy for the approved multi-platform round."""
def allowed_inputs(b, block="online"):
    names = list(b["input_names"])
    official = [i for i,n in enumerate(names) if n.startswith("official")]
    chosen = set(b.get("selected_input_names", []))
    if block == "base":
        chosen = set()
    if not chosen.issubset(names):
        raise ValueError("Unknown selected input: "+str(chosen-set(names)))
    out=[]
    for k in range(len(b["target_names"])):
        own=b["target_input_indices"][k]
        active=set(official)
        if k not in b["official_indices"]:
            active.add(own)
        active.update(i for i,n in enumerate(names) if n in chosen)
        out.append(sorted(active))
    return out

