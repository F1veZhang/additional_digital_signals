"""Reuse frozen finite-search mechanics, with new fit and exact native ranks."""
from pathlib import Path
import importlib.util,sys
sys.dont_write_bytecode=True
import numpy as np
import refined_state_model as model
spec=importlib.util.spec_from_file_location('refined_readonly_search',model.OLD/'code/online_search.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
legacy.fit_predict=model.fit_predict
save_json=legacy.save_json
append_jsonl=legacy.append_jsonl
config_key=legacy.config_key
materialize_config=legacy.materialize_config
calibration_from_entries=legacy.calibration_from_entries

def finite_search(b,inner_masks,settings,block,horizon,out,signature,deadline=None):
    selected=legacy.finite_search(b,inner_masks,settings,block,horizon,out,signature,deadline)
    if selected['success']:
        epochs=max(0,int(np.rint(np.median([e['selected_epochs'] for e in selected['winner']['entries']]))))
        selected['selected_epochs']=epochs
        selected['fixed_lr_schedule']=legacy.selected_schedule(selected['winner']['entries'],out,epochs) if epochs else []
        selected['selected_zero_residual_checkpoint']=epochs==0
        selected['epoch0_comparison_after_actual_candidate_training']=True
    return selected

def quantile_map(rows,coverage=.9):
    import math
    import pandas as pd
    frame=pd.DataFrame(rows);result={}
    if frame.empty:return result
    for (region,target),g in frame.groupby(['region','target']):
        a=np.sort(g.absolute_residual_native.to_numpy(float))
        if not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid calibration residual')
        rank=min(len(a),math.ceil((len(a)+1)*coverage))
        result[(str(region),str(target))]=(float(a[rank-1]),len(a))
    return result
