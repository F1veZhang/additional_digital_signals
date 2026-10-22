"""Native-scale state residual adaptation; old sources are imported read-only.

The recurrent encoder, state transition, causal feature masks, optimizer and
curve logging are reused. New behavior is explicit here: native skeleton,
zero residual heads, centered bounded observations and optional search loss.
"""
from pathlib import Path
import copy
import importlib.util
import sys
import threading
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'response_online_offline_20261013'
if str(OLD/'code') not in sys.path:
    sys.path.insert(0, str(OLD/'code'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import common_online as common
from online_features import build_features

_spec = importlib.util.spec_from_file_location('refined_readonly_legacy_state', OLD/'code/online_state_model.py')
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
_LegacyCore = legacy.OnlineStateSpace
_legacy_skeleton = legacy._fit_skeleton
_legacy_loss = legacy.balanced_loss
_fit_lock = threading.RLock()

DEFAULT_CONFIG = dict(legacy.DEFAULT_CONFIG, zero_residual_initialization=True,
    native_skeleton=True, search_softplus_temperature=.1, asinh_mixture=.5,
    allow_zero_checkpoint=True)


def bounded_native_residual(base, delta, lower, upper):
    """Exact legal base at delta=0; derivative 1 in the interior, bounded tails."""
    base = base.clamp(lower, upper)
    above = upper-base
    below = base-lower
    eps = torch.finfo(base.dtype).eps
    step = torch.where(delta >= 0, above*torch.tanh(delta/above.clamp_min(eps)),
        below*torch.tanh(delta/below.clamp_min(eps)))
    # The final min/max only enforces the analytic bounds against roundoff.
    return (base+step).clamp(lower, upper)


def native_observation(base, delta, specs, target_sd, temperature=.1):
    outputs=[]
    for k, spec in enumerate(specs):
        if spec == 'percent':
            y=bounded_native_residual(base[...,k], delta[...,k], 0., 100.)
        elif spec == 'identity':
            # The only identity target in these registered bundles is HFGI stat.
            y=bounded_native_residual(base[...,k], delta[...,k], -1., 1.)
        elif spec == 'nonnegative':
            tau=float(temperature)*target_sd[:,None,k]
            y=tau*F.softplus((base[...,k]+delta[...,k])/tau)
        else:
            raise ValueError('Unregistered target specification: '+str(spec))
        outputs.append(y)
    return torch.stack(outputs, dim=-1)


class RefinedStateSpace(nn.Module):
    def __init__(self,n_regions,n_inputs,n_targets,n_background,config,adjacency,
                 public_input_indices,specs,target_mu,target_sd,link_mu,link_sd,block):
        super().__init__()
        self.cfg=dict(config);self.specs=list(specs);self.block=block
        zeros=np.zeros_like(target_mu);ones=np.ones_like(target_sd)
        # Reuse the same recurrent state core, decoding its private heads as
        # standardized native residuals rather than link-scale outcomes.
        self.core=_LegacyCore(n_regions,n_inputs,n_targets,n_background,config,
            adjacency,public_input_indices,['identity']*n_targets,zeros,ones,zeros,ones,block)
        for head in self.core.heads:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        nn.init.zeros_(self.core.region_offset)
        for name,value in [('target_mu',target_mu),('target_sd',target_sd),
                           ('skeleton_mu',link_mu),('skeleton_sd',link_sd)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))

    def forward(self,input_window,observed_window,calendar_window,rollout_calendar,
                background_window,background_observed_window,skeleton,memory_at_origin=None,return_native=False):
        residual_z=self.core(input_window,observed_window,calendar_window,rollout_calendar,
            background_window,background_observed_window,torch.zeros_like(skeleton),memory_at_origin,True)
        base=skeleton*self.skeleton_sd[:,None,:]+self.skeleton_mu[:,None,:]
        delta=residual_z*self.target_sd[:,None,:]
        native=native_observation(base,delta,self.specs,self.target_sd,self.cfg.get('search_softplus_temperature',.1))
        return native if return_native else (native-self.target_mu[:,None,:])/self.target_sd[:,None,:]

    def region_regularization(self):
        return self.core.region_regularization()


def _native_skeleton(b,features,tr,target_sd,cfg):
    native_bundle=dict(b,target_specs=['identity']*len(b['target_specs']))
    return _legacy_skeleton(native_bundle,features,tr,target_sd,cfg)


legacy.OnlineStateSpace=RefinedStateSpace
legacy._fit_skeleton=_native_skeleton


def _mixed_loss_factory(b,tr,mu,sd,alpha):
    search=[i for i,s in enumerate(b['target_specs']) if s=='nonnegative']
    aux_sd=np.ones_like(sd)
    for k in search:
        y=np.where(b['valid'][:,tr,k],b['targets'][:,tr,k],np.nan)
        v=np.nanstd(np.arcsinh(y/sd[:,None,k]),axis=1)
        aux_sd[:,k]=np.where(np.isfinite(v)&(v>1e-6),v,1.)
    def loss(pred_z,target_z,mask,valid,weights,kind='mae',delta=1.,student_df=4.):
        if kind!='mixed_asinh' or not search:
            return _legacy_loss(pred_z,target_z,mask,valid,weights,'mae' if kind=='mixed_asinh' else kind,delta,student_df)
        usable=torch.isfinite(target_z)&valid&mask[None,:,None]
        truth=torch.nan_to_num(target_z,nan=0.)
        error=(pred_z-truth).abs()
        tm=torch.as_tensor(mu,dtype=pred_z.dtype,device=pred_z.device)[:,None]
        ts=torch.as_tensor(sd,dtype=pred_z.dtype,device=pred_z.device)[:,None]
        aux=torch.as_tensor(aux_sd,dtype=pred_z.dtype,device=pred_z.device)[:,None]
        pred_native=pred_z*ts+tm;truth_native=truth*ts+tm
        transformed=(torch.asinh(pred_native/ts)-torch.asinh(truth_native/ts)).abs()/aux
        use=torch.zeros(pred_z.shape[-1],dtype=torch.bool,device=pred_z.device);use[search]=True
        error=torch.where(use[None,None],float(alpha)*error+(1-float(alpha))*transformed,error)
        count=usable.sum(dim=1);cells=(error*usable).sum(dim=1)/count.clamp_min(1)
        parts=torch.stack([cells[count[:,k]>0,k].mean() for k in range(pred_z.shape[-1])])
        return (parts*weights).sum(),parts
    return loss,aux_sd


def _skeleton_native(b,fit):
    cfg=fit['diagnostics']['config'];h=fit['diagnostics']['horizon'];block=fit['diagnostics']['block']
    features=build_features(b,None,block,h,int(cfg['lag']),float(cfg['memory_halflife']),preprocessing=fit['feature_preprocessing'])
    z=legacy._skeleton_predict(features,fit['skeleton_coefficients'],fit['skeleton_metadata'])
    raw=z*np.asarray(fit['link_sd'])[:,None]+np.asarray(fit['link_mu'])[:,None]
    with torch.no_grad():
        native=native_observation(torch.as_tensor(raw,dtype=torch.float32),torch.zeros_like(torch.as_tensor(raw,dtype=torch.float32)),
            b['target_specs'],torch.as_tensor(fit['sd'],dtype=torch.float32),cfg['search_softplus_temperature']).numpy().astype(float)
    return raw,native


def fit_predict(b,train_mask,val_mask,predict_mask,config,seed,block,horizon=None):
    cfg=dict(DEFAULT_CONFIG);cfg.update(config or {})
    # The bounded identity mapping is authorized only for HFGI anxiety stat.
    for name,spec in zip(b['target_names'],b['target_specs']):
        if spec=='identity' and name!='anxiety':
            raise ValueError('Bounded identity requires the registered HFGI anxiety target')
    for k,spec in enumerate(b['target_specs']):
        values=np.asarray(b['targets'])[...,k][np.asarray(b['valid'])[...,k]]
        lower,upper=(0.,100.) if spec=='percent' else ((-1.,1.) if spec=='identity' else (0.,np.inf))
        if not np.isfinite(values).all() or (values<lower-1e-5).any() or (values>upper+1e-5).any():
            raise ValueError('Observed target outside its registered native support')
    tr=np.asarray(train_mask,bool);va=np.zeros_like(tr) if val_mask is None else np.asarray(val_mask,bool)
    pr=np.asarray(predict_mask,bool)
    if cfg.get('fixed_epochs')==0:
        cfg['baseline_only']=True;cfg['fixed_epochs']=1;cfg['fixed_lr_schedule']=None
    z,mu,sd=common.scale_targets(b,tr)
    loss,aux_sd=_mixed_loss_factory(b,tr,mu,sd,cfg['asinh_mixture'])
    with _fit_lock:
        previous=legacy.balanced_loss
        try:
            legacy.balanced_loss=loss
            fit=legacy.fit_predict(b,tr,va,pr|va,cfg,seed,block,horizon)
        finally:
            legacy.balanced_loss=previous
    diag=fit['diagnostics'];diag['model_family']='refined_state';diag['native_skeleton']=True
    diag['hfg_stat_bounded']=True;diag['zero_residual_initialization']=True
    diag['selected_zero_residual_checkpoint']=bool(cfg.get('baseline_only',False))
    diag['asinh_training_sd']=aux_sd.tolist()
    if not diag['success']:
        return fit
    raw,base=_skeleton_native(b,fit)
    diag['skeleton_head_projection']=[]
    for k,name in enumerate(b['target_names']):
        valid=b['valid'][...,k]&pr[None]
        change=np.abs(raw[...,k]-base[...,k])[valid]
        diag['skeleton_head_projection'].append(dict(target=name,n=int(valid.sum()),
            maximum_native_difference=float(change.max()) if len(change) else None,
            mean_native_difference=float(change.mean()) if len(change) else None))
    base_z=(base-mu[:,None])/sd[:,None]
    if va.any() and cfg['allow_zero_checkpoint']:
        initial=common.score(base_z,z,b,va);trained=common.score(fit['pred_z'],z,b,va)
        diag['epoch0_validation_mae']=initial;diag['trained_selected_validation_mae']=trained
        keep_initial=np.isfinite(initial) and initial<=trained+1e-7
        if keep_initial:
            for key,value in fit['weights'].items():
                if key.startswith('core.heads.') or key=='core.region_offset':value.zero_()
            if fit.get('model') is not None:fit['model'].load_state_dict(fit['weights'])
            pred=np.full_like(z,np.nan,dtype=float);pred[:,pr]=base_z[:,pr]
            fit.update(pred_z=pred,pred=pred,native=pred*sd[:,None]+mu[:,None],selected_epochs=0)
            diag['selected_epochs']=0;diag['best_validation_mae']=initial
            diag['selected_zero_residual_checkpoint']=True
    if cfg.get('baseline_only'):
        fit['selected_epochs']=0;diag['selected_epochs']=0
    for key in ['pred_z','pred','native']:
        fit[key][:,~pr]=np.nan
    fit['metadata']['observation_model']='native-centered-bounded-residual-v1'
    fit['metadata']['hfgi_bounds_source']='analysis_v6_hfgi_state_month_20261003/dictionary/HFGI五维州月字典与支持核验_v6.md'
    return fit


def predict_fitted(b,fitted,predict_mask=None,device=None):
    """Fixed checkpoint replay; returns training-standardized native predictions."""
    return legacy.predict_fitted(b,fitted,predict_mask,device)


def predict_skeleton(b,fitted,predict_mask=None):
    _,native=_skeleton_native(b,fitted)
    z=(native-np.asarray(fitted['mu'])[:,None])/np.asarray(fitted['sd'])[:,None]
    if predict_mask is not None:z[:,~np.asarray(predict_mask,bool)]=np.nan
    return z


def load_checkpoint(path):
    """Load a trusted local checkpoint produced by this project."""
    return torch.load(path,map_location='cpu',weights_only=False)
