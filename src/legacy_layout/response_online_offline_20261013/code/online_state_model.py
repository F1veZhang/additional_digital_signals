"""Past-window neural observation state model for native dual outcomes.

This module never loads research data. Inputs and outputs are separate: a search
signal can be an input without being labelled or trained as an emotion outcome.
The same encoder and transition parameters are shared across target-specific
masked state trajectories in every information block. Native nonnegative heads
use scale * softplus, never exponentiation of a predicted log response.
"""
from __future__ import annotations
import copy
import os
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CONFIG = dict(context=24, state_dim=8, hidden=16, layers=1,
    region_dim=2, region_shrink=.1, region_offset_shrink=1., residual_scale=.5,
    lag=6, memory_halflife=6., ridge=10., region_penalty=100.,
    lr=.003, wd=.001, loss="huber", huber_delta=1., student_df=4.,
    training_official_weight=.5, epochs=240, max_epochs=720,
    extension_epochs=240, extension_window=40, extension_relative_improvement=.002,
    min_epochs=60, patience=60, lr_patience=20, lr_factor=.5, min_lr=1e-5,
    threads=2, device="cpu", fixed_epochs=None, fixed_lr_schedule=None,
    return_weights=True, return_model=False, baseline_only=False)


def calendar_features(dates):
    dates=pd.DatetimeIndex(dates)
    cycle=(np.asarray(dates.dayofyear,float)-1)/365.2425
    years=np.asarray((dates-dates[0]).days,float)/365.2425
    return np.column_stack([np.ones(len(dates)),np.sin(2*np.pi*cycle),
        np.cos(2*np.pi*cycle),np.sin(4*np.pi*cycle),np.cos(4*np.pi*cycle),years]).astype("float32")


def causal_windows(values, observed, dates, context, horizon, background=None, background_observed=None):
    """Each row t ends at t-h; rollout calendars contain no observed future data."""
    x=np.asarray(values,dtype="float32");m=np.asarray(observed,dtype="float32")
    nr,nt,np_=x.shape;context=int(context);horizon=int(horizon)
    if context<1 or horizon<1 or m.shape!=x.shape:raise ValueError("invalid causal window specification")
    bg=np.empty((nr,nt,0),dtype="float32") if background is None else np.asarray(background,dtype="float32")
    bm=np.ones_like(bg) if background_observed is None else np.asarray(background_observed,dtype="float32")
    if bg.shape[:2]!=(nr,nt) or bm.shape!=bg.shape:raise ValueError("invalid background shape")
    xx=np.zeros((nr,nt,context,np_),dtype="float32");mm=np.zeros_like(xx)
    bb=np.zeros((nr,nt,context,bg.shape[2]),dtype="float32");bo=np.zeros_like(bb)
    cal=calendar_features(dates);cc=np.zeros((nt,context,cal.shape[1]),dtype="float32")
    rc=np.zeros((nt,horizon,cal.shape[1]),dtype="float32")
    for j in range(context):
        shift=horizon+context-1-j
        if shift<nt:
            xx[:,shift:,j]=x[:,:-shift];mm[:,shift:,j]=m[:,:-shift]
            bb[:,shift:,j]=bg[:,:-shift];bo[:,shift:,j]=bm[:,:-shift]
            cc[shift:,j]=cal[:-shift]
    for j in range(horizon):
        shift=horizon-1-j
        if shift==0:rc[:,j]=cal
        elif shift<nt:rc[shift:,j]=cal[:-shift]
    return dict(input_window=xx,observed_window=mm,calendar_window=cc,rollout_calendar=rc,
        background_window=bb,background_observed_window=bo)


def native_to_link(y,specs,output_scale):
    y=np.asarray(y,float);out=y.copy()
    for k,spec in enumerate(specs):
        finite=np.isfinite(y[...,k])
        if spec=="percent":
            if ((y[...,k][finite]<-1e-5)|(y[...,k][finite]>100.+1e-5)).any():raise ValueError("percentage target outside 0..100")
            p=np.clip(y[...,k]/100.,1e-6,1.-1e-6)
            out[...,k]=np.log(p)-np.log1p(-p)
        elif spec=="nonnegative":
            if (y[...,k][finite]<-1e-6).any():raise ValueError("negative nonnegative native target")
            q=np.maximum(y[...,k]/output_scale[:,None,k],1e-6)
            out[...,k]=q+np.log(-np.expm1(-q))
        elif spec!="identity":raise ValueError("unknown native target specification: "+str(spec))
    return out


def native_from_link(link,specs,output_scale):
    out=[]
    for k,spec in enumerate(specs):
        if spec=="percent":v=100.*torch.sigmoid(link[...,k])
        elif spec=="nonnegative":v=output_scale[:,None,k]*F.softplus(link[...,k])
        elif spec=="identity":v=link[...,k]
        else:raise ValueError("unknown output specification")
        out.append(v)
    return torch.stack(out,dim=-1)


class OnlineStateSpace(nn.Module):
    """Shared recurrent dynamics; private masked histories and observation heads.

    A past-only GRU encodes the finite observation window. Its final state is
    advanced by a shared GRUCell for each forecast step using known calendars
    and background carried from the observation origin. No future observation
    or retrospectively inferred phase is supplied to these updates.
    """
    def __init__(self,n_regions,n_inputs,n_targets,n_background,config,adjacency,
                 public_input_indices,specs,target_mu,target_sd,link_mu,link_sd,block):
        super().__init__();self.cfg=config;self.nr=n_regions;self.np=n_inputs;self.nk=n_targets;self.nb=n_background
        self.specs=list(specs);self.block=block
        adj=np.asarray(adjacency,bool)
        if adj.shape!=(n_targets,n_inputs):raise ValueError("adjacency shape mismatch")
        self.register_buffer("adjacency",torch.as_tensor(adj,dtype=torch.float32))
        public=np.zeros(n_inputs,dtype="float32");public[list(public_input_indices)]=1.
        self.register_buffer("public_mask",torch.as_tensor(public))
        self.register_buffer("target_identity",torch.eye(n_targets))
        self.region_embedding=nn.Embedding(n_regions,int(config["region_dim"]))
        nn.init.zeros_(self.region_embedding.weight)
        self.region_offset=nn.Parameter(torch.zeros(n_regions,n_targets))
        hidden=int(config["hidden"]);state=int(config["state_dim"]);layers=int(config["layers"])
        nf=2*n_inputs+6+2*n_background+int(config["region_dim"])+n_targets
        self.input_encoder=nn.Sequential(nn.Linear(nf,hidden),nn.Tanh())
        self.past_encoder=nn.GRU(hidden,state,num_layers=layers,batch_first=True)
        # Dimensions remain identical for all four information blocks. Inactive
        # backgrounds and interactions are zero; they do not remove parameters.
        self.interaction_dim=2*n_background*n_inputs+2*6*n_inputs
        nd=6+2*n_background+int(config["region_dim"])+n_targets+self.interaction_dim
        self.future_encoder=nn.Sequential(nn.Linear(nd,hidden),nn.Tanh())
        self.transition=nn.GRUCell(hidden,state)
        self.heads=nn.ModuleList([nn.Linear(state,1) for _ in range(n_targets)])
        for head in self.heads:
            nn.init.normal_(head.weight,std=.005);nn.init.zeros_(head.bias)
        for key,value in [("target_mu",target_mu),("target_sd",target_sd),
                          ("output_scale",target_sd),("link_mu",link_mu),("link_sd",link_sd)]:
            self.register_buffer(key,torch.as_tensor(value,dtype=torch.float32))

    def forward(self,input_window,observed_window,calendar_window,rollout_calendar,
                background_window,background_observed_window,skeleton,memory_at_origin=None,return_native=False):
        nr,nt,nw,np_=input_window.shape
        if nr!=self.nr or np_!=self.np:raise ValueError("region/input dimensions differ from fitted model")
        emb=self.region_embedding(torch.arange(nr,device=input_window.device))
        embw=emb[:,None,None,:].expand(-1,nt,nw,-1)
        embr=emb[:,None,:].expand(-1,nt,-1)
        calw=calendar_window[None].expand(nr,-1,-1,-1)
        use_bg=float(self.block!="internet_only")
        bgw=background_window*use_bg;bmw=background_observed_window*use_bg
        bg=bgw[:,:,-1];bmo=bmw[:,:,-1]
        results=[]
        for k in range(self.nk):
            allowed=self.adjacency[k]
            xx=input_window*allowed;obs=observed_window*allowed
            ident=self.target_identity[k][None,None,None].expand(nr,nt,nw,-1)
            encoded=self.input_encoder(torch.cat([xx,obs,calw,bgw,bmw,embw,ident],dim=-1))
            _,hs=self.past_encoder(encoded.reshape(nr*nt,nw,-1))
            state=hs[-1]
            identr=self.target_identity[k][None,None].expand(nr,nt,-1)
            recent=xx[:,:,-1]*self.public_mask
            if memory_at_origin is None:
                memory=(xx*obs).sum(dim=2)/obs.sum(dim=2).clamp_min(1.)
            else:memory=memory_at_origin*allowed
            memory=memory*self.public_mask
            for step in range(rollout_calendar.shape[1]):
                cal=rollout_calendar[:,step][None].expand(nr,-1,-1)
                if self.block=="interaction":
                    ip=[(bg[:,:,:,None]*recent[:,:,None,:]).reshape(nr,nt,-1),
                        (bg[:,:,:,None]*memory[:,:,None,:]).reshape(nr,nt,-1),
                        (cal[:,:,:,None]*recent[:,:,None,:]).reshape(nr,nt,-1),
                        (cal[:,:,:,None]*memory[:,:,None,:]).reshape(nr,nt,-1)]
                    interaction=torch.cat(ip,dim=-1)
                else:interaction=input_window.new_zeros(nr,nt,self.interaction_dim)
                future=torch.cat([cal,bg,bmo,embr,identr,interaction],dim=-1)
                state=self.transition(self.future_encoder(future).reshape(nr*nt,-1),state)
            results.append(self.heads[k](state).reshape(nr,nt))
        residual=torch.stack(results,dim=-1)+self.region_offset[:,None,:]
        if self.cfg.get("baseline_only",False):residual=residual*0.
        link=(skeleton+float(self.cfg["residual_scale"])*residual)*self.link_sd[:,None,:]+self.link_mu[:,None,:]
        native=native_from_link(link,self.specs,self.output_scale)
        if return_native:return native
        return (native-self.target_mu[:,None,:])/self.target_sd[:,None,:]

    def region_regularization(self):
        return float(self.cfg["region_shrink"])*self.region_embedding.weight.square().mean()+float(self.cfg["region_offset_shrink"])*self.region_offset.square().mean()


def balanced_loss(pred_z,target_z,mask,valid,weights,kind="mae",delta=1.,student_df=4.):
    usable=torch.isfinite(target_z)&valid&mask[None,:,None]
    diff=pred_z-torch.nan_to_num(target_z,nan=0.)
    if kind=="mae":error=diff.abs()
    elif kind=="huber":error=F.huber_loss(pred_z,torch.nan_to_num(target_z,nan=0.),delta=delta,reduction="none")
    elif kind=="student_t":
        if student_df<=2:raise ValueError("Student-t training score requires df > 2")
        error=.5*(student_df+1)*torch.log1p(diff.square()/student_df)
    else:raise ValueError("unknown training loss")
    count=usable.sum(dim=1);cells=(error*usable).sum(dim=1)/count.clamp_min(1)
    parts=[]
    for k in range(target_z.shape[-1]):
        available=count[:,k]>0
        parts.append(cells[available,k].mean() if available.any() else pred_z.new_tensor(float("nan")))
    parts=torch.stack(parts)
    return (parts*weights).sum(),parts


def _ema_at_origin(values,observed,horizon,halflife):
    nr,nt,np_=values.shape
    result=np.zeros_like(values,dtype="float32");numer=np.zeros((nr,np_));mass=np.zeros_like(numer)
    decay=np.exp(np.log(.5)/float(halflife))
    for origin in range(nt-int(horizon)):
        numer=decay*numer+(1-decay)*values[:,origin]*observed[:,origin]
        mass=decay*mass+(1-decay)*observed[:,origin]
        result[:,origin+int(horizon)]=np.divide(numer,mass,out=np.zeros_like(numer),where=mass>1e-12)
    return result


def _fit_skeleton(b,features,tr,target_sd,cfg):
    y=np.where(b["valid"]&tr[None,:,None],b["targets"],np.nan)
    link=native_to_link(y,b["target_specs"],target_sd)
    nr,nt,nk=y.shape
    lm=np.nanmean(link[:,tr],axis=1);ls=np.nanstd(link[:,tr],axis=1)
    for k in range(nk):
        pooled=link[:,tr,k];pm=np.nanmean(pooled);ps=max(float(np.nanstd(pooled)),1e-4)
        if not np.isfinite(pm):raise ValueError("target has no training observations")
        lm[:,k]=np.where(np.isfinite(lm[:,k]),lm[:,k],pm)
        ls[:,k]=np.where(np.isfinite(ls[:,k])&(ls[:,k]>=1e-4),ls[:,k],ps)
    lz=(link-lm[:,None])/ls[:,None]
    skeleton=np.zeros_like(b["targets"],dtype="float32");coeff=[];metadata=[]
    for k,xx in enumerate(features["designs"]):
        xx=np.asarray(xx,float);nf=xx.shape[-1]
        names=features["names"][k]
        usable=np.isfinite(lz[...,k])&tr[None]
        ri=np.broadcast_to(np.arange(nr)[:,None],(nr,nt))[usable]
        design=xx[usable];target=lz[...,k][usable]
        counts=np.bincount(ri,minlength=nr);w=1./np.maximum(counts[ri],1);w=w/w.mean()
        # If the shared feature builder supplies contrasts, penalize them there;
        # otherwise append exactly one shrunk intercept for each observed region.
        supplied_region=any(str(n).startswith(("region_","region[")) for n in names)
        append_region=nr>1 and not supplied_region
        penalty=np.repeat(float(cfg["ridge"]),nf)
        for j,name in enumerate(names):
            if name in ("intercept","constant"):penalty[j]=1e-8
            elif str(name).startswith(("region_","region[")):penalty[j]=float(cfg["region_penalty"])
        if append_region:
            design=np.column_stack([design,np.eye(nr)[ri]])
            penalty=np.r_[penalty,np.repeat(float(cfg["region_penalty"]),nr)]
        dx=torch.as_tensor(design,dtype=torch.float64);wt=torch.as_tensor(w,dtype=torch.float64)
        yy=torch.as_tensor(target,dtype=torch.float64)
        gram=dx.T@(dx*wt[:,None])+torch.diag(torch.as_tensor(penalty))
        beta=torch.linalg.solve(gram,dx.T@(yy*wt)).numpy()
        fitted=(torch.as_tensor(xx,dtype=torch.float64)@torch.as_tensor(beta[:nf],dtype=torch.float64)).numpy()
        if append_region:fitted+=beta[nf:][:,None]
        skeleton[...,k]=fitted;coeff.append(beta)
        metadata.append(dict(n_features=nf,appended_region_intercepts=append_region,feature_names=list(names)))
    return skeleton,lm.astype("float32"),ls.astype("float32"),coeff,metadata


def _skeleton_predict(features,coeff,metadata):
    nr,nt=features["designs"][0].shape[:2]
    result=np.zeros((nr,nt,len(coeff)),dtype="float32")
    for k,(xx,beta,meta) in enumerate(zip(features["designs"],coeff,metadata)):
        nf=int(meta["n_features"])
        if list(features["names"][k])!=meta["feature_names"]:raise ValueError("skeleton feature definitions changed")
        part=(torch.as_tensor(xx,dtype=torch.float64)@torch.as_tensor(beta[:nf],dtype=torch.float64)).numpy()
        if meta["appended_region_intercepts"]:part+=np.asarray(beta[nf:])[:,None]
        result[...,k]=part
    return result


def _prepare_inputs(b,features,config,horizon,skeleton):
    windows=causal_windows(features["input_z"],features["input_observed"],b["dates"],
        int(config["context"]),horizon,features["background_z"],features["background_observed"])
    windows["skeleton"]=skeleton
    windows["memory_at_origin"]=_ema_at_origin(features["input_z"],features["input_observed"],horizon,config["memory_halflife"])
    return windows


def _truncate(tensors,stop):
    return {key:(value[:stop] if key in ("calendar_window","rollout_calendar") else value[:,:stop]) for key,value in tensors.items()}


def fit_predict(b,train_mask,val_mask,predict_mask,config,seed,block,horizon=None):
    """Fit native targets, return train-standardized and native predictions.

    All preprocessing and the ridge link skeleton use train_mask only. Inner
    selection is balanced across the two subject groups, independently of the
    candidate's training weights or robust optimization loss. Outer fits accept
    fixed epochs and a fixed inner-derived learning-rate trajectory.
    """
    from common_online import allowed_inputs,scale_targets,outcome_weights
    from online_features import build_features
    started=time.perf_counter();cfg=dict(DEFAULT_CONFIG);cfg.update(config or {})
    h=int(horizon or b["horizon"]);nr,nt,nk=np.shape(b["targets"]);np_=b["inputs"].shape[2]
    tr=np.asarray(train_mask,bool).copy();va=np.zeros(nt,bool) if val_mask is None else np.asarray(val_mask,bool).copy()
    pr=np.asarray(predict_mask,bool).copy()
    if any(mask.shape!=(nt,) for mask in [tr,va,pr]):raise ValueError("time mask shape mismatch")
    if not tr.any() or (tr&va).any():raise ValueError("empty or overlapping training mask")
    if va.any() and np.flatnonzero(va).min()<=np.flatnonzero(tr).max():raise ValueError("validation must follow training")
    if len(b["target_specs"])!=nk or len(b["target_groups"])!=nk:raise ValueError("target metadata mismatch")
    if h<1:raise ValueError("horizon must be a positive native-period integer")
    torch.set_num_threads(int(cfg["threads"]));torch.manual_seed(int(seed));torch.use_deterministic_algorithms(True)
    device=torch.device(cfg["device"])
    if device.type=="cuda":
        if not torch.cuda.is_available():raise RuntimeError("configured local CUDA device is unavailable")
        torch.cuda.manual_seed_all(int(seed));torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    elif device.type!="cpu":raise ValueError("only local CPU/CUDA devices are supported")
    z,mu,sd=scale_targets(b,tr)
    features=build_features(b,tr,block,h,int(cfg["lag"]),float(cfg["memory_halflife"]))
    sk,lm,ls,coef,skmeta=_fit_skeleton(b,features,tr,sd,cfg)
    windows=_prepare_inputs(b,features,cfg,h,sk)
    tensors={key:torch.as_tensor(value,dtype=torch.float32,device=device) for key,value in windows.items()}
    adjacency=np.zeros((nk,np_),bool)
    for k,ids in enumerate(allowed_inputs(b,block)):adjacency[k,ids]=True
    model=OnlineStateSpace(nr,np_,nk,b["background"].shape[2],cfg,adjacency,b["internet_indices"],
        b["target_specs"],mu,sd,lm,ls,block).to(device)
    target=torch.as_tensor(z,dtype=torch.float32,device=device)
    valid=torch.as_tensor(b["valid"],dtype=torch.bool,device=device)
    tt=torch.as_tensor(tr,device=device);vv=torch.as_tensor(va,device=device)
    select_weights=torch.as_tensor(outcome_weights(b,.5),dtype=torch.float32,device=device)
    training_weights=torch.as_tensor(outcome_weights(b,float(cfg["training_official_weight"])),dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["lr"]),weight_decay=float(cfg["wd"]))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=float(cfg["lr_factor"]),
        patience=int(cfg["lr_patience"]),min_lr=float(cfg["min_lr"]))
    train_stop=int(np.flatnonzero(tr).max())+1;eval_stop=int(np.flatnonzero(va).max())+1 if va.any() else train_stop
    train_data=_truncate(tensors,train_stop);eval_data=_truncate(tensors,eval_stop)
    fixed=cfg.get("fixed_epochs")
    if not va.any() and fixed is None:fixed=int(cfg["epochs"])
    budget=int(cfg["epochs"]) if va.any() else int(fixed);maximum=int(cfg["max_epochs"]) if va.any() else budget
    schedule=cfg.get("fixed_lr_schedule")
    if budget<1 or maximum<budget:raise ValueError("invalid training budget")
    if schedule is not None and (va.any() or len(schedule)<budget or min(schedule)<=0 or not np.isfinite(schedule).all()):
        raise ValueError("fixed LR schedule is a positive outer-only trajectory")
    best=float("inf");best_epoch=0;best_state=None;stale=0;curves=[];extensions=[];epoch=0;failure=None;status="running";gradmax=0.
    try:
        if cfg["baseline_only"]:status="selected_transparent_skeleton";best_state=copy.deepcopy(model.state_dict())
        while epoch<budget and not cfg["baseline_only"]:
            epoch+=1;model.train();optimizer.zero_grad(set_to_none=True)
            if schedule is not None:
                for group in optimizer.param_groups:group["lr"]=float(schedule[epoch-1])
            pred=model(**train_data)
            objective,parts=balanced_loss(pred,target[:,:train_stop],tt[:train_stop],valid[:,:train_stop],training_weights,
                cfg["loss"],float(cfg["huber_delta"]),float(cfg["student_df"]))
            penalty=model.region_regularization();total=objective+penalty
            if not torch.isfinite(total):raise FloatingPointError("nonfinite_training_objective")
            total.backward();gradient=nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gradient):raise FloatingPointError("nonfinite_training_gradient")
            gradmax=max(gradmax,float(gradient));optimizer.step();model.eval()
            with torch.no_grad():
                pred_eval=model(**eval_data)
                tl,tp=balanced_loss(pred_eval,target[:,:eval_stop],tt[:eval_stop],valid[:,:eval_stop],select_weights,"mae")
                vl,vp=balanced_loss(pred_eval,target[:,:eval_stop],vv[:eval_stop],valid[:,:eval_stop],select_weights,"mae") if va.any() else (None,None)
            if not torch.isfinite(tl) or (vl is not None and not torch.isfinite(vl)):raise FloatingPointError("nonfinite_updated_loss")
            curves.append(dict(epoch=epoch,train_objective=float(objective.detach()),train_mae=float(tl),validation_mae=float(vl) if vl is not None else None,
                train_channel_mae=tp.detach().cpu().tolist(),validation_channel_mae=vp.detach().cpu().tolist() if vp is not None else None,
                learning_rate=optimizer.param_groups[0]["lr"],gradient_norm_before_clip=float(gradient),region_penalty=float(penalty.detach())))
            if va.any():
                value=float(vl)
                if value<best-1e-7:best=value;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
                else:stale+=1
                scheduler.step(value)
                if epoch>=int(cfg["min_epochs"]) and stale>=int(cfg["patience"]):status="validation_plateau_early_stop";break
                if epoch==budget:
                    win=int(cfg["extension_window"]);older=[c["validation_mae"] for c in curves if c["epoch"]<=max(1,epoch-win)]
                    old=min(older) if older else best;relative=(old-best)/max(abs(old),1e-8)
                    improving=best_epoch>epoch-win and relative>=float(cfg["extension_relative_improvement"])
                    if improving and budget<maximum:
                        nxt=min(maximum,budget+int(cfg["extension_epochs"]))
                        extensions.append(dict(at_epoch=epoch,new_budget=nxt,relative_improvement=relative,best_epoch=best_epoch));budget=nxt
                    else:status="validation_hard_cap_still_improving" if improving else "validation_budget_completed"
            else:
                best_epoch=epoch
                if schedule is None and epoch in {max(1,budget//2),max(1,3*budget//4)}:
                    for group in optimizer.param_groups:group["lr"]=max(float(cfg["min_lr"]),group["lr"]*float(cfg["lr_factor"]))
        if status=="running":status="fixed_epoch_completed"
        if va.any() and not cfg["baseline_only"]:
            if best_state is None:raise FloatingPointError("no_finite_validation_checkpoint")
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():native_estimate=model(**tensors,return_native=True).detach().cpu().numpy().astype(float)
        estimate=(native_estimate-mu[:,None])/sd[:,None]
        if not np.isfinite(estimate[:,pr]).all():raise FloatingPointError("nonfinite_forecast")
        pred_z=np.full_like(z,np.nan,dtype=float);pred_z[:,pr]=estimate[:,pr]
    except (RuntimeError,FloatingPointError) as exc:
        failure=repr(exc);status="numerical_failure";pred_z=np.full_like(z,np.nan,dtype="float32");best_epoch=0
    state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()} if cfg["return_weights"] and failure is None else None
    diagnostics=dict(success=failure is None,status=status,failure_reason=failure,seed=int(seed),epochs_run=epoch,selected_epochs=best_epoch,
        best_validation_mae=best if np.isfinite(best) else None,extensions=extensions,max_gradient_norm_before_clip=gradmax,
        parameter_count=sum(v.numel() for v in model.parameters()),seconds=time.perf_counter()-started,config=cfg,
        task_groups=list(b["target_groups"]),selection_weights=select_weights.cpu().tolist(),training_weights=training_weights.cpu().tolist(),
        n_training_by_target=(b["valid"]&tr[None,:,None]).sum(axis=(0,1)).tolist(),adjacency=adjacency.astype(int).tolist(),
        target_specs=list(b["target_specs"]),block=block,horizon=h,frequency=b["frequency"],context=int(cfg["context"]),
        device=str(device),deterministic_algorithms=True,cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        optimization_stationarity_proven=False,native_output_no_exponential=True,student_t_is_training_score_only=True)
    return dict(pred_z=pred_z,pred=pred_z,native=pred_z*sd[:,None]+mu[:,None],mu=mu,sd=sd,curves=curves,diagnostics=diagnostics,
        selected_epochs=int(best_epoch),weights=state,feature_preprocessing=features["preprocessing"],skeleton_coefficients=coef,
        skeleton_metadata=skmeta,link_mu=lm,link_sd=ls,model=model if cfg.get("return_model",False) else None,
        metadata=dict(regions=list(b["regions"]),input_names=list(b["input_names"]),target_names=list(b["target_names"]),
            target_specs=list(b["target_specs"]),background_names=list(b["background_names"]),internet_indices=list(b["internet_indices"])))


def predict_fitted(b,fitted,predict_mask=None,device=None):
    """Fixed-weight causal prediction for historical and conditional scenarios."""
    from online_features import build_features
    diag=fitted["diagnostics"];cfg=dict(diag["config"]);h=int(diag["horizon"])
    if device is not None:cfg["device"]=device
    nr,nt,np_=b["inputs"].shape;nk=len(b["target_names"])
    meta=fitted["metadata"]
    for key in ("regions","input_names","target_names","target_specs","background_names"):
        if list(b[key])!=meta[key]:raise ValueError("fixed prediction metadata mismatch: "+key)
    features=build_features(b,None,diag["block"],h,int(cfg["lag"]),float(cfg["memory_halflife"]),preprocessing=fitted["feature_preprocessing"])
    sk=_skeleton_predict(features,fitted["skeleton_coefficients"],fitted["skeleton_metadata"])
    windows=_prepare_inputs(b,features,cfg,h,sk)
    model=fitted.get("model")
    if model is None or str(next(model.parameters()).device)!=cfg["device"]:
        model=OnlineStateSpace(nr,np_,nk,b["background"].shape[2],cfg,np.asarray(diag["adjacency"],bool),meta["internet_indices"],
            meta["target_specs"],fitted["mu"],fitted["sd"],fitted["link_mu"],fitted["link_sd"],diag["block"])
        model.load_state_dict(fitted["weights"]);model=model.to(cfg["device"])
    model.eval()
    with torch.no_grad():native=model(**{key:torch.as_tensor(value,dtype=torch.float32,device=cfg["device"]) for key,value in windows.items()},return_native=True).cpu().numpy().astype(float)
    pred=(native-fitted["mu"][:,None])/fitted["sd"][:,None]
    if predict_mask is not None:pred[:,~np.asarray(predict_mask,bool)]=np.nan
    return pred
