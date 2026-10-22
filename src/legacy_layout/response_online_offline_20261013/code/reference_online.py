"""Transparent causal ridge on native-outcome training z scores.

No data loading, study fitting, or writes on import. Feature construction is shared
with the neural comparison; all information blocks receive the same candidates.
"""
import copy
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
import common_online as C
from online_features import build_features


def candidates(frequency):
    lags = [3, 6, 12] if frequency == 'month' else [4, 8, 13]
    halves = [3, 6] if frequency == 'month' else [8, 13]
    # A fixed, finite covering design, not the full Cartesian search.
    design = [(0,0,1,10),(0,1,10,100),(0,0,100,1000),(0,1,1,1000),
              (1,0,10,10),(1,1,100,100),(1,0,1,100),(1,1,10,1000),
              (2,0,100,10),(2,1,1,10),(2,0,10,100),(2,1,100,1000)]
    return [dict(config_id=f'r{i+1:02d}', lag=lags[a], memory_halflife=halves[h],
                 alpha=float(alpha), region_penalty=float(region))
            for i,(a,h,alpha,region) in enumerate(design)]


def project_native(native, specs):
    projected = np.array(native, dtype=float, copy=True)
    for k, spec in enumerate(specs):
        if spec == 'percent': projected[:,:,k] = np.clip(projected[:,:,k], 0., 100.)
        elif spec == 'nonnegative': projected[:,:,k] = np.maximum(projected[:,:,k], 0.)
        elif spec != 'identity': raise ValueError('Unknown target support: '+str(spec))
    return projected


def penalty_vector(names, config):
    penalty = np.full(len(names), float(config['alpha']))
    for j, name in enumerate(names):
        if name == 'intercept': penalty[j] = 0.
        elif name.startswith('region_'): penalty[j] = float(config['region_penalty'])
    return penalty


def equal_region_weights(valid):
    """Positive observation weights sum to N; every represented region sums to N/R."""
    counts = valid.sum(axis=1)
    weights = np.divide(valid, counts[:,None], out=np.zeros(valid.shape,float), where=counts[:,None]>0)
    if weights.sum() <= 0: raise ValueError('No qualified training observations')
    weights *= valid.sum()/weights.sum()
    return weights, counts


def fit_predict(b, train_mask, predict_mask, config, block, horizon=None):
    start=time.perf_counter(); tr=np.asarray(train_mask,bool); pr=np.asarray(predict_mask,bool)
    if tr.shape!=(len(b['dates']),) or pr.shape!=tr.shape: raise ValueError('Mask shape mismatch')
    if not tr.any(): raise ValueError('Empty fitting window')
    h=int(horizon or b['horizon'])
    z,mu,sd=C.scale_targets(b,tr)
    features=build_features(b,tr,block,h,int(config['lag']),float(config['memory_halflife']))
    coefficients=[]; head_diagnostics=[]; raw_z=np.empty_like(z,dtype=float)
    with threadpool_limits(limits=1):
        for k, design in enumerate(features['designs']):
            valid=b['valid'][:,:,k]&tr[None,:]&np.isfinite(z[:,:,k])
            weights,counts=equal_region_weights(valid)
            xx=design[valid]; yy=z[:,:,k][valid]; ww=weights[valid]
            if not np.isfinite(xx).all(): raise FloatingPointError('Nonfinite training design')
            weighted=xx*np.sqrt(ww[:,None]); target=yy*np.sqrt(ww)
            penalty=penalty_vector(features['names'][k],config)
            normal=weighted.T@weighted
            normal.flat[::len(penalty)+1]+=penalty
            rhs=weighted.T@target
            coefficient=cho_solve(cho_factor(normal,lower=True,check_finite=True),rhs,check_finite=True)
            if not np.isfinite(coefficient).all(): raise FloatingPointError('Nonfinite ridge coefficients')
            estimate=design.reshape(-1,design.shape[-1])@coefficient
            raw_z[:,:,k]=estimate.reshape(design.shape[:2]); coefficients.append(coefficient)
            gradient=normal@coefficient-rhs
            residual=xx@coefficient-yy
            head_diagnostics.append(dict(target=b['target_names'][k],training_rows=int(valid.sum()),
                training_rows_by_region=counts.tolist(),region_weight_totals=weights.sum(axis=1).tolist(),
                features=design.shape[-1],active_columns=int(np.any(xx!=0,axis=0).sum()),
                weighted_sse=float(np.sum(ww*residual**2)),penalty=float(np.sum(penalty*coefficient**2)),
                normal_equation_max_abs_residual=float(np.max(np.abs(gradient))),
                normal_equation_relative_residual=float(np.max(np.abs(gradient))/max(1.,np.max(np.abs(rhs))))))
    raw_native=raw_z*sd[:,None,:]+mu[:,None,:]
    native=project_native(raw_native,b['target_specs']); pred_z=(native-mu[:,None,:])/sd[:,None,:]
    if not np.isfinite(pred_z[:,pr]).all(): raise FloatingPointError('Nonfinite predicted design')
    changed=np.abs(native-raw_native)>1e-10
    projected_by_head=[int((changed[:,:,k]&b['valid'][:,:,k]&pr[None,:]).sum()) for k in range(len(coefficients))]
    pred_z[:,~pr]=np.nan; raw_z[:,~pr]=np.nan
    diagnostics=dict(success=True,status='closed_form_cholesky',seconds=time.perf_counter()-start,
        block=block,horizon=h,config=copy.deepcopy(config),heads=head_diagnostics,
        projection_count_by_target=projected_by_head,projection_count=sum(projected_by_head),
        training_objective='sum weighted native-z squared residuals + coefficient ridge penalty',
        region_weighting='equal region totals; observation weights sum to qualified training N',
        optimization_iterative=False,selection_uses_projected_predictions=True)
    return dict(pred_z=pred_z,pred=pred_z,unconstrained_z=raw_z,mu=mu,sd=sd,
        coefficients=coefficients,feature_names=features['names'],feature_preprocessing=features['preprocessing'],
        diagnostics=diagnostics,config=copy.deepcopy(config),block=block,horizon=h,
        metadata={key:list(b[key]) for key in ('regions','input_names','target_names','target_specs','background_names')})


def predict_fitted(b, fitted, predict_mask=None, return_unconstrained=False):
    for key, value in fitted['metadata'].items():
        if list(b[key])!=value: raise ValueError('Fixed prediction metadata mismatch: '+key)
    cfg=fitted['config'];features=build_features(b,None,fitted['block'],fitted['horizon'],
        cfg['lag'],cfg['memory_halflife'],preprocessing=fitted['feature_preprocessing'])
    if features['names']!=fitted['feature_names']: raise ValueError('Fixed feature-name mismatch')
    raw_z=np.empty_like(b['targets'],dtype=float)
    with threadpool_limits(limits=1):
        for k,design in enumerate(features['designs']):
            raw_z[:,:,k]=(design.reshape(-1,design.shape[-1])@fitted['coefficients'][k]).reshape(design.shape[:2])
    native=raw_z*fitted['sd'][:,None,:]+fitted['mu'][:,None,:]
    pred=(project_native(native,b['target_specs'])-fitted['mu'][:,None,:])/fitted['sd'][:,None,:]
    if predict_mask is not None:
        pred[:,~np.asarray(predict_mask,bool)]=np.nan;raw_z[:,~np.asarray(predict_mask,bool)]=np.nan
    return (pred,raw_z) if return_unconstrained else pred


def calibration_native(fitted, b, validation_mask):
    """Region/head finite-rank 90% absolute native residual; empirical after selection."""
    residual=np.abs(fitted['pred_z']-(b['targets']-fitted['mu'][:,None,:])/fitted['sd'][:,None,:])*fitted['sd'][:,None,:]
    q=np.full(fitted['mu'].shape,np.nan);counts=np.zeros(q.shape,int);ranks=np.full(q.shape,-1,int)
    for r in range(q.shape[0]):
        for k in range(q.shape[1]):
            ok=np.asarray(validation_mask,bool)&b['valid'][r,:,k]&np.isfinite(residual[r,:,k])
            values=np.sort(residual[r,ok,k]);n=len(values);counts[r,k]=n
            if n:
                rank=min(n-1,max(0,int(np.ceil(.9*(n+1)))-1));ranks[r,k]=rank;q[r,k]=values[rank]
    return q,counts,ranks


def compact_fitted(fitted):
    return {key:value for key,value in fitted.items() if key not in ('pred_z','pred','unconstrained_z')}
