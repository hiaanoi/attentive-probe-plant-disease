"""Correction-overlap estimators used by the released prediction checker."""
import numpy as np


def correction_overlap(y, baseline, attentive, fusion):
    """Directional corrected-image overlap, not Jaccard similarity.

    y, baseline and fusion are [N] class IDs. attentive is [seeds,N].
    The observed denominator is each attention seed's corrections of baseline
    errors. Between-seed overlap averages all ordered pairs of distinct seeds.
    Independence conditions on the images misclassified by the baseline.
    """
    y=np.asarray(y);baseline=np.asarray(baseline)
    attentive=np.asarray(attentive);fusion=np.asarray(fusion)
    if y.ndim!=1 or baseline.shape!=y.shape or fusion.shape!=y.shape:
        raise ValueError('Truth, baseline and fusion must be aligned [N] arrays')
    if attentive.ndim!=2 or attentive.shape[1]!=len(y) or len(attentive)<2:
        raise ValueError('At least two aligned attention seeds are required')
    wrong=baseline!=y;att=attentive==y;extra=fusion==y
    E=att&wrong;F=extra&wrong
    if not wrong.any() or not np.all(E.sum(1)>0):
        raise ValueError('Overlap is undefined with no baseline errors or no corrections')
    observed=np.mean([100*(e&F).sum()/e.sum() for e in E])
    between=np.mean([100*(a&b).sum()/a.sum() for i,a in enumerate(E) for j,b in enumerate(E) if i!=j])
    expect=100*F.sum()/wrong.sum()
    return {'observed_overlap_pct':float(observed),
            'between_seed_overlap_pct':float(between),
            'independent_expectation_pct':float(expect)}


def byte_overlap_mask(test_identifiers, retained_identifiers):
    """Apply the released retained-record list without examining image pixels."""
    retained=set(retained_identifiers)
    return np.array([v in retained for v in test_identifiers],dtype=bool)
