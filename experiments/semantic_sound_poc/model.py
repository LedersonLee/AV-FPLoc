"""Inference module. Receives known FP source support, never evaluation truth/source."""
import numpy as np
from scipy.special import logsumexp

def wrap(x):return (np.asarray(x)+np.pi)%(2*np.pi)-np.pi
def softmax(x):return np.exp(x-logsumexp(x,axis=-1,keepdims=True))
def bearing(poses,locations):
 d=np.asarray(locations)[None,:, [0,2]]-np.asarray(poses)[:,None,:2]
 return wrap(np.arctan2(d[:,:,1],d[:,:,0])-np.deg2rad(np.asarray(poses)[:,None,2]))
def room_weights(event,labels,c):
 rule=c['room_event_weights'][event];w=np.array([rule.get(label,rule['default']) for label in labels]);return w/w.sum()
def direction_likelihood(poses,angle,event,support,c,labels=None):
 labels=[r['label'] for r in support] if labels is None else labels
 w=room_weights(event,labels,c);kappa=1/np.deg2rad(c['angular_sigma_deg'])**2
 # Common normalizer omitted: cancels across pose and semantic conditions (fixed kappa).
 return sum(ww*np.exp(kappa*(np.cos(wrap(bearing(poses,r['integration'])-angle))-1)).mean(1) for ww,r in zip(w,support))
def posterior(poses,visual_scores,event,angle,support,c,method='B3',labels=None):
 v=softmax(np.asarray(visual_scores)/c['visual_temperature'])
 if event=='silence' or method in ['B0','B1']:return v
 if method=='B2':
  events=list(c['room_event_weights']);p=np.array([c['event_probabilities'][e] for e in events]);p=p/p.sum();lik=sum(pp*direction_likelihood(poses,angle,e,support,c) for pp,e in zip(p,events))
 else:lik=direction_likelihood(poses,angle,event,support,c,labels)
 # L_sem(c|q) = P(c) independent of q by declared activation model; cancels.
 return softmax(np.log(v)+c['beta']*np.log(np.maximum(lik,1e-300)))
