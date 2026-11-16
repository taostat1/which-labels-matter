"""A finite-cost model calculation, separate from the empirical text study."""
from core import *
from scipy.optimize import brentq

def main():
    n,r,lam,g,delta=128,8,2.,.005,.05
    k=lam+.25; v=1/(2*k); G=.5+1/(8*lam)
    epsilon=2*(G*v**3/(48*lam)+v**4/192)
    a=1/(32*k*k*n);b=(8*k-1+1/n)/(32*k*k);D=a+b
    root=brentq(lambda t:lam*t+.5*np.tanh(t/2)-.5,0,1/(2*lam),xtol=1e-15)
    R=float(np.logaddexp(root/2,-root/2)-np.log(2)+root/2)
    Nm=int(np.ceil(2*D*D*np.log(2/delta)/(g-epsilon)**2))
    Nr=int(np.ceil(2*R*R*np.log(2/delta)/g**2))
    labels=64*r+2*Nm+n*r;images=64+2*Nm+n
    operations=10000+200*Nm
    low_ref=Nr*r*(n+1)
    row=dict(n=n,r=r,lambda_=lam,tolerance=g,confidence=1-delta,
             epsilon=epsilon,weight_same_image=a,weight_cross_image=b,moment_range=D,refit_exact_range=R,
             moment_blocks=Nm,refit_blocks=Nr,moment_label_upper=labels,moment_image_upper=images,
             moment_operation_upper=operations,refit_label_lower=low_ref,
             label_ratio=labels/low_ref,
             surrogate_gap_at_independent_mean_half=-.25*(8*k-1)/(32*k*k),theta_max=root)
    assert epsilon<g and labels<low_ref
    save(ROOT/'cost_instance.json',row);print(json.dumps(row,indent=2))

if __name__=='__main__':main()
