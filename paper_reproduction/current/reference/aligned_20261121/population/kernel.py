"""Sample one term after substituting labels known at batch entry."""
import numpy as np
from numba import njit

@njit(cache=True)
def sample(c,v,w,left,right,keys,known,u,uniform):
    count,nodes=keys.shape
    const=c.copy()
    weight=np.zeros(count)
    first=np.full(count,-1,np.int64)
    second=first.copy()
    for i in range(count):
        # Position labels alias only when the two training draws coincide.
        slot=np.arange(nodes)
        for a in range(nodes):
            for b in range(a):
                if keys[i,a]==keys[i,b]:
                    slot[a]=b
                    break
        acc=np.zeros(nodes*(nodes+1))
        used=np.zeros(len(acc),np.bool_)
        order=np.empty(len(acc),np.int64)
        num=0
        for j in range(nodes+len(left)):
            if j<nodes:
                x=slot[j];y=-1;val=v[i,j]
            else:
                t=j-nodes;x=slot[left[t]];y=slot[right[t]];val=w[i,t]
            if val==0:continue
            if x==y:const[i]+=val;continue
            k=keys[i,x]
            if np.isfinite(known[k]):
                val*=2*known[k]-1;x=-1
            if y>=0:
                k=keys[i,y]
                if np.isfinite(known[k]):
                    val*=2*known[k]-1;y=-1
            if x<0:x,y=y,x
            if x<0:const[i]+=val;continue
            if y>=0 and keys[i,x]>keys[i,y]:x,y=y,x
            idx=x*(nodes+1)+y+1
            if not used[idx]:
                used[idx]=True;order[num]=idx;num+=1
            acc[idx]+=val
        mass=0.;nz=0
        for j in range(num):
            z=acc[order[j]]
            if z!=0:mass+=abs(z);nz+=1
        if nz==0:continue
        s=0.;last=-1
        threshold=u[i]*(nz if uniform else mass)
        for j in range(num):
            idx=order[j];z=acc[idx]
            if z==0:continue
            s+=(1. if uniform else abs(z));last=idx
            if s>threshold:break
        x=last//(nodes+1);y=last%(nodes+1)-1
        first[i]=keys[i,x]
        if y>=0:second[i]=keys[i,y]
        weight[i]=nz*acc[last] if uniform else np.sign(acc[last])*mass
    return const,weight,first,second
