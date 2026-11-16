// Independent verification from the displayed feature matrices, without audit inputs.
// Run with Node.js. Rational operations use BigInt; square roots and log(20)
// are bounded outwards for the integer replication certificates.

const gcd=(a,b)=>{a=a<0n?-a:a;b=b<0n?-b:b;while(b){const r=a%b;a=b;b=r;}return a;};
const R=(n,d=1n)=>{n=BigInt(n);d=BigInt(d);if(d<0n){n=-n;d=-d;}const g=gcd(n,d);return [n/g,d/g];};
const add=(a,b)=>R(a[0]*b[1]+b[0]*a[1],a[1]*b[1]);
const neg=a=>[-a[0],a[1]],sub=(a,b)=>add(a,neg(b)),mul=(a,b)=>R(a[0]*b[0],a[1]*b[1]),div=(a,b)=>R(a[0]*b[1],a[1]*b[0]);
const abs=a=>[a[0]<0n?-a[0]:a[0],a[1]],sgn=a=>a[0]===0n?0:a[0]>0n?1:-1,num=a=>Number(a[0]*(10n**40n)/a[1])/1e40,str=a=>a[0]+(a[1]===1n?'':'/'+a[1]);
const sum=xs=>xs.reduce(add,R(0)), dot=(x,y)=>sum(x.map((a,i)=>mul(a,y[i])));
const trans=A=>A[0].map((_,i)=>A.map(r=>r[i])), scale=(A,c)=>A.map(r=>r.map(a=>mul(a,c)));
const mm=(A,B)=>{const Bt=trans(B);return A.map(r=>Bt.map(c=>dot(r,c)));};
const ma=(A,B)=>A.map((r,i)=>r.map((a,j)=>add(a,B[i][j]))),ms=(A,B)=>ma(A,scale(B,R(-1)));
const eye=n=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>R(i===j?1:0)));
const inv2=A=>scale([[A[1][1],neg(A[0][1])],[neg(A[1][0]),A[0][0]]],div(R(1),sub(mul(A[0][0],A[1][1]),mul(A[0][1],A[1][0]))));
const parse=A=>A.map(r=>r.map(s=>{const [n,d='1']=s.split('/');return R(n,d);}));
const ZA=parse([['1/2','1/4'],['1/3','-1/2'],['-1/4','2/3'],['3/5','1/5']]);
const ZB=parse([['1/3','2/5'],['-1/2','1/4'],['2/3','-1/3'],['1/5','3/5']]);
const WA=parse([['1/3','1/2'],['-1/2','1/3'],['1/4','3/5'],['-3/5','1/4']]);
const WB=parse([['1/2','-1/4'],['2/5','1/3'],['-1/3','3/5'],['1/4','-1/2']]);
const lambda=R(10), N=4,M=4,u=Array.from({length:4},()=>R(2,5));
const L=Z=>scale(mm(inv2(ma(scale(eye(2),lambda),scale(mm(trans(Z),Z),R(1,16)))),trans(Z)),R(1,8));
const LA=L(ZA),LB=L(ZB),QA=mm(WA,LA),QB=mm(WB,LB);
const C=scale(ms(QA,QB),R(-1,8)),H=scale(ms(mm(trans(QA),QA),mm(trans(QB),QB)),R(1,32));
const c0=sum(H.map((r,i)=>r[i]));
const coeff=Array.from({length:8},()=>Array.from({length:8},()=>R(0)));
for(let i=0;i<4;i++)for(let j=i+1;j<4;j++)coeff[i][j]=coeff[j][i]=mul(R(2),H[i][j]);
for(let i=0;i<4;i++)for(let j=0;j<4;j++)coeff[i+4][j]=coeff[j][i+4]=C[i][j];
const h0=add(c0,sum(Array.from({length:4},(_,i)=>sum(Array.from({length:4-i-1},(_,k)=>mul(coeff[i][i+k+1],mul(u[i],u[i+k+1])))))));
const Cu=C.map(r=>dot(r,u)),Cu1=sum(Cu.map(abs)),root=Cu.map(a=>mul(neg(div(h0,Cu1)),R(sgn(a))));
const mu0=u.concat(root),b=coeff.map(r=>dot(r,mu0)),B1=sum(b.map(abs));
const S1=sum(coeff.map((r,i)=>sum(r.slice(i+1).map(abs)))),d=coeff.map(r=>sum(r.map(abs))),D2=dot(d,d);
const alpha=R(1,5),g1=mul(div(alpha,R(4)),B1),g2=div(mul(B1,B1),mul(R(32),S1)),g=num(g1)<num(g2)?g1:g2;
const shift=div(mul(R(2),g),B1),v=b.map(sgn),muP=mu0.map((a,i)=>add(a,mul(shift,R(v[i])))),muM=mu0.map((a,i)=>sub(a,mul(shift,R(v[i]))));
const tau=mu=>add(c0,sum(coeff.map((r,i)=>sum(r.slice(i+1).map((a,k)=>mul(a,mul(mu[i],mu[i+k+1])))))));
const variance=mu=>{const bb=coeff.map(r=>dot(r,mu)),ss=mu.map(a=>sub(R(1),mul(a,a)));return add(sum(bb.map((a,i)=>mul(mul(a,a),ss[i]))),sum(coeff.map((r,i)=>sum(r.slice(i+1).map((a,k)=>mul(mul(a,a),mul(ss[i],ss[i+k+1])))))));};
const squaredNorm=r=>num(dot(r,r)),rowNorms=A=>A.map(r=>Math.sqrt(squaredNorm(r)));
const separateError=(Z,W,L,Q)=>{const P=mm(Z,L),bb=P.map(r=>num(sum(r.map(abs)))),cc=Q.map(r=>num(sum(r.map(abs)))), zn=rowNorms(Z),wn=rowNorms(W);
const dSimple=bb.reduce((a,b,i)=>a+zn[i]*b**3,0)/(48*4*10);
const dVec=Math.hypot(...trans(Z).map(row=>row.reduce((a,z,i)=>a+num(abs(z))*bb[i]**3,0)/(48*4*10)));
const fit=(Math.min(dSimple,dVec))*wn.reduce((a,b)=>a+b,0)/4;
const rem=cc.reduce((a,c)=>a+c**4,0)/(192*4);return {fit,rem,total:fit+rem,dSimple,dVec,bb,cc};};
const errA=separateError(ZA,WA,LA,QA),errB=separateError(ZB,WB,LB,QB);
const epsilon=errA.total+errB.total;
const vp=num(variance(muP)),vm=num(variance(muM)),tp=num(tau(muP)),tm=num(tau(muM)),ng=num(g),ns=num(S1);
const margins={minus:-tm-ng-epsilon,plus:tp-ng-epsilon};
const BD=32*num(D2)*Math.log(20)/ng**2;
const bern=(variance,range,margin)=>2*(variance+range*margin/3)*Math.log(20)/margin**2;
const BM=bern(vm,num(c0)+ns-tm,margins.minus),BP=bern(vp,tp-num(c0)+ns,margins.plus);
const leading=ms(mm(WA,trans(ZA)),mm(WB,trans(ZB))),lu=leading.map(r=>dot(r,u));
const leadC=scale(leading,R(-1,64)),sInf=sum(leadC.flat().map(abs));
const out={C:C.map(r=>r.map(num)),h0:num(h0),c0:num(c0),Cu:Cu.map(num),root:root.map(num),b:b.map(num),d:d.map(num),B1:num(B1),S1:num(S1),ratio:num(div(B1,S1)),D2:num(D2),g:ng,shift:num(shift),tauMinus:tm,tauPlus:tp,varMinus:vm,varPlus:vp,epsilon,errA,errB,margins,Kbd:BD,KbdCeil:Math.ceil(BD),KbernMinus:BM,KbernPlus:BP,KbernCeil:Math.ceil(Math.max(BM,BP)),leadingU:lu.map(str),SInf:str(sInf),BInf:str(div(sum(lu.map(abs)),R(64))),bLeading:lu.map(a=>str(div(a,R(-64)))),matrixRowNorms:[ZA,ZB,WA,WB].map(rowNorms)};
console.log(JSON.stringify(out,null,2));
const isqrt=n=>{if(n<0n)throw Error('negative square root');if(n<2n)return n;let x=1n<<BigInt(Math.ceil(n.toString(2).length/2));while(true){let y=(x+n/x)/2n;if(y>=x)return x;x=y;}};
const sqrtUpper=(a,dp=30)=>{const s=10n**BigInt(dp),nn=a[0]*s*s,dd=a[1];let u=isqrt(nn/dd);if(u*u*dd<nn)u++;return R(u,s);};
const pow=(a,n)=>{let y=R(1);for(let k=0;k<n;k++)y=mul(y,a);return y;};
const epsUpperFor=(Z,W,L,Q)=>{
 const P=mm(Z,L),br=P.map(r=>sum(r.map(abs))),cr=Q.map(r=>sum(r.map(abs)));
 const re=trans(Z).map(row=>div(sum(row.map((z,i)=>mul(abs(z),pow(br[i],3)))),R(48*4)));
 const du=div(sqrtUpper(dot(re,re)),lambda), grad=div(sum(W.map(r=>sqrtUpper(dot(r,r)))),R(4));
 return add(mul(du,grad),div(sum(cr.map(c=>pow(c,4))),R(192*4)));};
const E=add(epsUpperFor(ZA,WA,LA,QA),epsUpperFor(ZB,WB,LB,QB));
const logInterval=(x,terms=40)=>{let s=R(0);for(let k=0;k<terms;k++)s=add(s,div(pow(x,2*k+1),R(2*k+1)));s=mul(R(2),s);return [s,add(s,div(mul(R(2),pow(x,2*terms+1)),mul(R(2*terms+1),sub(R(1),mul(x,x)))))];};
const l2=logInterval(R(1,3)),l54=logInterval(R(1,9)),l20=l2.map((l,i)=>add(mul(R(4),l),l54[i]));
const exactCeil=a=>(a[0]+a[1]-1n)/a[1];
const KBD=div(mul(mul(R(32),D2),l20[1]),mul(g,g));
const Bernstein=(mu,sign)=>{const tau0=tau(mu);const margin=sub(sub(mul(R(sign),tau0),g),E),range=sign===1?add(sub(tau0,c0),S1):sub(add(c0,S1),tau0);
 return div(mul(mul(R(2),add(variance(mu),div(mul(range,margin),R(3)))),l20[1]),mul(margin,margin));};
const Km=Bernstein(muM,-1),Kp=Bernstein(muP,1);
const denseCertificate={epsilonUpperDecimal:num(E),epsilonLessThan2_425e_9:sub(R(2425,10n**12n),E)[0]>0n,log20Interval:l20.map(num),KbdCeilExact:exactCeil(KBD).toString(),KbernMinusCeilExact:exactCeil(Km).toString(),KbernPlusCeilExact:exactCeil(Kp).toString(),K307100SatisfiesOriginal:sub(R(307100),KBD)[0]>=0n,K307100SatisfiesVariance:sub(R(307100),Km)[0]>=0n&&sub(R(307100),Kp)[0]>=0n};

const K=R(24442334),geq=(a,b)=>sub(a,b)[0]>=0n;
denseCertificate.assumptionChecks={
interior:mu0.every(a=>geq(sub(R(1),mul(R(2),alpha)),abs(a))),
exactTie:tau(mu0)[0]===0n,
positiveSensitivity:B1[0]>0n,
approximationSlack:geq(div(g,R(4)),E),
dispersionSlack:geq(mul(mul(g,g),K),mul(mul(R(32),D2),l20[1])),
shiftSlack:geq(div(mul(alpha,B1),R(2)),g),
curvatureSlack:geq(div(mul(B1,B1),mul(R(16),S1)),g),
fixedKappa:geq(B1,div(S1,R(5)))
};
if(!Object.values(denseCertificate.assumptionChecks).every(Boolean))throw Error('Assumption check failed');
console.log(JSON.stringify(denseCertificate,null,2));
