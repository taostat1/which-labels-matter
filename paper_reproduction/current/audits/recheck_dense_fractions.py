"""Independent Fraction/Walsh check of the displayed Appendix B.5 design.

Uses generic Gaussian elimination and reconstructs the multilinear polynomial
from all 256 signed label vectors, independently of the Node coefficient code.
No external dependencies. Run with Python 3; optional --output writes JSON.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from itertools import product
from pathlib import Path
import argparse
import json


def matrix(rows):
    return [[F(x) for x in row.split()] for row in rows]


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def transpose(a):
    return list(map(list, zip(*a)))


def mm(a, b):
    return [[dot(row, col) for col in transpose(b)] for row in a]


def inverse(a):
    n = len(a)
    aug = [row[:] + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        p = next(i for i in range(j, n) if aug[i][j])
        aug[j], aug[p] = aug[p], aug[j]
        pivot = aug[j][j]
        aug[j] = [v/pivot for v in aug[j]]
        for i in range(n):
            if i != j:
                t = aug[i][j]
                aug[i] = [v-t*w for v, w in zip(aug[i], aug[j])]
    return [row[n:] for row in aug]


def dec(x):
    with localcontext() as c:
        c.prec = 60
        return str(Decimal(x.numerator)/Decimal(x.denominator))


ZA = matrix(['1/2 1/4', '1/3 -1/2', '-1/4 2/3', '3/5 1/5'])
ZB = matrix(['1/3 2/5', '-1/2 1/4', '2/3 -1/3', '1/5 3/5'])
WA = matrix(['1/3 1/2', '-1/2 1/3', '1/4 3/5', '-3/5 1/4'])
WB = matrix(['1/2 -1/4', '2/5 1/3', '-1/3 3/5', '1/4 -1/2'])


def q_matrix(z, w):
    gram = mm(transpose(z), z)
    h = [[v/16 + 10*F(i == j) for j, v in enumerate(row)] for i, row in enumerate(gram)]
    l = [[v/8 for v in row] for row in mm(inverse(h), transpose(z))]
    return mm(w, l)


QA, QB = q_matrix(ZA, WA), q_matrix(ZB, WB)
C = [[-(a-b)/8 for a, b in zip(ra, rb)] for ra, rb in zip(QA, QB)]
R = [[(a-b)/32 for a, b in zip(ra, rb)]
     for ra, rb in zip(mm(transpose(QA), QA), mm(transpose(QB), QB))]


def direct_value(x):
    train, test = x[:4], x[4:]
    a, b = [dot(row, train) for row in QA], [dot(row, train) for row in QB]
    return sum((-y*(aa-bb)/8 + (aa*aa-bb*bb)/32
                for y, aa, bb in zip(test, a, b)), F(0))


paths = list(product((-1, 1), repeat=8))
values = [direct_value(x) for x in paths]
c0 = sum(values, F(0))/256
linear = [sum((t*x[i] for x, t in zip(paths, values)), F(0))/256 for i in range(8)]
quadratic = {(i, j): sum((t*x[i]*x[j] for x, t in zip(paths, values)), F(0))/256
             for i in range(8) for j in range(i+1, 8)}
assert not any(linear)
assert c0 == sum(R[i][i] for i in range(4))
assert all(values[k] == c0 + sum(v*x[i]*x[j] for (i,j),v in quadratic.items())
           for k,x in enumerate(paths))
u = [F(2, 5)]*4
h0 = c0 + sum(quadratic[i,j]*u[i]*u[j] for i in range(4) for j in range(i+1,4))
Cu = [dot(row, u) for row in C]
root = [-h0*(1 if a > 0 else -1)/sum(map(abs, Cu)) for a in Cu]
mu = u + root
b = [sum(v*mu[j if i == k else i] for (i,j),v in quadratic.items() if k in (i,j))
     for k in range(8)]

# Verify the tie and gradient by product-law expectations, without differentiating
# the matrix expression (whose diagonal must first be reduced using x_i**2=1).
def probability(x):
    p = F(1)
    for xx, m in zip(x, mu):
        p *= (1 + xx*m)/2
    return p
weights = [probability(x) for x in paths]
tie = sum((p*t for p,t in zip(weights,values)), F(0))
gradient = [sum((p*t*(x[i]-mu[i])/(1-mu[i]**2)
                for x,p,t in zip(paths,weights,values)), F(0)) for i in range(8)]
assert tie == 0 and gradient == b
B1 = sum(map(abs, b))
S1 = sum(map(abs, quadratic.values()))
d = [sum(abs(v) for (i,j),v in quadratic.items() if k in (i,j)) for k in range(8)]
D2 = dot(d,d)
g = B1*B1/(32*S1)
with localcontext() as c:
    c.prec = 60
    ddec=lambda a: Decimal(a.numerator)/Decimal(a.denominator)
    kmin=32*ddec(D2)*Decimal(20).ln()/ddec(g)**2
    lhs=(2*ddec(D2)*Decimal(20).ln()/Decimal(24442334)).sqrt()
    rhs=ddec(g)/4
    assert lhs < rhs
    details={'K_dispersion_threshold':str(kmin), 'K_used':24442334,
             'dispersion_lhs':str(lhs), 'g_over_4':str(rhs), 'slack':str(rhs-lhs)}
result={'method':'Fraction Gaussian elimination; exhaustive Walsh reconstruction; exact product-law gradient',
        'decimal':{k:dec(v) for k,v in dict(h0=h0,c0=c0,B1=B1,S1=S1,D2=D2,g=g).items()},
        'exact':{k:str(v) for k,v in dict(h0=h0,c0=c0,B1=B1,S1=S1,D2=D2,g=g).items()},
        'evaluation_means':list(map(dec,root)), 'gradient':list(map(dec,b)),
        'checks':{'all_256_polynomial_values':True,'exact_tie':tie==0,
                  'product_law_gradient':gradient==b,'original_replication_valid':lhs<rhs},
        **details}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    txt=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(txt,encoding='utf-8')
    print(txt)
