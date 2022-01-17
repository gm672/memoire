import conll3
import dlm
import stanza
from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL
import random

def DL_T(tree):
    distance = 0
    for el in tree : 
        head = get_head(el,tree)
        if head == 0:
            pass
        else:
            distance = distance + abs(head - el)
    return distance


def DL_L(lin,tree):
	"""
	returns the depedency length of a linearization
	"""
	distance = 0
	for el in lin :
		pos = lin.index(el)
		head = get_head(el,tree)
		if head == 0:
			pass
		else:
			h_pos = lin.index(head)
			d = abs(pos - h_pos)
			distance = distance + d
	return distance

def true_random(tree):
    #random permutation according to Durstenfeld,1964
    #https://dl.acm.org/doi/pdf/10.1145/364520.364540
        a = [x for x in tree]
        n = len(a)

        for i in range(n-1,2,-1):
            j = random.randint(0,i)
            b = a[i] 
            a[i]= a[j]
            a[j] = b
        return a

def omega(dmin,drandom,d,l):
    drda = 1/3 * ((l*l)-1)
    if (drda - dmin) == 0:
        drda = drda + 1
    o = (drda - d) / (drda - dmin)
    return o

def gamma(dmin,d):
    if dmin == 0:
        g = d/1
    else:
        g = d/dmin
    return g

def MDD(d,l):
    if (l-1) == 0:
        m = d
    else:
        m = (1/(l-1))*d
    return m

def get_head(el,tree):
	d = tree.__getitem__(el)
	g = d['gov']
	head = [key for key in g]
	return head[0]


