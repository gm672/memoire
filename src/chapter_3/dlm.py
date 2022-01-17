"""This script is used to generate dependency trees which are optimal in terms of Dependency Length Minimization"""

import random
import copy

# local scripts
import conll3
import random_linearisation
import dl

def get_weight_kids(tree, node, weights):
	"""
	Computes the weight (i.e the number of direct and indirect dependents) for each node.
	
	Recursive function called inside optimal_linearization.

	Parameters
	----------
	tree : Tree
		a dependency Tree in dict format
	node : dict
		the dict corresponding to the current root node
	weights : dict
		the dict updated with each node's weight

	Returns
	-------
	weights : dict
		the dict updated with each node's weight
	"""
	kids = node.get("kids").keys()
	weights[node["id"]] = 0
	if not kids:
		return weights
	for k in kids:
		weights[node["id"]] +=1
		weights = get_weight_kids(tree, tree[k], weights)
		weights[node["id"]] += weights[k]
	return weights
	

def optimal_linearization(tree):
	"""
	Reorders the nodes to minimize Dependency Length.

	The resulting relinearized tree will be one of the possible trees
	that minimize dependency length without altering the structure.
	
	Parameters
	----------
	tree : Tree
		the original dependency Tree in dict format

	Returns
	-------
	linearization : list
		the new sequence of nodes that minimizes Dependency Length
	"""
	tree.addkids()
	root = tree.get_root()

	# creates a dictionary that indicates how many descendents every node has
	weights = get_weight_kids(tree, tree[root], dict())

	# start the linearization
	linearization = list()
	kidz = [[root,sorted([x for x in tree[root].get("kids")], key=lambda x:weights[x], reverse=True)]]
	#print(kidz)
	linearization.append(root)
	first_direction = 0
	count = 0
	new_kids = []

	# # do it as long as some nodes are missing from the linearization
	while len(linearization) < len(tree):
		for idgov, kids in kidz:
			nb_kids = len(kids)

			if count > 0:
				# finding head_direction
				grand_idgov, _ = tree.idgovRel(idgov)
				grand_idgov_index = linearization.index(grand_idgov)
				idgov_index = linearization.index(idgov)

				if idgov_index - grand_idgov_index > 0:
					head_direction = 1
				else:
					head_direction = 0
			else:
				head_direction = first_direction

			for i, k in enumerate(kids):
				idgov_index = linearization.index(idgov)
				if tree[k].get("kids"):

					new_kids += [[k, sorted([x for x in tree[k].get("kids")], key=lambda x:weights[x], reverse=True)]]
				
				# pair kid will always go in the direction of its governor
				if i % 2 == 0:
					first_direction = head_direction
					linearization.insert(idgov_index+first_direction, k)

				# odd kid, inverse direction compare to head direction
				else:
					if head_direction == 1:
						linearization.insert(idgov_index, k)

					else:
						linearization.insert(idgov_index+1, k)

		kidz = new_kids
		count += 1

	return linearization



if __name__ == "__main__":
	conll = """1	Je	il	PRON	_	Number=Sing|Person=1|PronType=Prs	3	nsubj	_	wordform=je
2	me	se	PRON	_	Number=Sing|Person=1|PronType=Prs	3	dep:comp	_	_
3	demande	demander	VERB	_	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin	0	root	_	_
4	s'	si	SCONJ	_	_	7	mark	_	SpaceAfter=No
5	il	il	PRON	_	Gender=Masc|Number=Sing|Person=3|PronType=Prs	7	expl:subj	_	_
6	y	y	PRON	_	Person=3|PronType=Prs	7	dep:comp	_	_
7	a	avoir	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	ccomp:obj	_	_
8	une	un	DET	_	Definite=Ind|Gender=Fem|Number=Sing|PronType=Art	9	det	_	_
9	loi	loi	NOUN	_	Gender=Fem|Number=Sing	7	obj	_	_
10	chez	chez	ADP	_	_	11	case	_	_
11	nous	lui	PRON	_	Number=Plur|Person=1|PronType=Prs	7	obl:mod	_	_
12	qui	qui	PRON	_	PronType=Rel	13	nsubj	_	_
13	peut	pouvoir	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	9	acl:relcl	_	_
14	condamner	condamner	VERB	_	VerbForm=Inf	13	xcomp:obj	_	_
15	des	un	DET	_	Definite=Ind|Number=Plur|PronType=Art	16	det	_	_
16	citoyens	citoyen	NOUN	_	Gender=Masc|Number=Plur	14	obj	_	_
17	pour	pour	ADP	_	_	19	case	_	_
18	des	un	DET	_	Definite=Ind|Number=Plur|PronType=Art	19	det	_	_
19	rêves	rêve	NOUN	_	Gender=Masc|Number=Plur	14	obl:mod	_	_
20	illégaux	illégal	ADJ	_	Gender=Masc|Number=Plur	19	amod	_	_
21	?	?	PUNCT	_	_	3	punct	_	_
"""

