from sys import prefix
from graphical_models.classes.dags.dag import DAG
import numpy as np
from numpy.linalg import inv
import graphical_models as gm
from .dag import *


# generate DAG
def gen_dag(nnodes, DAG_type):

	DAG_gen = {
		'random': random_graph,
		'barabasialbert': barabasialbert_graph,
		'line': line_graph,
		'path': path_graph,
		'instar': instar_graph,
		'outstar': outstar_graph,
		'tree':  tree_graph,
		'complete': complete_graph,
		'chordal': chordal_graph,
		'rootedtree': rooted_tree_graph,
		'cliquetree': cliquetree_graph
	}.get(DAG_type, None)
	assert DAG_type is not None, 'Unsuppoted DAG type!'

	return DAG_gen(nnodes)	
