import numpy as np
import sys

sys.path.extend(['../'])
from graph import tools
import networkx as nx

# Joint index:
    # {0,  "Nose"},
    # {1, "LEye"},
    # {2, "REye"},
    # {3, "LEar"},
    # {4, "REar"},
    # {5,  "LShoulder"},
    # {6,  "RShoulder"},
    # {7,  "LElbow"},
    # {8,  "RElbow"},
    # {9,  "LHand"},
    # {10,  "RHand"},
    # {11, "LHip"},
    # {12,  "RHip"},
    # {13, "LKnee"},
    # {14,  "RKnee"},
    # {15, "LAnkle"},
    # {16, "RAnkle"},
# Edge format: (origin, neighbor)
num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward = [(1, 2), (2, 4), (3, 1), (3, 4), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), 
          (5, 6), (7, 5), (8, 6), (9, 7), (10, 8), (7, 8), (9, 10), (13, 14), (15, 16), (7, 11), (8, 12),
          (11, 5), (12, 6), (11, 12), (13, 11), (14, 12), (15, 13), (16, 14), (9, 15), (10, 16)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')
