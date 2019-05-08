import numpy as np

def euclidDistance(vec1, vec2):
    dist = np.linalg.norm(vec1-vec2)
    return(dist)

def cosSimilarity(vec1, vec2):
    num=float(numpy.sum(vec1*vec2))
    denom=numpy.linalg.norm(vec1)*numpy.linalg.norm(vec2)
    cos=num/denom
    sim=0.5+0.5*cos
    return(sim)
