# Python program to find
# maximal Bipartite matching.
from typing import List, Tuple, Union, Literal
import math


class GFG:
    def __init__(self, graph):

        # residual graph
        self.graph = graph
        self.ppl = len(graph)
        self.jobs = len(graph[0])

    # A DFS based recursive function
    # that returns true if a matching
    # for vertex u is possible
    def bpm(self, u, matchR, seen):

        # Try every job one by one
        for v in range(self.jobs):

            # If applicant u is interested
            # in job v and v is not seen
            if self.graph[u][v] and seen[v] == False:

                # Mark v as visited
                seen[v] = True

                '''If job 'v' is not assigned to
                   an applicant OR previously assigned
                   applicant for job v (which is matchR[v])
                   has an alternate job available.
                   Since v is marked as visited in the
                   above line, matchR[v]  in the following
                   recursive call will not get job 'v' again'''
                if matchR[v] == -1 or self.bpm(matchR[v],
                                               matchR, seen):
                    matchR[v] = u
                    return True
        return False

    # Returns maximum number of matching
    def maxBPM(self):
        '''An array to keep track of the
           applicants assigned to jobs.
           The value of matchR[i] is the
           applicant number assigned to job i,
           the value -1 indicates nobody is assigned.'''
        matchR = [-1] * self.jobs

        # Count of jobs assigned to applicants
        result = 0
        for i in range(self.ppl):

            # Mark all jobs as not seen for next applicant.
            seen = [False] * self.jobs

            # Find if the applicant 'u' can get a job
            if self.bpm(i, matchR, seen):
                result += 1
        return result


def maximum_matching(cord_list1: List[Tuple[int, int]], cord_list2: List[Tuple[int, int]], DIST_THRESHOLD: int) -> int:
    return process_matrix(matrix_maker(cord_list1, cord_list2, DIST_THRESHOLD))


def process_matrix(matrix: List[List[Union[Literal[0], Literal[1]]]]) -> int:
    return GFG(matrix).maxBPM()


def matrix_maker(cord_list1: List[Tuple[int, int]], cord_list2: List[Tuple[int, int]], DIST_THRESHOLD: int) -> List[
    List[Union[Literal[0], Literal[1]]]]:
    matrix = []
    for i in range(len(cord_list1)):
        row = []
        for j in range(len(cord_list2)):
            if is_close_enough(cord_list1[i], cord_list2[j], DIST_THRESHOLD):
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return matrix


def is_close_enough(cord1: Tuple[int, int], cord2: Tuple[int, int], DIST_THRESHOLD: int) -> bool:
    return math.dist(cord1, cord2) <= DIST_THRESHOLD