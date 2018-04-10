"""
MSP(max sub-array problem)
wiki(https://en.wikipedia.org/wiki/Maximum_subarray_problem).

1-D:
    kadene's algorithm time complexity: O(n)
2-D:
    ~=O(n^3)
"""
import numpy as np
from itertools import combinations, accumulate


def msp_1D(A):
    """
    1D array MSP.

    Parameters
    ----------
    A: list or np.ndarray

    Returns
    -------
    max_so_far: max sum of subarraies.
 
    >>> A = np.array([1, 2, 3, 0, -1, -2, 3, 4, 1, -2, 0])
    >>> max_so_far = msp_1D(A)
    >>> max_so_far == 11
    True
    """
    max_ending_here = max_so_far = 0
    for x in A:
        max_ending_here = max(0, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far


def msp_1D_with_index(A):
    """
    solve 1D max sub-array problem.
    
    Parameters
    ----------
    A: list or np.ndarray

    Returns
    -------
    start_ind: int
    end_ind: int
    max_so_far: max sum of subarraies.
    
    That is to say:
    >>> A = np.array([1, 2, 3, 0, -1, -2, 3, 4, 1, -2, 0])
    >>> start_ind, end_ind, max_so_far = msp_1D_with_index(A)
    >>> start_ind, end_ind, max_so_far
    (0, 8, 11)
    >>> sum(A[start_ind:end_ind+1]) == max_so_far
    True
    """
    start_ind, end_ind = 0, 0
    max_ending_here, max_so_far = 0, 0
    for idx, x in enumerate(A):
        max_ending_here1 = max_ending_here + x
        if max_ending_here1 > 0:
            max_ending_here = max_ending_here1
        else:
            # start with this idx.
            start_ind = idx + 1
            max_ending_here = 0
        if max_so_far < max_ending_here:
            # end with this idx.
            end_ind = idx
            max_so_far = max_ending_here
    return start_ind, end_ind, max_so_far


def msp_2D(A):
    """
    2D array MSP.
    
    Parameters
    ----------
    A: np.ndarray

    Returns
    -------
    start_ind: int
    end_ind: int
    max_so_far: max sum of subarraies.
    
    Steps
    -----
    1. caculate acc matrix(accumulate of A).
    2. for all indexes i, j pairs(i<j),
        calculating sum of block for a k column(using acc matrix).
        do 1-d MSP(get a max sum block under i, j)
    3. return the max.

    >>> A = np.array([\
        [0, -2, -7, 0],\
        [9, 2, -6, 2],\
        [-4, 1, -4, 1],\
        [-1, 8, 0, -2]])
    >>> m = msp_2D(A)
    >>> m == 15
    True
    """
    A = np.array(A)
    acc = np.array([*accumulate(A)])
    m = 0
    for i, j in combinations(range(A.shape[0]), 2):
        m_tmp = msp_1D([
            acc[j, k] - acc[i - 1, k] if i != 0 else acc[j, k]
            for k in range(A.shape[1])
        ])
        if m_tmp > m:
            m = m_tmp
    return m


def msp_2D_with_index(A):
    """
    2D array MSP.
    
    Parameters
    ----------
    A: np.ndarray

    Returns
    -------
    start_ind: tuple
        max sum block left-top position.
    end_ind: tuple
        max sum block right-bottom position.
    max_so_far: max sum of subarraies.
    
    That is to say:
    >>> A = np.array([\
        [0, -2, -7, 0],\
        [9, 2, -6, 2],\
        [-4, 1, -4, 1],\
        [-1, 8, 0, -2]])
    >>> start_ind, end_ind, m = msp_2D_with_index(A)
    >>> A[start_ind[0]:end_ind[0]+1, start_ind[1]:end_ind[1]+1].sum() == m
    True
    """
    A = np.array(A)
    acc = np.array([*accumulate(A)])
    start_ind, end_ind, m = None, None, 0
    for i, j in combinations(range(A.shape[0]), 2):
        sind, eind, m_tmp = msp_1D_with_index([
            acc[j, k] - acc[i - 1, k] if i != 0 else acc[j, k]
            for k in range(A.shape[1])
        ])
        if m_tmp > m:
            m = m_tmp
            start_ind = (i, sind)
            end_ind = (j, eind)
    return start_ind, end_ind, m
