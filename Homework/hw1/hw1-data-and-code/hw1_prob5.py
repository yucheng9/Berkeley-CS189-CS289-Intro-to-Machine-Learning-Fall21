def assemble_feature(x, D):
    '''
    x should be an Nx5 dimensional numpy array, where N is the number of data points
    D is the maximum degree of the multivariate polynomial
    '''
    n_feature = x.shape[1]
    Q = [(np.ones(x.shape[0]), 0, 0)]
    i = 0
    while Q[i][1] < D:
        cx, degree, last_index = Q[i]
        for j in range(last_index, n_feature):
            Q.append((cx * x[:, j], degree + 1, j))
        i += 1
    return np.column_stack([q[0] for q in Q])

