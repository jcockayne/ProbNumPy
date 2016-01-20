import numpy as np


def gram_matrix(k, points):
    # todo: cythonize Gram matrix.
    ret = np.empty((points.shape[0], points.shape[0]))
    for i in xrange(points.shape[0]):
        for j in xrange(i, points.shape[0]):
            ret[i,j] = k(points[i,:], points[j,:])
            ret[j,i] = ret[i,j]
    return ret


def bq_weights(mu_p, design_points, kernel):
    """
    points will be a matrix of shape (n_points, d)
    """
    cov_p = gram_matrix(kernel, design_points)
    # TODO: this is going to be ill conditioned
    cov_p_inv = np.linalg.inv(cov_p)

    return mu_p.T.dot(cov_p_inv)


def frank_wolfe_scores(cur_weights, cur_points, test_points, k, kernel_mean_map):
    assert len(cur_weights) == len(cur_points)
    scores = np.empty(test_points.shape[0])
    # todo: nested loop is bad news.
    for i in xrange(len(test_points)):
        x = test_points[i,:]
        k_evals = np.array([k(x, cur_points[j,:]) for j in xrange(cur_points.shape[0])])
        scores[i] = np.sum(cur_weights * k_evals) - kernel_mean_map(x)

    return scores


def frank_wolfe_weights(rhos, iteration):
    ret = np.empty(iteration)
    # todo: this is really confusing, need to go through with FX.
    for l in xrange(iteration):
        prev_rho = rhos[l-1] if l > 0 else 1. # rho_0 = 1
        ret[l] = np.prod(1-rhos[l:iteration-1])*prev_rho
    return ret


def frank_wolfe_steps(method, iterations):
    if method == 'kernel-herding':
        return 1. / (np.arange(iterations) + 2)
    raise Exception('Method {} not understood.'.format(method))

    
def frank_wolfe_step_line_search(new_pt, cur_weights, cur_points, kernel, kernel_mean_map):
    gram = gram_matrix(kernel, cur_points)
    term1 = cur_weights.dot(gram).dot(cur_weights)
    term2 = 0.
    weighted_mean_map = 0.
    for pt, wt in zip(cur_points, cur_weights):
        term2 += wt*kernel(pt, new_pt)
        weighted_mean_map += wt*kernel_mean_map(pt)
    numerator = term1 - term2 - weighted_mean_map + kernel_mean_map(new_pt)
    denominator = term1 - 2*term2 + kernel(new_pt, new_pt)
    return numerator / denominator

    
def frank_wolfe(initial_point, iterations, kernel, kernel_mean_map, test_points, steps='line-search'):
    """
    Determine Frank-Wolfe design points for the supplied kernel and mean-map.
    :param initial_point: The seed point for the Frank-Wolfe algorithm.
    :param iterations: The number of iterations.
    :param kernel: The kernel
    :param kernel_mean_map: The kernel mean map.
    :param test_points: Set of candidate points for the Frank-Wolfe algorithm
    :param steps: The step sizes for the Frank-Wolfe algorithm
    :return:
    """
    line_search = False
    if steps == 'line-search':
        line_search = True
        rho_arr = np.empty(iterations)
        rho_arr[0] = frank_wolfe_step_line_search(initial_point, np.zeros((0)), np.zeros((0,2)), kernel, kernel_mean_map)
    elif type(steps) is str:
        rho_arr = frank_wolfe_steps(steps, iterations)
    elif type(steps) in [list, np.ndarray]:
        rho_arr = np.asarray(steps)
    else:
        raise Exception("Don't understand rho_method={}".format(steps))

    assert len(rho_arr) == iterations
    ret = np.empty((iterations, initial_point.shape[1]))
    ret[0, :] = initial_point
    for i in xrange(1, iterations):
        # todo: optimal weights
        weights = frank_wolfe_weights(rho_arr, i)
        scores = frank_wolfe_scores(weights, ret[:i, :], test_points, kernel, kernel_mean_map)
        best_score_ix = np.argmin(scores)
        new_pt = test_points[best_score_ix, :]
        ret[i, :] = new_pt
        
        if line_search:
            rho_arr[i] = frank_wolfe_step_line_search(new_pt, weights, ret[:i, :], kernel, kernel_mean_map)
    final_weights = frank_wolfe_weights(rho_arr, iterations)
    return ret, final_weights


def fwbq(integrand, x_0, kernel, kernel_mean_map, initial_error, fw_iterations, fw_test_points, fw_steps='line-search', return_points=False):
    """
    Integrate function given by integrand using FWBQ
    :param integrand: The function to integrate
    :param x_0: The initial point with which to seed the Frank-Wolfe algorithm
    :param kernel: The kernel to use for integration.
    :param kernel_mean_map: Mean map for the kernel
    :param initial_error: Integral of the kernel mean map; represents the maximum worst case error when approximating
    with zero observations.
    :param fw_iterations: Number of Frank-Wolfe points to use.
    :param fw_test_points: The set of candidate points for the Frank-Wolfe algorithm
    :param fw_step_method: Step method to use in Frank-Wolfe.
    One of 'line-search', 'kernel-herding' or an array of step sizes.
    'line-search':    Use the line search method in each iteration to determine the optimal step size.
    'kernel-herding': Corresponds to an equal weighting for each FW point.
    If an array is passed the number of elements must be the same as fw_iterations.
    :return: (mu, sigma), the mean and variance for the integral.
    """
    # we are not interested in the FW weights
    design_points, _ = frank_wolfe(x_0, fw_iterations, kernel, kernel_mean_map, fw_test_points, fw_steps)
    fun_evals = integrand(design_points)

    mu_p = np.empty((len(design_points), 1))

    # todo: what's the best way to do this?
    for i in xrange(len(design_points)):
        mu_p[i, :] = kernel_mean_map(design_points[i,:])

    weights = bq_weights(mu_p, design_points, kernel)

    if return_points:
        return design_points, weights

    cov = initial_error - weights.dot(mu_p)

    # todo: also return covariance here.
    return weights.dot(fun_evals), cov

