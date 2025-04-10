import math
import numpy as np
from numpy.linalg import inv, pinv
from scipy import optimize
from warnings import warn
import numbers
import torch
import copy

def check_random_state(seed):
    """Turn seed into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : None, int or np.random.RandomState
           If `seed` is None, return the RandomState singleton used by `np.random`.
           If `seed` is an int, return a new RandomState instance seeded with `seed`.
           If `seed` is already a RandomState instance, return it.

    Raises
    ------
    ValueError
        If `seed` is of the wrong type.

    """
    # Function originally from scikit-learn's module sklearn.utils.validation
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)

def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError('Input data must be at least 2D.')

def _norm_along_axis(x, axis):
    """NumPy < 1.8 does not support the `axis` argument for `np.linalg.norm`."""
    return np.sqrt(np.einsum('ij,ij->i', x, x))

def _center_and_normalize_points(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.

    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(D).

    If the points are all identical, the returned values will contain nan.

    Parameters
    ----------
    points : (N, D) array
        The coordinates of the image points.

    Returns
    -------
    matrix : (D+1, D+1) array_like
        The transformation matrix to obtain the new points.
    new_points : (N, D) array
        The transformed image points.

    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.

    """
    n, d = points.shape
    centroid = np.mean(points, axis=0)

    centered = points - centroid
    rms = np.sqrt(np.sum(centered ** 2) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan)

    norm_factor = np.sqrt(d) / rms

    part_matrix = norm_factor * np.concatenate(
            (np.eye(d), -centroid[:, np.newaxis]), axis=1
            )
    matrix = np.concatenate(
            (part_matrix, [[0,] * d + [1]]), axis=0
            )

    points_h = np.vstack([points.T, np.ones(n)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points

def _center_and_normalize_points_batch(points):
    b, n, d = points.shape
    assert d == 2
    centroid = torch.mean(points, dim=1)

    centered = points - centroid[:,None,:]
    rms = torch.sqrt(torch.sum(centered ** 2,dim=(1,2)) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    xy_abs = abs(centered).sum(1)
    valid_mask = (xy_abs[:,0] > 0) * (xy_abs[:,1] > 0)
    # if rms == 0:
    #     return np.full((b, d + 1, d + 1), np.nan), np.full_like(points, np.nan)
    rms[~(rms>0)] = 1e-6
    norm_factor = np.sqrt(d) / rms
    part_matrix = norm_factor[:,None,None]*torch.cat([torch.eye(d)[None,:,:].repeat(b,1,1).to(centroid.device),-centroid[:, :, None]], dim=2)
    matrix = torch.cat([part_matrix, torch.tensor([[[0,0,1]]]).repeat(b,1,1).to(part_matrix.device)], dim=1)

    points_h = torch.cat([points, torch.ones((b,n,1)).to(points.device)],dim=2)

    new_points_h = torch.einsum('bmd,bnd->bnm',matrix,points_h)

    new_points = new_points_h[:, :, :d]
    new_points /= new_points_h[:, :, d:]
    
    # matrix = matrix.float()
    # new_points = new_points.float()
    # matrix = torch.where(valid_mask[:,None,None], matrix,  torch.full((b, d + 1, d + 1), float('nan')).to(matrix.device))
    # new_points = torch.where(valid_mask[:,None,None], new_points, torch.full_like(new_points, float('nan')).to(new_points.device))
    
    # part_matrix = norm_factor * np.concatenate(
    #         (np.eye(d), -centroid[:, np.newaxis]), axis=1
    #         )
    # matrix = np.concatenate(
    #         (part_matrix, [[0,] * d + [1]]), axis=0
    #         )

    # points_h = np.vstack([points.T, np.ones(n)])

    # new_points_h = (matrix @ points_h).T

    # new_points = new_points_h[:, :d]
    # new_points /= new_points_h[:, d:]

    return matrix, new_points, valid_mask

class BaseModel(object):

    def __init__(self):
        self.params = None

class LineModelND(BaseModel):
    """Total least squares estimator for N-dimensional lines.

    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.

    Lines are defined by a point (origin) and a unit vector (direction)
    according to the following vector equation::

        X = origin + lambda * direction

    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `direction`.

    Examples
    --------
    >>> x = np.linspace(1, 2, 25)
    >>> y = 1.5 * x + 3
    >>> lm = LineModelND()
    >>> lm.estimate(np.stack([x, y], axis=-1))
    True
    >>> tuple(np.round(lm.params, 5))
    (array([1.5 , 5.25]), array([0.5547 , 0.83205]))
    >>> res = lm.residuals(np.stack([x, y], axis=-1))
    >>> np.abs(np.round(res, 9))
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    >>> np.round(lm.predict_y(x[:5]), 3)
    array([4.5  , 4.562, 4.625, 4.688, 4.75 ])
    >>> np.round(lm.predict_x(y[:5]), 3)
    array([1.   , 1.042, 1.083, 1.125, 1.167])

    """

    def estimate(self, data):
        """Estimate line model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        _check_data_atleast_2D(data)

        origin = data.mean(axis=0)
        data = data - origin

        if data.shape[0] == 2:  # well determined
            direction = data[1] - data[0]
            norm = np.linalg.norm(direction)
            if norm != 0:  # this should not happen to be norm 0
                direction /= norm
        elif data.shape[0] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _, _, v = np.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:  # under-determined
            raise ValueError('At least 2 input points needed.')

        self.params = (origin, direction)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        _check_data_atleast_2D(data)
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params
        res = (data - origin) - \
              ((data - origin) @ direction)[..., np.newaxis] * direction
        return _norm_along_axis(res, axis=1)

    def predict(self, x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.

        Parameters
        ----------
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        data : (n, m) array
            Predicted coordinates.

        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params

        if direction[axis] == 0:
            # line parallel to axis
            raise ValueError('Line parallel to axis %s' % axis)

        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., np.newaxis] * direction
        return data

    def predict_x(self, y, params=None):
        """Predict x-coordinates for 2D lines using the estimated model.

        Alias for::

            predict(y, axis=1)[:, 0]

        Parameters
        ----------
        y : array
            y-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        x : array
            Predicted x-coordinates.

        """
        x = self.predict(y, axis=1, params=params)[:, 0]
        return x

    def predict_y(self, x, params=None):
        """Predict y-coordinates for 2D lines using the estimated model.

        Alias for::

            predict(x, axis=0)[:, 1]

        Parameters
        ----------
        x : array
            x-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        y : array
            Predicted y-coordinates.

        """
        y = self.predict(x, axis=0, params=params)[:, 1]
        return y

class CircleModel(BaseModel):

    """Total least squares estimator for 2D circles.

    The functional model of the circle is::

        r**2 = (x - xc)**2 + (y - yc)**2

    This estimator minimizes the squared distances from all points to the
    circle::

        min{ sum((r - sqrt((x_i - xc)**2 + (y_i - yc)**2))**2) }

    A minimum number of 3 points is required to solve for the parameters.

    Attributes
    ----------
    params : tuple
        Circle model parameters in the following order `xc`, `yc`, `r`.

    Examples
    --------
    >>> t = np.linspace(0, 2 * np.pi, 25)
    >>> xy = CircleModel().predict_xy(t, params=(2, 3, 4))
    >>> model = CircleModel()
    >>> model.estimate(xy)
    True
    >>> tuple(np.round(model.params, 5))
    (2.0, 3.0, 4.0)
    >>> res = model.residuals(xy)
    >>> np.abs(np.round(res, 9))
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def estimate(self, data):
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]

        # http://www.had2know.com/academics/best-fit-circle-least-squares.html
        x2y2 = (x ** 2 + y ** 2)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        m1 = np.stack([[np.sum(x ** 2), sum_xy, sum_x],
                       [sum_xy, np.sum(y ** 2), sum_y],
                       [sum_x, sum_y, float(len(x))]])
        m2 = np.stack([[np.sum(x * x2y2),
                        np.sum(y * x2y2),
                        np.sum(x2y2)]], axis=-1)
        a, b, c = pinv(m1) @ m2
        a, b, c = a[0], b[0], c[0]
        xc = a / 2
        yc = b / 2
        r = np.sqrt(4 * c + a ** 2 + b ** 2) / 2

        self.params = (xc, yc, r)

        return True

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the circle is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        """

        _check_data_dim(data, dim=2)

        xc, yc, r = self.params

        x = data[:, 0]
        y = data[:, 1]

        return r - np.sqrt((x - xc)**2 + (y - yc)**2)

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (3, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """
        if params is None:
            params = self.params
        xc, yc, r = params

        x = xc + r * np.cos(t)
        y = yc + r * np.sin(t)

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)

class EllipseModel(BaseModel):
    """Total least squares estimator for 2D ellipses.

    The functional model of the ellipse is::

        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)

    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.

    The estimator is based on a least squares minimization. The optimal
    solution is computed directly, no iterations are required. This leads
    to a simple, stable and robust fitting method.

    The ``params`` attribute contains the parameters in the following order::

        xc, yc, a, b, theta

    Attributes
    ----------
    params : tuple
        Ellipse model parameters in the following order `xc`, `yc`, `a`, `b`,
        `theta`.

    Examples
    --------

    >>> xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25),
    ...                                params=(10, 15, 4, 8, np.deg2rad(30)))
    >>> ellipse = EllipseModel()
    >>> ellipse.estimate(xy)
    True
    >>> np.round(ellipse.params, 2)
    array([10.  , 15.  ,  4.  ,  8.  ,  0.52])
    >>> np.round(abs(ellipse.residuals(xy)), 5)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def estimate(self, data):
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.


        References
        ----------
        .. [1] Halir, R.; Flusser, J. "Numerically stable direct least squares
               fitting of ellipses". In Proc. 6th International Conference in
               Central Europe on Computer Graphics and Visualization.
               WSCG (Vol. 98, pp. 125-132).

        """
        # Original Implementation: Ben Hammel, Nick Sullivan-Molina
        # another REFERENCE: [2] http://mathworld.wolfram.com/Ellipse.html
        _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]

        # Quadratic part of design matrix [eqn. 15] from [1]
        D1 = np.vstack([x ** 2, x * y, y ** 2]).T
        # Linear part of design matrix [eqn. 16] from [1]
        D2 = np.vstack([x, y, np.ones(len(x))]).T

        # forming scatter matrix [eqn. 17] from [1]
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2

        # Constraint matrix [eqn. 18]
        C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

        try:
            # Reduced scatter matrix [eqn. 29]
            M = inv(C1) @ (S1 - S2 @ inv(S3) @ S2.T)
        except np.linalg.LinAlgError:  # LinAlgError: Singular matrix
            return False

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors
        # from this equation [eqn. 28]
        eig_vals, eig_vecs = np.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[2, :]) \
               - np.power(eig_vecs[1, :], 2)
        a1 = eig_vecs[:, (cond > 0)]
        # seeks for empty matrix
        if 0 in a1.shape or len(a1.ravel()) != 3:
            return False
        a, b, c = a1.ravel()

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -inv(S3) @ S2.T @ a1
        d, f, g = a2.ravel()

        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 (eqn. 15) from [2]
        b /= 2.
        d /= 2.
        f /= 2.

        # finding center of ellipse [eqn.19 and 20] from [2]
        x0 = (c * d - b * f) / (b ** 2. - a * c)
        y0 = (a * f - b * d) / (b ** 2. - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from [2]
        numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 \
                    - 2 * b * d * f - a * c * g
        term = np.sqrt((a - c) ** 2 + 4 * b ** 2)
        denominator1 = (b ** 2 - a * c) * (term - (a + c))
        denominator2 = (b ** 2 - a * c) * (- term - (a + c))
        width = np.sqrt(2 * numerator / denominator1)
        height = np.sqrt(2 * numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse
        # to x-axis [eqn. 23] from [2].
        phi = 0.5 * np.arctan((2. * b) / (a - c))
        if a > c:
            phi += 0.5 * np.pi

        self.params = np.nan_to_num([x0, y0, width, height, phi]).tolist()
        self.params = [float(np.real(x)) for x in self.params]
        return True

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the ellipse is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        """

        _check_data_dim(data, dim=2)

        xc, yc, a, b, theta = self.params

        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        def fun(t, xi, yi):
            ct = math.cos(t)
            st = math.sin(t)
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt) ** 2 + (yi - yt) ** 2

        # def Dfun(t, xi, yi):
        #     ct = math.cos(t)
        #     st = math.sin(t)
        #     xt = xc + a * ctheta * ct - b * stheta * st
        #     yt = yc + a * stheta * ct + b * ctheta * st
        #     dfx_t = - 2 * (xi - xt) * (- a * ctheta * st
        #                                - b * stheta * ct)
        #     dfy_t = - 2 * (yi - yt) * (- a * stheta * st
        #                                + b * ctheta * ct)
        #     return [dfx_t + dfy_t]

        residuals = np.empty((N, ), dtype=np.double)

        # initial guess for parameter t of closest point on ellipse
        t0 = np.arctan2(y - yc, x - xc) - theta

        # determine shortest distance to ellipse for each point
        for i in range(N):
            xi = x[i]
            yi = y[i]
            # faster without Dfun, because of the python overhead
            t, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
            residuals[i] = np.sqrt(fun(t, xi, yi))

        return residuals

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """

        if params is None:
            params = self.params

        xc, yc, a, b, theta = params

        ct = np.cos(t)
        st = np.sin(t)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)

def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.
    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.
    Returns
    -------
    trials : int
        Number of trials.
    """
    if n_inliers == 0:
        return np.inf

    nom = 1 - probability
    if nom == 0:
        return np.inf

    inlier_ratio = n_inliers / float(n_samples)
    denom = 1 - inlier_ratio ** min_samples
    if denom == 0:
        return 1
    elif denom == 1:
        return np.inf

    nom = np.log(nom)
    denom = np.log(denom)
    if denom == 0:
        return 0

    return int(np.ceil(nom / denom))

def _check_rationality(model):
    #对缩放因子和剪切角进行限制
    (_model_scale_x,_model_scale_y) = model.scale
    _model_shear = model.shear
    
    if _model_scale_x > 1.2 or _model_scale_x < 0.8:
        return False
    if _model_scale_y > 1.2 or _model_scale_y < 0.8:
        return False
    if _model_shear > 0.1745 or _model_shear < -0.1745: #10*(pi/180)
        return False
    return True
    
    
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:

    1. Select `min_samples` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.

    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.

    Parameters
    ----------
    data : [list, tuple of] (N, ...) array
        Data set to which the model is fitted, where N is the number of data
        points and the remaining dimension are depending on model requirements.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:

         * ``success = estimate(*data)``
         * ``residuals(*data)``

        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples : int in range (0, N)
        The minimum number of data points to fit a model to.
    residual_threshold : float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to a high value
        such as 0.99, e is the current fraction of inliers w.r.t. the
        total number of samples, and m is the min_samples value.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    initial_inliers : array-like of bool, shape (N,), optional
        Initial samples selection for model estimation


    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N, ) array
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] "RANSAC", Wikipedia, https://en.wikipedia.org/wiki/RANSAC

    Examples
    --------

    Generate ellipse data without tilt and add noise:

    >>> t = np.linspace(0, 2 * np.pi, 50)
    >>> xc, yc = 20, 30
    >>> a, b = 5, 10
    >>> x = xc + a * np.cos(t)
    >>> y = yc + b * np.sin(t)
    >>> data = np.column_stack([x, y])
    >>> np.random.seed(seed=1234)
    >>> data += np.random.normal(size=data.shape)

    Add some faulty data:

    >>> data[0] = (100, 100)
    >>> data[1] = (110, 120)
    >>> data[2] = (120, 130)
    >>> data[3] = (140, 130)

    Estimate ellipse model using all available data:

    >>> model = EllipseModel()
    >>> model.estimate(data)
    True
    >>> np.round(model.params)  # doctest: +SKIP
    array([ 72.,  75.,  77.,  14.,   1.])

    Estimate ellipse model using RANSAC:

    >>> ransac_model, inliers = ransac(data, EllipseModel, 20, 3, max_trials=50)
    >>> abs(np.round(ransac_model.params))
    array([20., 30.,  5., 10.,  0.])
    >>> inliers # doctest: +SKIP
    array([False, False, False, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True], dtype=bool)
    >>> sum(inliers) > 40
    True

    RANSAC can be used to robustly estimate a geometric transformation. In this section,
    we also show how to use a proportion of the total samples, rather than an absolute number.

    >>> from skimage.transform import SimilarityTransform
    >>> np.random.seed(0)
    >>> src = 100 * np.random.rand(50, 2)
    >>> model0 = SimilarityTransform(scale=0.5, rotation=1, translation=(10, 20))
    >>> dst = model0(src)
    >>> dst[0] = (10000, 10000)
    >>> dst[1] = (-100, 100)
    >>> dst[2] = (50, 50)
    >>> ratio = 0.5  # use half of the samples
    >>> min_samples = int(ratio * len(src))
    >>> model, inliers = ransac((src, dst), SimilarityTransform, min_samples, 10,
    ...                         initial_inliers=np.ones(len(src), dtype=bool))
    >>> inliers
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True])

    """
def slransac_slow(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None, weight_pairs=None, angles=None):


    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    #random_state = check_random_state(random_state)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])

    if not (0 < min_samples < num_samples):
        raise ValueError("`min_samples` must be in range (0, <number-of-samples>)")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), num_samples))

    # for the first run use initial guess of inliers
    # spl_idxs = (initial_inliers if initial_inliers is not None
    #             else random_state.choice(num_samples, min_samples, replace=False))
        
    def get_angle_diff(angles):
        pi_coef = 3.1415926
        diff = np.zeros([angles[0].shape[0], 4])

        diff[:,0] = angles[0] - angles[1] #[-pi,pi]
        mask = diff[:,0] < 0
        diff[mask, 0] = diff[mask, 0] + 2 * pi_coef  #[0,2pi]
        diff[:,1] = 2 * pi_coef - diff[:,0]

        diff[:,2] = pi_coef + angles[0] - angles[1] #[0,2*pi]  #避免0,180干扰
        diff[:,3] = 2 * pi_coef - diff[:,2] #[0,2*pi]

        diff_c = np.min(diff,axis = 1)  

        return diff_c

    def _check_sample_angle(sample_angles_diff):
        pi_coef = 3.1415926
        if np.max(np.abs(sample_angles_diff - np.mean(sample_angles_diff))) > pi_coef/12 :  #误差不超过+-15度
            return False
        return True
    
    def _check_sample_dist(x = 1,y = 1):
        if x < 5/6 or x > 6/5:
            return False
        if y < 5/6 or y > 6/5:
            return False
        return True

    def is_data_valid(scale2_matrix, angle_diff, a, b, c):
        if not _check_sample_angle(angle_diff[[a,b,c]]):
           return False
        if not _check_sample_dist(x = scale2_matrix[b,c], y = scale2_matrix[c,a]):
            return False
        return True

    #def is_model_valid(sample_model, *samples):
    #    return _check_rationality(sample_model)
    
    dis_sum0 = np.square(data[0][:,None] - data[0]).sum(axis = 2)
    dis_sum1 = np.square(data[1][:,None] - data[1]).sum(axis = 2)
    dis_sum1[range(num_samples), range(num_samples)] = 1   #对角线防止除0
    scale2_matrix = dis_sum0 / dis_sum1

    angle_diff = get_angle_diff(angles)

    num_trials = 0
    for a in range(num_samples-2):
        for b in range(a+1, num_samples-1):
            x = scale2_matrix[a,b]  
            if x < 5/6 or x > 6/5:      #sample点合理性限制
                continue
            for c in range(b+1, num_samples):
                # do sample selection according data pairs
                spl_idxs = np.array([a,b,c])
                samples = [d[spl_idxs] for d in data]

                # for next iteration choose random sample set and be sure that no samples repeat
                #spl_idxs = random_state.choice(num_samples, min_samples, replace=False)

                num_trials = num_trials + 1  #按拟合次数算 目前工程按进c的次数算
                if num_trials > max_trials:
                    break

                # optional check if random sample set is valid
                if is_data_valid is not None and not is_data_valid(scale2_matrix, angle_diff, a,b,c): 
                    continue

                # estimate model for current random sample set
                sample_model = model_class()
        
                success = sample_model.estimate(*samples)

                # backwards compatibility
                if success is not None and not success:
                    continue
        
                # optional check if estimated model is valid
                if is_model_valid is not None and not is_model_valid(sample_model, *samples):
                    continue
                
                ##根据现有模型计算每个点的变换点，然后计算与匹配点的欧氏距离
                # if a == 1 and b == 13 and c == 22:
                #     sample_model.params[0,0] = 256./256
                #     sample_model.params[0,1] = 44./256
                #     sample_model.params[0,2] = -4227./256
                #     sample_model.params[1,0] = -43./256
                #     sample_model.params[1,1] = 246./256
                #     sample_model.params[1,2] = 1413./256
                #     #print(sample_model(data[0])*256)
                #     #print(np.sum(np.sum((sample_model(data[0])*256 - data[1]*256)**2, axis=1)))

                sample_model_residuals = np.abs(sample_model.residuals(*data))
                # consensus set / inliers
                sample_model_inliers = sample_model_residuals < residual_threshold
                sample_model_residuals_sum = np.sum((sample_model_residuals[sample_model_inliers]) ** 2)

                # choose as new best model if number of inliers is maximal
                sample_inlier_num = np.sum(sample_model_inliers)
                # if sample_inlier_num == 0:
                #     continue
                # sample_model_residuals = sample_model_residuals_sum / sample_inlier_num
                #print(num_trials, a,b,c,best_inlier_num, sample_inlier_num, sample_model_residuals_sum*256*256)

                #对内点进行匹配分数加权
                if weight_pairs is not None:
                    weight_pairs_inlier = weight_pairs[sample_model_inliers]
                    inliers_score = np.mean(weight_pairs_inlier)
                    sample_inlier_num = sample_inlier_num*inliers_score
        
                if (
                    (# more inliers
                    sample_inlier_num > best_inlier_num
                    # same number of inliers but less "error" in terms of residuals
                    or (sample_inlier_num == best_inlier_num
                        and sample_model_residuals_sum < best_inlier_residuals_sum))
                    and np.linalg.matrix_rank(sample_model.params) == 3
                ):
                    #print(a,b,c,best_inlier_num, sample_inlier_num)
                    best_model = sample_model
                    best_inlier_num = sample_inlier_num
                    best_inlier_residuals_sum = sample_model_residuals_sum
                    best_inliers = sample_model_inliers
                    dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                             num_samples,
                                                             min_samples,
                                                             stop_probability)
                    if (best_inlier_num >= stop_sample_num):  
                        #or best_inlier_residuals_sum < stop_residuals_sum or num_trials >= dynamic_max_trials
                        break

                if num_trials >= max_trials or best_inlier_num >= stop_sample_num:
                    break
            if num_trials >= max_trials or best_inlier_num >= stop_sample_num:
                break
        if num_trials >= max_trials or best_inlier_num >= stop_sample_num:
            break
    '''
    # estimate final model using all inliers
    if best_inliers is not None and any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)
    else:
        best_model = None
        best_inliers = None
        #warn("No inliers found. Model not fitted")
    '''
    return best_model, best_inliers, best_inlier_residuals_sum

class slmodel_batch():
    def __init__(self, matrix=None):
        if matrix is None:
            # default to an identity transform
            matrix = np.eye(3)
        if matrix.shape != (3, 3):
            raise ValueError("invalid shape of transformation matrix")
        self.params = matrix
        self._coeffs = range(6)
    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        # coords = np.array(coords, copy=False, ndmin=2)

        x, y = coords[:,:,0][:,:,None], coords[:,:,1][:,:,None]
        src = torch.cat([x, y, torch.ones_like(x)],dim=2)
        dst = torch.einsum('bnc,bdc->bnd',src, matrix)

        # below, we will divide by the last dimension of the homogeneous
        # coordinate matrix. In order to avoid division by zero,
        # we replace exact zeros in this column with a very small number.
        
        zero_add = (dst[:,:,2] == 0)*torch.finfo(float).eps
        dst[:,:,2] += zero_add
        # rescale to homogeneous coordinates
        dst[:,:,:2] /= dst[:,:,2:3]

        return dst[:,:,:2]

    def __call__(self, coords, trans):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Destination coordinates.

        """
        return self._apply_mat(coords, trans)

    def residuals(self, data, trans):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the Euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        data : (2, N, 2) tensor
            Source coordinates.
        trans : (b, 3, 3) tensor
            Destination coordinates.

        Returns
        -------
        residuals : (b, N, ) tensor
            Residual for coordinate.

        """
        src, dst = data[:,0], data[:,1]
        return torch.sqrt(torch.sum((self(src, trans) - dst)**2, axis=2))

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Destination coordinates.

        Returns
        -------
        coords : (N, 2) array
            Source coordinates.

        """
        return self._apply_mat(coords, self._inv_matrix)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
            Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)

        These equations can be transformed to the following form::

            0 = a0*x + a1*y + a2 - c0*x*X - c1*y*X - X
            0 = b0*x + b1*y + b2 - c0*x*Y - c1*y*Y - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x y 1 0 0 0 -x*X -y*X -X]
                   [0 0 0 x y 1 -x*Y -y*Y -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c0 c1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        In case of the affine transformation the coefficients c0 and c1 are 0.
        Thus the system of equations is::

            A   = [[x y 1 0 0 0 -X]
                   [0 0 0 x y 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c3]

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        # try:
        #     src_matrix, src = _center_and_normalize_points(src)
        #     dst_matrix, dst = _center_and_normalize_points(dst)
        # except ZeroDivisionError:
        #     self.params = np.nan * np.empty((3, 3))
            # return False
        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
        if not np.all(np.isfinite(src_matrix + dst_matrix)):
            self.params = np.nan * np.empty((3, 3))
            return False

        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]

        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = np.zeros((rows * 2, 9))
        A[:rows, 0] = xs
        A[:rows, 1] = ys
        A[:rows, 2] = 1
        A[:rows, 6] = - xd * xs
        A[:rows, 7] = - xd * ys
        A[rows:, 3] = xs
        A[rows:, 4] = ys
        A[rows:, 5] = 1
        A[rows:, 6] = - yd * xs
        A[rows:, 7] = - yd * ys
        A[:rows, 8] = xd
        A[rows:, 8] = yd

        # Select relevant columns, depending on params
        A = A[:, list(self._coeffs) + [8]]

        _, _, V = np.linalg.svd(A)
        # if the last element of the vector corresponding to the smallest
        # singular value is close to zero, this implies a degenerate case
        # because it is a rank-defective transform, which would map points
        # to a line rather than a plane.
        if np.isclose(V[-1, -1], 0):
            return False

        H = np.zeros((3, 3))
        # solution is right singular vector that corresponds to smallest
        # singular value
        H.flat[list(self._coeffs) + [8]] = - V[-1, :-1] / V[-1, -1]
        H[2, 2] = 1

        # De-center and de-normalize
        H = np.linalg.inv(dst_matrix) @ H @ src_matrix

        self.params = H

        return True

    def estimate_batch(self, samples, data):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
            Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)

        These equations can be transformed to the following form::

            0 = a0*x + a1*y + a2 - c0*x*X - c1*y*X - X
            0 = b0*x + b1*y + b2 - c0*x*Y - c1*y*Y - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x y 1 0 0 0 -x*X -y*X -X]
                    [0 0 0 x y 1 -x*Y -y*Y -Y]
                    ...
                    ...
                    ]
            x.T = [a0 a1 a2 b0 b1 b2 c0 c1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalised, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        In case of the affine transformation the coefficients c0 and c1 are 0.
        Thus the system of equations is::

            A   = [[x y 1 0 0 0 -X]
                    [0 0 0 x y 1 -Y]
                    ...
                    ...
                    ]
            x.T = [a0 a1 a2 b0 b1 b2 c3]

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        '''
        samples: [b 4(batch_idx a b c)]
        data: [b 2 N 2]
        '''
        if len(samples) == 0:
            return [], []
        data = data.double()
        batchsize,_,s_n,_ =  data.size()

        samples_index = samples[:,1:] + (samples[:,0]*s_n)[:,None]
        _src = data[:,0].reshape(-1,2)[samples_index]
        _dst = data[:,1].reshape(-1,2)[samples_index]
        b, n, d = _src.shape

        src_matrix, src, valid_mask_s = _center_and_normalize_points_batch(_src)
        dst_matrix, dst, valid_mask_d = _center_and_normalize_points_batch(_dst)
        valid_mask = valid_mask_s*valid_mask_d
        dst[~valid_mask] = src[~valid_mask] #过滤有问题输入，防止SVD GG

        A = torch.zeros((b,n*d,(d+1) ** 2)).double().to(src.device)
        assert d == 2
        # for ddim in range(d):
        #     A[:, ddim*n : (ddim+1)*n, ddim*(d+1) : ddim*(d+1) + d] = src
        #     A[:, ddim*n : (ddim+1)*n, ddim*(d+1) + d] = 1
        #     A[:, ddim*n : (ddim+1)*n, -d-1:-1] = src
        #     A[:, ddim*n : (ddim+1)*n, -1] = -1
        #     A[:, ddim*n : (ddim+1)*n, -d-1:] *= -dst[:, :, ddim:(ddim+1)]
        
        ddim = 0
        A[:, ddim*n : (ddim+1)*n, ddim*(d+1) : ddim*(d+1) + d] = src
        A[:, ddim*n : (ddim+1)*n, ddim*(d+1) + d] = 1
        A[:, ddim*n : (ddim+1)*n, -d-1:-1] = src
        A[:, ddim*n : (ddim+1)*n, -1] = -1
        A[:, ddim*n : (ddim+1)*n, -d-1:] *= -dst[:, :, ddim:(ddim+1)]

        ddim = 1
        A[:, ddim*n : (ddim+1)*n, ddim*(d+1) : ddim*(d+1) + d] = src
        A[:, ddim*n : (ddim+1)*n, ddim*(d+1) + d] = 1
        A[:, ddim*n : (ddim+1)*n, -d-1:-1] = src
        A[:, ddim*n : (ddim+1)*n, -1] = -1
        A[:, ddim*n : (ddim+1)*n, -d-1:] *= -dst[:, :, ddim:(ddim+1)]

        # Select relevant columns, depending on params
        A = A[:, :, list(self._coeffs) + [-1]]

        # Get the vectors that correspond to singular values, also applying
        # the weighting if provided
      
        _, _, V = torch.linalg.svd(A)


        # if the last element of the vector corresponding to the smallest
        # singular value is close to zero, this implies a degenerate case
        # because it is a rank-defective transform, which would map points
        # to a line rather than a plane.
        nonzeros_mask = (~torch.isclose(V[:, -1, -1], torch.zeros_like(V[:, -1, -1])))

        H = torch.zeros((b, d+1, d+1)).double().to(src.device)
        # solution is right singular vector that corresponds to smallest
        # singular value
        H[:,:d,:d+1] = - (V[:, -1, :-1] / V[:, -1, -1][:,None]).view(b,d,d+1)
        H[:,d, d] = 1

        # De-center and de-normalize
        H = torch.linalg.inv(dst_matrix) @ H @ src_matrix

        # Small errors can creep in if points are not exact, causing the last
        # element of H to deviate from unity. Correct for that here.
        H /= H[:, -1, -1][:,None,None].clone()

        valid_mask *= nonzeros_mask
        H = torch.where(valid_mask[:,None,None], H, torch.eye(d+1)[None,:,:].repeat(b,1,1).double().to(src.device))

        return H, valid_mask
    
 
def slransac_batch_svd(bdata, mask, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None, weight_pairs=None, bangles=None):

    # in case data is not pair of input and output, male it like it
    # if not isinstance(bdata, (tuple, list)):
    #     bdata = (bdata, )

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), num_samples))

    # for the first run use initial guess of inliers
    # spl_idxs = (initial_inliers if initial_inliers is not None
    #             else random_state.choice(num_samples, min_samples, replace=False))
        

    # bdata = torch.from_numpy(np.array([data]))  #bdata： batch x 2 x N x 2  mask: batch x N
    # bdata = torch.cat([bdata[0][:,None,:,:],bdata[1][:,None,:,:]],dim=1)
    # bangles = torch.cat([angles[0][:,None,:],angles[1][:,None,:]],dim=1)

    num_samples = bdata.shape[2]
    batch = bdata.shape[0]

    best_model = torch.eye(3,device=bdata.device)[None,:].repeat(batch,1,1).double()
    best_inlier_num = torch.zeros(batch,device=bdata.device)
    best_inlier_residuals_sum = torch.ones(batch,device=bdata.device)*float('inf')
    best_inliers = torch.zeros_like(mask)
    model_valid = torch.zeros(batch,device=bdata.device)


    # mask = torch.ones(batch, num_samples)
    mask0 = mask.unsqueeze(1)
    mask1 = mask0.permute(0,2,1)
    mask_matrix = (mask1@mask0).bool()
    dis_sum0_batch = torch.square(bdata[:,0][:,:,None] - bdata[:, 0][:,None,:]).sum(axis = 3)
    dis_sum0_batch[~mask_matrix] = 0   #mask位置置0，距离比值也置0，会被距离阈值筛选掉
    dis_sum1_batch = torch.square(bdata[:,1][:,:,None] - bdata[:, 1][:,None,:]).sum(axis = 3)
    dis_sum1_batch[~mask_matrix] = 1
    dis_sum1_batch[:, range(num_samples), range(num_samples)] = 1
    scale2_matrix_batch = dis_sum0_batch/dis_sum1_batch
    scale2_matrix_batch = torch.triu(scale2_matrix_batch)
    bmask = (scale2_matrix_batch >= 5/6) * (scale2_matrix_batch <= 6/5)
    bmaskac = bmask.repeat(1,1,num_samples).reshape(batch, num_samples, num_samples, num_samples)  #[a,1,c]
    bmaskab = bmask.reshape(-1, 1).repeat(1,num_samples).reshape(batch, num_samples, num_samples, num_samples)  #[a,b,1]
    bmaskbc = bmask.repeat(1,num_samples,1).reshape(batch, num_samples, num_samples, num_samples)   #[1,b,c]
    dist_mask_batch = bmaskac * bmaskab * bmaskbc
    bmaskab = torch.triu(bmaskab, 1)
    bcumsum_ab = bmaskab.reshape(batch,-1).cumsum(1).reshape(batch,num_samples,num_samples,num_samples)
    bmaskab[bcumsum_ab > max_trials] = False
    blist_ab = bmaskab.reshape(batch,num_samples,num_samples,num_samples).nonzero()
    dist_mask_batch_after_ab = dist_mask_batch[blist_ab[:, 0],blist_ab[:, 1], blist_ab[:, 2], blist_ab[:, 3]]
    blist_abc = blist_ab[dist_mask_batch_after_ab]

    # print(list_abc == np.array(blist_abc[:,1:4]))

    def get_angle_diff_batch(angles):
        pi_coef = 3.1415926
        diff = torch.zeros([angles.shape[0], 4, angles.shape[2]],device=angles.device)

        diff[:,0] = angles[:,0,:] - angles[:,1,:] #[-pi,pi]
        mask = diff[:,0,:] < 0
        diff[mask, 0] = diff[mask, 0] + 2 * pi_coef  #[0,2pi]
        diff[:,1] = 2 * pi_coef - diff[:,0]

        diff[:,2] = pi_coef + angles[:,0,:] - angles[:,1,:] #[0,2*pi]  #避免0,180干扰
        diff[:,3] = 2 * pi_coef - diff[:,2] #[0,2*pi]

        diff_c = torch.min(diff,axis = 1)  

        return diff_c

    if blist_abc.shape[0] == 0:
        return best_model, best_inliers, best_inlier_residuals_sum, model_valid

    bangle_diff = get_angle_diff_batch(bangles)
    pi_coef = 3.1415926
    sample_angles_diff_batch = bangle_diff.values[blist_abc[:,0].unsqueeze(1), blist_abc[:,1:4]]  #第1维度的索引转置后广播为三倍
    sample_angles_diff_batch = torch.abs(sample_angles_diff_batch - torch.mean(sample_angles_diff_batch, axis = 1)[:,None])
    sample_angles_diff_mask_batch = torch.max(sample_angles_diff_batch,axis = 1).values > pi_coef/12
    sample_angles_check_batch = blist_abc[~sample_angles_diff_mask_batch]
    
    if len(sample_angles_check_batch) == 0:
        return best_model, best_inliers, best_inlier_residuals_sum, model_valid
    sample_model = slmodel_batch()
    bdata = bdata.double()
    H_all, success_all = sample_model.estimate_batch(sample_angles_check_batch, bdata)
    #去除无效数据
    H_valid_all = (torch.linalg.matrix_rank(H_all) == 3)
    H_valid_all *= success_all
    sample_angles_check_batch = sample_angles_check_batch[H_valid_all]
    if len(sample_angles_check_batch) == 0:
        return best_model, best_inliers, best_inlier_residuals_sum, model_valid
    H_all = H_all[H_valid_all]

    bdata_all = bdata[sample_angles_check_batch[:,0]]
    mask_all = mask[sample_angles_check_batch[:,0]]
    sample_model_residuals_all = torch.abs(sample_model.residuals(bdata_all,H_all))
    # consensus set / inliers
    sample_model_inliers_all = (sample_model_residuals_all < residual_threshold)*mask_all
    sample_model_residuals_sum_all = torch.sum((sample_model_residuals_all*sample_model_inliers_all) ** 2,dim=1)

    # choose as new best model if number of inliers is maximal
    sample_inlier_num_all = torch.sum(sample_model_inliers_all,1)


    sample_batch_index = sample_angles_check_batch[:,0].cpu().numpy()
    H_all_np = H_all.cpu().numpy()
    sample_model_inliers_all_np = sample_model_inliers_all.cpu().numpy()
    sample_model_residuals_sum_all_np = sample_model_residuals_sum_all.cpu().numpy()
    sample_inlier_num_all_np = sample_inlier_num_all.cpu().numpy()

    #torch for循环很慢 转numpy
    best_model = best_model.cpu().numpy()
    best_inlier_num = best_inlier_num.cpu().numpy()
    best_inlier_residuals_sum = best_inlier_residuals_sum.cpu().numpy()
    best_inliers = best_inliers.cpu().numpy()
    model_valid = model_valid.cpu().numpy()

    for b_idx in range(batch):
        # do sample selection according data pairs
        #spl_idxs = idx
        sample_mask = sample_batch_index == b_idx
        H_batch = H_all_np[sample_mask]

        if len(H_batch) == 0: #没有trans或者trans全部失败直接跳过
            # model_valid[b_idx] = 0
            continue
        sample_model_inliers = sample_model_inliers_all_np[sample_mask]
        sample_model_residuals_sum = sample_model_residuals_sum_all_np[sample_mask]

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = sample_inlier_num_all_np[sample_mask]

        # if sample_inlier_num == 0:
        #     continue
        # sample_model_residuals = sample_model_residuals_sum / sample_inlier_num
        #print(idx, best_inlier_num, sample_inlier_num, sample_model_residuals_sum*256*256)

        #对内点进行匹配分数加权
        # if weight_pairs is not None:
        #     weight_pairs_inlier = weight_pairs[sample_model_inliers]
        #     inliers_score = np.mean(weight_pairs_inlier)
        #     sample_inlier_num = sample_inlier_num*inliers_score
        
        best_idx = 0
        for i in range(len(H_batch)):
            if (
                (# more inliers
                sample_inlier_num[i] > best_inlier_num[b_idx]
                # same number of inliers but less "error" in terms of residuals
                or (sample_inlier_num[i] == best_inlier_num[b_idx]
                    and sample_model_residuals_sum[i] < best_inlier_residuals_sum[b_idx]))
            ):
                #print(idx, best_inlier_num, sample_inlier_num)
                best_model[b_idx] = H_batch[i]
                best_inlier_num[b_idx] = sample_inlier_num[i]
                best_inlier_residuals_sum[b_idx] = sample_model_residuals_sum[i]
                best_idx = i
                model_valid[b_idx] = 1
                best_inliers[b_idx] = sample_model_inliers[i]
                # dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                #                                             num_samples,
                #                                             min_samples,
                #                                             stop_probability)
                if (best_inlier_num[b_idx] >= stop_sample_num):  
                    #or best_inlier_residuals_sum <= stop_residuals_sum or num_trials >= dynamic_max_trials
                    break

    best_model = torch.from_numpy(best_model).to(bdata.device)
    best_inlier_num = torch.from_numpy(best_inlier_num).to(bdata.device)
    best_inlier_residuals_sum = torch.from_numpy(best_inlier_residuals_sum).to(bdata.device)
    best_inliers = torch.from_numpy(best_inliers).to(bdata.device)
    model_valid = torch.from_numpy(model_valid).to(bdata.device)

    return best_model, best_inliers, best_inlier_residuals_sum, model_valid

def get_residuals(data, trans):
    """Determine residuals of transformed destination coordinates.

    For each transformed source coordinate the Euclidean distance to the
    respective destination coordinate is determined.

    Parameters
    ----------
    data : (2, N, 2) tensor
        Source coordinates.
    trans : (b, 3, 3) tensor
        Destination coordinates.

    Returns
    -------
    residuals : (b, N, ) tensor
        Residual for coordinate.

    """
    def _apply_mat(coords, matrix):
        # coords = np.array(coords, copy=False, ndmin=2)

        x, y = coords[:,:,0][:,:,None], coords[:,:,1][:,:,None]
        src = torch.cat([x, y, torch.ones_like(x)],dim=2)
        dst = torch.einsum('bnc,bdc->bnd',src, matrix)

        # below, we will divide by the last dimension of the homogeneous
        # coordinate matrix. In order to avoid division by zero,
        # we replace exact zeros in this column with a very small number.
        
        zero_add = (dst[:,:,2] == 0)*torch.finfo(float).eps
        dst[:,:,2] += zero_add
        # rescale to homogeneous coordinates
        dst[:,:,:2] /= dst[:,:,2:3]

        return dst[:,:,:2]

    src, dst = data[:,0], data[:,1]
    return torch.sqrt(torch.sum((_apply_mat(src, trans) - dst)**2, axis=2))

def slransac_batch(bdata, mask, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None, weight_pairs=None, bangles=None):

    # in case data is not pair of input and output, male it like it
    # if not isinstance(bdata, (tuple, list)):
    #     bdata = (bdata, )

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), num_samples))

    # for the first run use initial guess of inliers
    # spl_idxs = (initial_inliers if initial_inliers is not None
    #             else random_state.choice(num_samples, min_samples, replace=False))
        

    # bdata = torch.from_numpy(np.array([data]))  #bdata： batch x 2 x N x 2  mask: batch x N
    # bdata = torch.cat([bdata[0][:,None,:,:],bdata[1][:,None,:,:]],dim=1)
    # bangles = torch.cat([angles[0][:,None,:],angles[1][:,None,:]],dim=1)

    num_samples = bdata.shape[2]
    batch = bdata.shape[0]

    best_model = torch.eye(3,device=bdata.device)[None,:].repeat(batch,1,1).double()
    best_inlier_num = torch.zeros(batch,device=bdata.device)
    best_inlier_residuals_sum = torch.ones(batch,device=bdata.device)*float('inf')
    best_inliers = torch.zeros_like(mask)
    model_valid = torch.zeros(batch,device=bdata.device)


    # mask = torch.ones(batch, num_samples)
    mask0 = mask.unsqueeze(1)
    mask1 = mask0.permute(0,2,1)
    mask_matrix = (mask1@mask0).bool()
    delta_abc0_batch = bdata[:,0][:,:,None] - bdata[:, 0][:,None,:]
    delta_abc1_batch = bdata[:,1][:,:,None] - bdata[:, 1][:,None,:]
    dis_sum0_batch = torch.square(delta_abc0_batch).sum(axis = 3)
    dis_sum0_batch[~mask_matrix] = 0   #mask位置置0，距离比值也置0，会被距离阈值筛选掉
    dis_sum1_batch = torch.square(delta_abc1_batch).sum(axis = 3)
    dis_sum1_batch[~mask_matrix] = 1
    dis_sum1_batch[:, range(num_samples), range(num_samples)] = 1
    scale2_matrix_batch = dis_sum0_batch/dis_sum1_batch
    scale2_matrix_batch = torch.triu(scale2_matrix_batch)
    bmask = (scale2_matrix_batch >= 5/6) * (scale2_matrix_batch <= 6/5)
    bmaskac = bmask.repeat(1,1,num_samples).reshape(batch, num_samples, num_samples, num_samples)  #[a,1,c]
    bmaskab = bmask.reshape(-1, 1).repeat(1,num_samples).reshape(batch, num_samples, num_samples, num_samples)  #[a,b,1]
    bmaskbc = bmask.repeat(1,num_samples,1).reshape(batch, num_samples, num_samples, num_samples)   #[1,b,c]
    dist_mask_batch = bmaskac * bmaskab * bmaskbc
    bmaskab *= mask[:,None,None,:].bool()
    bmaskab = torch.triu(bmaskab, 1)
    bcumsum_ab = bmaskab.reshape(batch,-1).cumsum(1).reshape(batch,num_samples,num_samples,num_samples)
    bmaskab[bcumsum_ab > max_trials] = False
    blist_ab = bmaskab.reshape(batch,num_samples,num_samples,num_samples).nonzero()
    dist_mask_batch_after_ab = dist_mask_batch[blist_ab[:, 0],blist_ab[:, 1], blist_ab[:, 2], blist_ab[:, 3]]
    blist_abc = blist_ab[dist_mask_batch_after_ab]

    # print(list_abc == np.array(blist_abc[:,1:4]))

    def get_angle_diff_batch(angles):
        pi_coef = 3.1415926
        diff = torch.zeros([angles.shape[0], 4, angles.shape[2]],device=angles.device)

        diff[:,0] = angles[:,0,:] - angles[:,1,:] #[-pi,pi]
        mask = diff[:,0,:] < 0
        diff[mask, 0] = diff[mask, 0] + 2 * pi_coef  #[0,2pi]
        diff[:,1] = 2 * pi_coef - diff[:,0]

        diff[:,2] = pi_coef + angles[:,0,:] - angles[:,1,:] #[0,2*pi]  #避免0,180干扰
        diff[:,3] = 2 * pi_coef - diff[:,2] #[0,2*pi]

        diff_c = torch.min(diff,axis = 1)  

        return diff_c

    if blist_abc.shape[0] == 0:
        return best_model, best_inliers, best_inlier_residuals_sum, model_valid

    bangle_diff = get_angle_diff_batch(bangles)
    pi_coef = 3.1415926
    sample_angles_diff_batch = bangle_diff.values[blist_abc[:,0].unsqueeze(1), blist_abc[:,1:4]]  #第1维度的索引转置后广播为三倍
    sample_angles_diff_batch = torch.abs(sample_angles_diff_batch - torch.mean(sample_angles_diff_batch, axis = 1)[:,None])
    sample_angles_diff_mask_batch = torch.max(sample_angles_diff_batch,axis = 1).values > pi_coef/12
    sample_angles_check_batch = blist_abc[~sample_angles_diff_mask_batch]
    
    if len(sample_angles_check_batch) == 0:
        return best_model, best_inliers, best_inlier_residuals_sum, model_valid
    
    deltax1ab = delta_abc0_batch[sample_angles_check_batch[:,0],sample_angles_check_batch[:,1],sample_angles_check_batch[:,2],0]
    deltax1ac = delta_abc0_batch[sample_angles_check_batch[:,0],sample_angles_check_batch[:,1],sample_angles_check_batch[:,3],0]
    deltay1ab = delta_abc0_batch[sample_angles_check_batch[:,0],sample_angles_check_batch[:,1],sample_angles_check_batch[:,2],1]
    deltay1ac = delta_abc0_batch[sample_angles_check_batch[:,0],sample_angles_check_batch[:,1],sample_angles_check_batch[:,3],1]

    deltax2ab = delta_abc1_batch[sample_angles_check_batch[:,0],sample_angles_check_batch[:,1],sample_angles_check_batch[:,2],0]
    deltax2ac = delta_abc1_batch[sample_angles_check_batch[:,0],sample_angles_check_batch[:,1],sample_angles_check_batch[:,3],0]
    deltay2ab = delta_abc1_batch[sample_angles_check_batch[:,0],sample_angles_check_batch[:,1],sample_angles_check_batch[:,2],1]
    deltay2ac = delta_abc1_batch[sample_angles_check_batch[:,0],sample_angles_check_batch[:,1],sample_angles_check_batch[:,3],1]

    a0_coeff = deltax1ab * deltay1ac - deltax1ac * deltay1ab
    a1_coeff = -a0_coeff
    b0_coeff = a0_coeff
    b1_coeff = -b0_coeff
    success_all = (a0_coeff != 0) #共线判断
    success_all_idx = success_all.nonzero()[:,0]

    a0_result = deltax2ab * deltay1ac - deltax2ac * deltay1ab
    a0 = (a0_result[success_all]*256 / a0_coeff[success_all] + 0.5).int().double()
    a1_result = deltax2ab * deltax1ac - deltax2ac * deltax1ab
    a1 = (a1_result[success_all]*256 / a1_coeff[success_all] + 0.5).int().double()
    a2_result = bdata[sample_angles_check_batch[:,0],1,sample_angles_check_batch[:,1],0]
    a2 = a2_result[success_all]*256 - bdata[sample_angles_check_batch[:,0],0,sample_angles_check_batch[:,1],0][success_all]*a0 - bdata[sample_angles_check_batch[:,0],0,sample_angles_check_batch[:,1],1][success_all]*a1
                
    b0_result = deltay2ab * deltay1ac - deltay2ac * deltay1ab
    b0 = (b0_result[success_all]*256 / b0_coeff[success_all] + 0.5).int().double()
    b1_result = deltay2ab * deltax1ac - deltay2ac * deltax1ab
    b1 = (b1_result[success_all]*256 / b1_coeff[success_all] + 0.5).int().double()
    b2_result = bdata[sample_angles_check_batch[:,0],1,sample_angles_check_batch[:,1],1]
    b2 = b2_result[success_all]*256 - bdata[sample_angles_check_batch[:,0],0,sample_angles_check_batch[:,1],0][success_all]*b0 - bdata[sample_angles_check_batch[:,0],0,sample_angles_check_batch[:,1],1][success_all]*b1

    H_valid_all = success_all.clone()
    H_valid_all[success_all] = ((a0 * b1 - a1 * b0) != 0)

    H_all = torch.eye(3,device=bdata.device)[None,:].repeat(len(sample_angles_check_batch),1,1).double()
    H_all[success_all_idx,0,0] = a0 / 256 
    H_all[success_all_idx,0,1] = a1 / 256
    H_all[success_all_idx,0,2] = a2 / 256
    H_all[success_all_idx,1,0] = b0 / 256
    H_all[success_all_idx,1,1] = b1 / 256
    H_all[success_all_idx,1,2] = b2 / 256

    bdata = bdata.double()

    # SVD
    # sample_model = slmodel_batch()
    # H_all, success_all = sample_model.estimate_batch(sample_angles_check_batch, bdata)
    # #去除无效数据
    # H_valid_all = (torch.linalg.matrix_rank(H_all) == 3)
    # H_valid_all *= success_all

    sample_angles_check_batch = sample_angles_check_batch[H_valid_all]
    if len(sample_angles_check_batch) == 0:
        return best_model, best_inliers, best_inlier_residuals_sum, model_valid
    H_all = H_all[H_valid_all]

    bdata_all = bdata[sample_angles_check_batch[:,0]]
    mask_all = mask[sample_angles_check_batch[:,0]]
    sample_model_residuals_all = torch.abs(get_residuals(bdata_all,H_all))
    # consensus set / inliers
    sample_model_inliers_all = (sample_model_residuals_all < residual_threshold)*mask_all
    sample_model_residuals_sum_all = torch.sum((sample_model_residuals_all*sample_model_inliers_all) ** 2,dim=1)

    # choose as new best model if number of inliers is maximal
    sample_inlier_num_all = torch.sum(sample_model_inliers_all,1)


    sample_batch_index = sample_angles_check_batch[:,0].cpu().numpy()
    H_all_np = H_all.cpu().numpy()
    sample_model_inliers_all_np = sample_model_inliers_all.cpu().numpy()
    sample_model_residuals_sum_all_np = sample_model_residuals_sum_all.cpu().numpy()
    sample_inlier_num_all_np = sample_inlier_num_all.cpu().numpy()

    #torch for循环很慢 转numpy
    best_model = best_model.cpu().numpy()
    best_inlier_num = best_inlier_num.cpu().numpy()
    best_inlier_residuals_sum = best_inlier_residuals_sum.cpu().numpy()
    best_inliers = best_inliers.cpu().numpy()
    model_valid = model_valid.cpu().numpy()

    for b_idx in range(batch):
        # do sample selection according data pairs
        #spl_idxs = idx
        sample_mask = sample_batch_index == b_idx
        H_batch = H_all_np[sample_mask]

        if len(H_batch) == 0: #没有trans或者trans全部失败直接跳过
            # model_valid[b_idx] = 0
            continue
        sample_model_inliers = sample_model_inliers_all_np[sample_mask]
        sample_model_residuals_sum = sample_model_residuals_sum_all_np[sample_mask]

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = sample_inlier_num_all_np[sample_mask]

        # if sample_inlier_num == 0:
        #     continue
        # sample_model_residuals = sample_model_residuals_sum / sample_inlier_num
        #print(idx, best_inlier_num, sample_inlier_num, sample_model_residuals_sum*256*256)

        #对内点进行匹配分数加权
        # if weight_pairs is not None:
        #     weight_pairs_inlier = weight_pairs[sample_model_inliers]
        #     inliers_score = np.mean(weight_pairs_inlier)
        #     sample_inlier_num = sample_inlier_num*inliers_score
        
        best_idx = 0
        for i in range(len(H_batch)):
            if (
                (# more inliers
                sample_inlier_num[i] > best_inlier_num[b_idx]
                # same number of inliers but less "error" in terms of residuals
                or (sample_inlier_num[i] == best_inlier_num[b_idx]
                    and sample_model_residuals_sum[i] < best_inlier_residuals_sum[b_idx]))
            ):
                #print(idx, best_inlier_num, sample_inlier_num)
                best_model[b_idx] = H_batch[i]
                best_inlier_num[b_idx] = sample_inlier_num[i]
                best_inlier_residuals_sum[b_idx] = sample_model_residuals_sum[i]
                best_idx = i
                model_valid[b_idx] = 1
                best_inliers[b_idx] = sample_model_inliers[i]
                # dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                #                                             num_samples,
                #                                             min_samples,
                #                                             stop_probability)
                if (best_inlier_num[b_idx] >= stop_sample_num):  
                    #or best_inlier_residuals_sum <= stop_residuals_sum or num_trials >= dynamic_max_trials
                    break

    best_model = torch.from_numpy(best_model).to(bdata.device)
    best_inlier_num = torch.from_numpy(best_inlier_num).to(bdata.device)
    best_inlier_residuals_sum = torch.from_numpy(best_inlier_residuals_sum).to(bdata.device)
    best_inliers = torch.from_numpy(best_inliers).to(bdata.device)
    model_valid = torch.from_numpy(model_valid).to(bdata.device)

    return best_model, best_inliers, best_inlier_residuals_sum, model_valid

def slransac_random(data, model_class, min_samples, residual_threshold,
       is_data_valid=None, is_model_valid=None,
       max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
       stop_probability=1, random_state=None, initial_inliers=None, weight_pairs=None):

    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    random_state = check_random_state(random_state)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])

    if not (0 < min_samples < num_samples):
        raise ValueError("`min_samples` must be in range (0, <number-of-samples>)")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), num_samples))

    # for the first run use initial guess of inliers
    spl_idxs = (initial_inliers if initial_inliers is not None
                else random_state.choice(num_samples, min_samples, replace=False))
                
    def _check_sample(samples):
        a = samples[0]
        ra = np.roll(a, 1, axis = 0)
        dis_2_a = np.sum((a - ra)**2, axis = 1)
        b = samples[1]
        rb = np.roll(b, 1, axis = 0)
        dis_2_b = np.sum((b - rb)**2, axis = 1)
        scale = dis_2_b / dis_2_a
        #print(scale)
        if np.min(scale) < 0.6 or  np.max(scale) > 1.66:
            return False
        return True
        
    for num_trials in range(max_trials):
        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]
        # for next iteration choose random sample set and be sure that no samples repeat
        spl_idxs = random_state.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        _rationality = _check_sample(samples)
        #sample点合理性限制
        if not _rationality:
            continue
        success = sample_model.estimate(*samples)
        _rationality = _check_rationality(sample_model)

        #model合理性限制
        if not _rationality:
            continue

        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if is_model_valid is not None and not is_model_valid(sample_model, *samples):
            continue
        
        ##根据现有模型计算每个点的变换点，然后计算与匹配点的欧氏距离
        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals ** 2)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)
        #print(num_trials, best_inlier_num, sample_inlier_num, success, _rationality, sample_model.scale, sample_model.shear)
        #对内点进行匹配分数加权
        if weight_pairs is not None:
            weight_pairs_inlier = weight_pairs[sample_model_inliers]
            inliers_score = np.mean(weight_pairs_inlier)
            sample_inlier_num = sample_inlier_num*inliers_score

        if (
            # more inliers
            sample_inlier_num > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (sample_inlier_num == best_inlier_num
                and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     num_samples,
                                                     min_samples,
                                                     stop_probability)
            if (best_inlier_num >= stop_sample_num
                or best_inlier_residuals_sum <= stop_residuals_sum
                or num_trials >= dynamic_max_trials):
                break

    # estimate final model using all inliers
    if best_inliers is not None and any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)
    else:
        best_model = None
        best_inliers = None
        #warn("No inliers found. Model not fitted")

    return best_model, best_inliers
