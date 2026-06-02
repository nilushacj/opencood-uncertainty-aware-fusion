import math
import scipy.optimize
import numpy as np
import cv2

def GetDistOfPoints2D(p1, p2):
    '''
    求出两个点的距离
    '''
    return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))

def GetClosestID(p, p_set):
    '''
    在p_set点集中找到距p最近点的id
    '''
    id = 0
    min = float("inf")  # 初始化取最大值
    for i in range(p_set.shape[1]):
        dist = GetDistOfPoints2D(p, p_set[:, i])
        if dist < min:
            id = i
            min = dist
    return id

def GetDistOf2DPointsSet(set1, set2):
    '''
    求两个点集之间的平均点距
    '''
    loss = 0
    for i in range(set1.shape[1]):
        id = GetClosestID(set1[:, i], set2)
        dist = GetDistOfPoints2D(set1[:, i], set2[:, id])
        loss += dist
    return loss/set1.shape[1]


def ICP_2D(targetPoints, sourcePoints):
    '''
    二维 ICP 配准算法
    '''
    A = targetPoints  # A是目标点云（地图值）
    B = sourcePoints  # B是源点云（感知值）

    # 初始化迭代参数
    iteration_times = 0  # 迭代次数
    dist_now = 1  # A,B两点集之间初始化距离
    dist_improve = 1  # A,B两点集之间初始化距离提升
    dist_before = GetDistOf2DPointsSet(A, B)  # A,B两点集之间距离
    # print("迭代：第{}次，距离：{:.2f}，缩短：{:.2f}".format(iteration_times, dist_before, dist_before))
    # 初始化 R，T
    R = np.identity(2)
    T = np.zeros((2, 1))

    # 均一化点云
    A_mean = np.mean(A, axis=1).reshape((2, 1))
    A_ = A - A_mean
    # A_ = A
    # 迭代次数小于10并且距离大于0.01时，继续迭代
    while iteration_times < 2 and dist_now > 0.01:
        # 均一化点云
        B_mean = np.mean(B, axis=1).reshape((2, 1))
        B_ = B - B_mean
        # B_ = B
        # t_nume表示角度公式中的分子 t_deno表示分母
        t_nume, t_deno = 0, 0
        # 对源点云B中每个点进行循环
        for i in range(B_.shape[1]):
            j = GetClosestID(B_[:, i], A_)  # 找到距离最近的目标点云A点id
            # j = i
            t_nume += A_[1][j] * B_[0][i] - A_[0][j] * B_[1][i]  # 获得求和公式，分子的一项
            t_deno += A_[0][j] * B_[0][i] + A_[1][j] * B_[1][i]  # 获得求和公式，分母的一项
        # 计算旋转弧度θ
        theta = math.atan2(t_nume, t_deno)
        # print(theta)
        # 由旋转弧度θ得到旋转矩阵Ｒ
        delta_R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        # 计算平移矩阵Ｔ
        delta_T = A_mean - np.matmul(delta_R, B_mean)
        # 更新最新的点云
        B = np.matmul(delta_R, B) + delta_T
        # B = np.matmul(delta_R, B)
        # 更新旋转矩阵Ｒ和平移矩阵Ｔ
        R = np.matmul(delta_R, R)
        T = np.matmul(delta_R, T) + delta_T
        # 更新迭代
        iteration_times += 1  # 迭代次数+1
        dist_now = GetDistOf2DPointsSet(A, B)  # 更新两个点云之间的距离
        dist_improve = dist_before - dist_now  # 计算这一次迭代两个点云之间缩短的距离
        dist_before = dist_now  # 将"现在距离"赋值给"以前距离"

        # 打印迭代次数、损失距离、损失提升
        # print("迭代：第{}次，距离：{:.2f}，缩短：{:.2f}".format(iteration_times, dist_now, dist_improve))

    return R, T,theta,A_mean,B_mean

def MAD(dataset, n):
    median = np.median(dataset)  # 中位数
    deviations = abs(dataset - median)
    mad = np.median(deviations)
    remove_idx = np.where(abs(dataset - median) > n * mad)
    new_data = np.delete(dataset, remove_idx)
    return remove_idx,new_data


def distance(a,b):
    return math.sqrt(float(a[0]-b[0])**2+float(a[1]-b[1])**2)

def get_t_matrix(mask0, mask1):
    mu_affine, _, _, _ = get_transform_distribution(mask0, mask1)
    return mu_affine

# def get_transform_distribution_level_1(mask0, mask1):
#     """
#     Estimate a rigid 2D transform distribution aligning mask1 -> mask0.

#     Returns:
#         mu_affine: np.float32 of shape (2, 3)
#             Mean affine transform [[cos(theta), -sin(theta), tx],
#                                    [sin(theta),  cos(theta), ty]]

#         mu_params: np.float32 of shape (3,)
#             [theta, tx, ty]

#         Sigma_params: np.float32 of shape (3, 3)
#             Covariance of [theta, tx, ty]

#         stats: dict
#             Diagnostic information useful for debugging/logging.
#     """
#     mask0 = mask0.astype(np.uint8)
#     mask1 = mask1.astype(np.uint8)

#     identity_affine = np.float32([[1, 0, 0],
#                                   [0, 1, 0]])
#     identity_params = np.float32([0.0, 0.0, 0.0])

#     # A conservative default covariance for failure cases.
#     # You can tune these numbers later.
#     default_sigma_theta = 0.10   # radians
#     default_sigma_tx = 5.0       # pixels
#     default_sigma_ty = 5.0       # pixels
#     default_cov = np.float32(np.diag([
#         default_sigma_theta ** 2,
#         default_sigma_tx ** 2,
#         default_sigma_ty ** 2
#     ]))

#     # ---------------------------------
#     # 1) Connected components + centroids
#     # ---------------------------------
#     num_labels0, labels0, stats0, centroids0 = cv2.connectedComponentsWithStats(
#         mask0, connectivity=4, ltype=None
#     )
#     num_labels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(
#         mask1, connectivity=4, ltype=None
#     )

#     centroids0 = centroids0[1:]   # remove background centroid
#     centroids1 = centroids1[1:]   # remove background centroid

#     stats = {
#         'num_centroids0_raw': int(len(centroids0)),
#         'num_centroids1_raw': int(len(centroids1)),
#         'num_centroids0_filtered': 0,
#         'num_centroids1_filtered': 0,
#         'num_matches_before_mad': 0,
#         'num_matches_after_mad': 0,
#         'mean_match_cost_before_mad': None,
#         'mean_match_cost_after_mad': None,
#         'icp_residual_mean': None,
#         'icp_residual_std': None,
#         'status': 'init'
#     }

#     # ---------------------------------
#     # 2) Proximity filtering
#     # ---------------------------------
#     idx0_list = []
#     idx1_list = []
#     Dis_thre = 10

#     for i in range(len(centroids0)):
#         for j in range(len(centroids1)):
#             if distance(centroids0[i], centroids1[j]) < Dis_thre:
#                 idx0_list.append(i)
#                 break

#     for j in range(len(centroids1)):
#         for i in range(len(centroids0)):
#             if distance(centroids1[j], centroids0[i]) < Dis_thre:
#                 idx1_list.append(j)
#                 break

#     centroids0 = centroids0[idx0_list]
#     centroids1 = centroids1[idx1_list]

#     stats['num_centroids0_filtered'] = int(len(centroids0))
#     stats['num_centroids1_filtered'] = int(len(centroids1))

#     if len(centroids0) < 1 or len(centroids1) < 1:
#         stats['status'] = 'no_valid_centroids'
#         return identity_affine, identity_params, default_cov, stats

#     # ---------------------------------
#     # 3) Hungarian assignment
#     # ---------------------------------
#     cost_matrix = np.zeros((len(centroids0), len(centroids1)), np.float32)
#     for i in range(len(centroids0)):
#         for j in range(len(centroids1)):
#             cost_matrix[i, j] = distance(centroids0[i], centroids1[j])

#     index0, index1 = scipy.optimize.linear_sum_assignment(cost_matrix)

#     match_costs = np.array([cost_matrix[i, j] for i, j in zip(index0, index1)], dtype=np.float32)

#     stats['num_matches_before_mad'] = int(len(match_costs))
#     stats['mean_match_cost_before_mad'] = float(np.mean(match_costs)) if len(match_costs) > 0 else None

#     if len(match_costs) < 1:
#         stats['status'] = 'no_matches'
#         return identity_affine, identity_params, default_cov, stats

#     # ---------------------------------
#     # 4) MAD outlier rejection
#     # ---------------------------------
#     remove_idx, filtered_costs = MAD(match_costs, 1)

#     index0 = np.delete(index0, remove_idx)
#     index1 = np.delete(index1, remove_idx)

#     centroids0_matched = centroids0[index0]
#     centroids1_matched = centroids1[index1]

#     stats['num_matches_after_mad'] = int(len(centroids0_matched))
#     stats['mean_match_cost_after_mad'] = (
#         float(np.mean(filtered_costs)) if len(filtered_costs) > 0 else None
#     )

#     if len(centroids0_matched) < 1 or len(centroids1_matched) < 1:
#         stats['status'] = 'no_matches_after_mad'
#         return identity_affine, identity_params, default_cov, stats

#     # ---------------------------------
#     # 5) Estimate mean rigid transform using the current ICP
#     # ---------------------------------
#     target_pts = centroids0_matched.T   # shape (2, N)
#     source_pts = centroids1_matched.T   # shape (2, N)

#     R, T, _, A_mean, B_mean = ICP_2D(target_pts, source_pts)
#     theta = math.atan2(R[1, 0], R[0, 0])
#     tx = float(T[0, 0])
#     ty = float(T[1, 0])

#     mu_affine = np.float32([
#         [R[0, 0], R[0, 1], tx],
#         [R[1, 0], R[1, 1], ty]
#     ])
#     mu_params = np.float32([theta, tx, ty])

#     # ---------------------------------
#     # 6) Residuals after alignment
#     # ---------------------------------
#     aligned_source = (R @ source_pts) + T    # shape (2, N)
#     residuals = target_pts - aligned_source  # shape (2, N)
#     residual_norms = np.linalg.norm(residuals, axis=0)  # shape (N,)

#     residual_mean = float(np.mean(residual_norms)) if residual_norms.size > 0 else 0.0
#     residual_std = float(np.std(residual_norms)) if residual_norms.size > 0 else 0.0

#     stats['icp_residual_mean'] = residual_mean
#     stats['icp_residual_std'] = residual_std

#     # ---------------------------------
#     # 7) Simple first-pass covariance estimate
#     # ---------------------------------
#     #
#     # This is intentionally simple for Step 1.
#     # Intuition:
#     # - more residual => more uncertainty
#     # - fewer matches => more uncertainty
#     # - wider spatial spread of points => less rotation uncertainty
#     #
#     n_matches = max(int(target_pts.shape[1]), 1)

#     centered_target = target_pts - np.mean(target_pts, axis=1, keepdims=True)
#     point_spread = float(np.mean(np.linalg.norm(centered_target, axis=0))) if n_matches > 0 else 1.0
#     point_spread = max(point_spread, 1.0)  # avoid division by zero / tiny spread

#     # A rough measurement noise scale from residuals + assignment cost
#     mean_cost_after = stats['mean_match_cost_after_mad']
#     if mean_cost_after is None:
#         mean_cost_after = residual_mean

#     noise_scale = max(1e-3, 0.5 * residual_mean + 0.5 * float(mean_cost_after))

#     # Translation uncertainty: larger noise, fewer matches => larger variance
#     sigma_tx = noise_scale / np.sqrt(n_matches)
#     sigma_ty = noise_scale / np.sqrt(n_matches)

#     # Rotation uncertainty should also depend on how spread out the points are
#     sigma_theta = noise_scale / (np.sqrt(n_matches) * point_spread)

#     # Put some lower bounds so covariance never collapses to exact zero
#     sigma_theta = max(sigma_theta, 1e-3)
#     sigma_tx = max(sigma_tx, 1e-2)
#     sigma_ty = max(sigma_ty, 1e-2)

#     Sigma_params = np.float32(np.diag([
#         sigma_theta ** 2,
#         sigma_tx ** 2,
#         sigma_ty ** 2
#     ]))

#     stats['status'] = 'ok'

#     return mu_affine, mu_params, Sigma_params, stats

def get_transform_distribution(mask0, mask1):
    """
    Estimate a rigid 2D transform distribution aligning mask1 -> mask0, uncertainty calculated using least-squares approximation

    Returns:
        mu_affine: np.float32 of shape (2, 3)
            Mean affine transform [[cos(theta), -sin(theta), tx],
                                   [sin(theta),  cos(theta), ty]]

        mu_params: np.float32 of shape (3,) i.e. the best transform
            [theta, tx, ty]

        Sigma_params: np.float32 of shape (3, 3) i.e. the uncertainty of the transform
            Covariance of [theta, tx, ty]
             = [Var(θ)    Cov(tx,θ)  Cov(ty,θ)
                Cov(θ,tx) Var(tx)    Cov(ty,tx)
                Cov(θ,ty) Cov(tx,ty) Var(ty)]


        stats: dict
            Diagnostic information useful for debugging/logging.
    """
    mask0 = mask0.astype(np.uint8)
    mask1 = mask1.astype(np.uint8)

    identity_affine = np.float32([[1, 0, 0],
                                  [0, 1, 0]])
    identity_params = np.float32([0.0, 0.0, 0.0])

    # A conservative default covariance for failure cases.
    # You can tune these numbers later.
    default_sigma_theta = 0.10   # radians
    default_sigma_tx = 5.0       # pixels
    default_sigma_ty = 5.0       # pixels
    default_cov = np.float32(np.diag([
        default_sigma_theta ** 2,
        default_sigma_tx ** 2,
        default_sigma_ty ** 2
    ]))

    # ---------------------------------
    # 1) Connected components + centroids
    # ---------------------------------
    num_labels0, labels0, stats0, centroids0 = cv2.connectedComponentsWithStats(
        mask0, connectivity=4, ltype=None
    )
    num_labels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(
        mask1, connectivity=4, ltype=None
    )

    centroids0 = centroids0[1:]   # remove background centroid
    centroids1 = centroids1[1:]   # remove background centroid

    stats = {
        'num_centroids0_raw': int(len(centroids0)),
        'num_centroids1_raw': int(len(centroids1)),
        'num_centroids0_filtered': 0,
        'num_centroids1_filtered': 0,
        'num_matches_before_mad': 0,
        'num_matches_after_mad': 0,
        'mean_match_cost_before_mad': None,
        'mean_match_cost_after_mad': None,
        'icp_residual_mean': None,
        'icp_residual_std': None,
        'status': 'init'
    }

    # ---------------------------------
    # 2) Proximity filtering
    # ---------------------------------
    idx0_list = []
    idx1_list = []
    Dis_thre = 10

    for i in range(len(centroids0)):
        for j in range(len(centroids1)):
            if distance(centroids0[i], centroids1[j]) < Dis_thre:
                idx0_list.append(i)
                break

    for j in range(len(centroids1)):
        for i in range(len(centroids0)):
            if distance(centroids1[j], centroids0[i]) < Dis_thre:
                idx1_list.append(j)
                break

    centroids0 = centroids0[idx0_list]
    centroids1 = centroids1[idx1_list]

    stats['num_centroids0_filtered'] = int(len(centroids0))
    stats['num_centroids1_filtered'] = int(len(centroids1))

    if len(centroids0) < 1 or len(centroids1) < 1:
        stats['status'] = 'no_valid_centroids'
        return identity_affine, identity_params, default_cov, stats

    # ---------------------------------
    # 3) Hungarian assignment
    # ---------------------------------
    cost_matrix = np.zeros((len(centroids0), len(centroids1)), np.float32)
    for i in range(len(centroids0)):
        for j in range(len(centroids1)):
            cost_matrix[i, j] = distance(centroids0[i], centroids1[j])

    index0, index1 = scipy.optimize.linear_sum_assignment(cost_matrix)

    match_costs = np.array([cost_matrix[i, j] for i, j in zip(index0, index1)], dtype=np.float32)

    stats['num_matches_before_mad'] = int(len(match_costs))
    stats['mean_match_cost_before_mad'] = float(np.mean(match_costs)) if len(match_costs) > 0 else None

    if len(match_costs) < 1:
        stats['status'] = 'no_matches'
        return identity_affine, identity_params, default_cov, stats

    # ---------------------------------
    # 4) MAD outlier rejection
    # ---------------------------------
    remove_idx, filtered_costs = MAD(match_costs, 1)

    index0 = np.delete(index0, remove_idx)
    index1 = np.delete(index1, remove_idx)

    centroids0_matched = centroids0[index0]
    centroids1_matched = centroids1[index1]

    stats['num_matches_after_mad'] = int(len(centroids0_matched))
    stats['mean_match_cost_after_mad'] = (
        float(np.mean(filtered_costs)) if len(filtered_costs) > 0 else None
    )

    if len(centroids0_matched) < 1 or len(centroids1_matched) < 1:
        stats['status'] = 'no_matches_after_mad'
        return identity_affine, identity_params, default_cov, stats

    # ---------------------------------
    # 5) Estimate mean rigid transform using the current ICP
    # ---------------------------------
    target_pts = centroids0_matched.T   # shape (2, N)
    source_pts = centroids1_matched.T   # shape (2, N)

    R, T, _, A_mean, B_mean = ICP_2D(target_pts, source_pts)
    theta = math.atan2(R[1, 0], R[0, 0])
    tx = float(T[0, 0])
    ty = float(T[1, 0])

    mu_affine = np.float32([
        [R[0, 0], R[0, 1], tx],
        [R[1, 0], R[1, 1], ty]
    ])
    mu_params = np.float32([theta, tx, ty])

    # ---------------------------------
    # 6) Residuals after alignment, residual = actual target point - predicted target point
    # ---------------------------------
    aligned_source = (R @ source_pts) + T    # shape (2, N)
    residuals = target_pts - aligned_source  # shape (2, N)
    residual_norms = np.linalg.norm(residuals, axis=0)  # shape (N,)

    residual_mean = float(np.mean(residual_norms)) if residual_norms.size > 0 else 0.0
    residual_std = float(np.std(residual_norms)) if residual_norms.size > 0 else 0.0

    stats['icp_residual_mean'] = residual_mean
    stats['icp_residual_std'] = residual_std

    # ---------------------------------
    # 7) Level-2 covariance from least-squares geometry
    # ---------------------------------
    # We estimate covariance over p = [theta, tx, ty]
    # using Sigma ~= sigma^2 * (J^T J)^(-1),
    # where J is the Jacobian of residuals w.r.t. parameters.
    #
    # Residual for match i:
    #   r_i = a_i - (R(theta) b_i + t)
    #
    # with:
    #   a_i = target point
    #   b_i = source point
    #   t   = [tx, ty]^T
    #
    # For one source point b_i = [x_i, y_i]^T, predicted target is:
    #   f_i(theta, tx, ty) = R(theta) b_i + [tx, ty]^T
    #
    # Residual Jacobian J_i = d r_i / d p is:
    #   [ sin(theta)*x_i + cos(theta)*y_i,   -1,   0 ]
    #   [ -cos(theta)*x_i + sin(theta)*y_i,   0,  -1 ]
    #
    # Stacking all J_i gives a (2N x 3) Jacobian.
    n_matches = int(target_pts.shape[1])

    # Fallback if too few matches for stable covariance estimation
    # Two points give 4 equations for 3 unknowns, but that is fragile
    if n_matches < 3:
        stats['status'] = 'too_few_matches_for_covariance'
        stats['covariance_method'] = 'default_fallback'
        stats['cov_condition_number'] = None
        return mu_affine, mu_params, default_cov, stats

    c = math.cos(theta)
    s = math.sin(theta)
    
    # Build stacked Jacobian J of shape (2N, 3)
    J = np.zeros((2 * n_matches, 3), dtype=np.float64)

    for i in range(n_matches):
        x_i = float(source_pts[0, i])
        y_i = float(source_pts[1, i])

        # d residual / d theta, d residual / d tx, d residual / d ty
        J[2 * i + 0, 0] = s * x_i + c * y_i
        J[2 * i + 0, 1] = -1.0
        J[2 * i + 0, 2] = 0.0

        J[2 * i + 1, 0] = -c * x_i + s * y_i
        J[2 * i + 1, 1] = 0.0
        J[2 * i + 1, 2] = -1.0

    # Stack residuals into shape (2N, 1)
    r = residuals.T.reshape(-1, 1).astype(np.float64)

    # Residual variance estimate:
    # 2N residual equations, 3 fitted parameters
    dof = 2 * n_matches - 3

    if dof <= 0:
        stats['status'] = 'insufficient_dof_for_covariance'
        stats['covariance_method'] = 'default_fallback'
        stats['cov_condition_number'] = None
        return mu_affine, mu_params, default_cov, stats

    sigma2_hat = float((r.T @ r) / dof)

    # Guard against pathological zero/negative values
    sigma2_hat = max(sigma2_hat, 1e-8)

    JTJ = J.T @ J
    cond_number = float(np.linalg.cond(JTJ))
    stats['cov_condition_number'] = cond_number

    # If geometry is nearly singular, fall back to default covariance
    if not np.isfinite(cond_number) or cond_number > 1e12:
        stats['status'] = 'ill_conditioned_covariance'
        stats['covariance_method'] = 'default_fallback'
        return mu_affine, mu_params, default_cov, stats

    try:
        JTJ_inv = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        stats['status'] = 'singular_covariance'
        stats['covariance_method'] = 'default_fallback'
        return mu_affine, mu_params, default_cov, stats

    Sigma_params = sigma2_hat * JTJ_inv

    # Numerical cleanup: enforce symmetry
    Sigma_params = 0.5 * (Sigma_params + Sigma_params.T)

    # If tiny negative eigenvalues appear from numerical noise, clip them
    eigvals, eigvecs = np.linalg.eigh(Sigma_params)
    eigvals = np.clip(eigvals, 1e-10, None)
    Sigma_params = eigvecs @ np.diag(eigvals) @ eigvecs.T

    Sigma_params = Sigma_params.astype(np.float32)

    stats['covariance_method'] = 'least_squares'
    stats['sigma2_hat'] = float(sigma2_hat)
    stats['cov_theta_var'] = float(Sigma_params[0, 0])
    stats['cov_tx_var'] = float(Sigma_params[1, 1])
    stats['cov_ty_var'] = float(Sigma_params[2, 2])

    stats['status'] = 'ok' # NOTE: small residuals / small covariance values = good alignment
    return mu_affine, mu_params, Sigma_params, stats


def compute_optimal_transport(M, r, c, lam, epsilon=1e-6):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm
    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter
    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))#行归r化，注意python中*号含义
        # print(c.shape, P.sum(0).shape)
        P *= (c / P.sum(0)).reshape((1, -1))#列归c化
    return P, np.sum(P * M)



def get_t_matrix_2(mask0,mask1):
 
    
    # 连通域数量
    H,W = mask0.shape
    # t_matrix = np.float32([[2,0,0],
    #                       [0,2,0]])
    # mask0 = cv2.warpAffine(mask0,t_matrix,(W,H))
    # mask1 = cv2.warpAffine(mask1,t_matrix,(W,H))
    # H,W = mask0.shape
    mask0 = mask0.astype(np.uint8)
    mask1 = mask1.astype(np.uint8)
    # 连通域数量 num_labels0
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积(外接矩阵) stats0
    # 连通域的中心点 centroids0
    num_labels0, labels0, stats0, centroids0 = cv2.connectedComponentsWithStats(mask0, connectivity=4, ltype=None)
    num_labels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(mask1, connectivity=4, ltype=None)

    centroids0 = centroids0[1:] # 第一行是nan 代表背景的
    centroids1 = centroids1[1:]
    idx0_list = []
    idx1_list = []
    
    #周围必须有object,否则视为噪声点
    Dis_thre = 10

    for i in range(len(centroids0)):
        for j in range(len(centroids1)):
            if distance(centroids0[i],centroids1[j])<Dis_thre:
                idx0_list.append(i)
                break

    for j in range(len(centroids1)):
        for i in range(len(centroids0)):
            if distance(centroids1[j],centroids0[i])<Dis_thre:
                idx1_list.append(j)
                break
    centroids0 = centroids0[idx0_list]
    centroids1 = centroids1[idx1_list]

    if len(centroids0)<1 or len(centroids1)<1:
        return  np.float32([[1,0,0],
                       [0,1,0]]
                      )
    
    # cost ~C
    cost_matrix = np.ones((len(centroids0)+1, len(centroids1)+1), np.float32)
    for i in range(len(centroids0)):
        for j in range(len(centroids1)):
            # dis = distance(centroids0[i], centroids1[j])
            cost_matrix[i][j] = distance(centroids0[i],centroids1[j])
    # print(cost_matrix)
    P, _ = compute_optimal_transport(cost_matrix, np.ones(len(centroids0)+1), np.ones(len(centroids1)+1),1)
    # index0,index1 = scipy.optimize.linear_sum_assignment(cost_matrix)
    # print(index0, index1)
    P = P[:len(centroids0),:len(centroids1)]
    index0 = np.linspace(0,len(centroids0)-1,len(centroids0)).astype('int')
    index1 = np.argmax(P, axis=1)
    # print(index0, index1)
    # exit()
    cost = []
    for i in range(len(index0)):
        # print(index0[i],index1[i])
        cost.append(cost_matrix[index0[i]][index1[i]])
    cost = np.array(cost)
    # print(cost)
    #-----------------------------去除异常点-----------------------
    remove_idx,cost = MAD(cost,1)
    # print(cost)
    
    # if np.mean(cost)<10:
    #     # print(np.mean(cost))
    #     return np.float32([[1,0,0],
    #                       [0,1,0]])
    index0 = np.delete(index0, remove_idx)
    index1 = np.delete(index1, remove_idx)
    centroids0 = centroids0[index0]
    centroids1 = centroids1[index1]
    # print(centroids0)
    # print("----------------------------------------")
    # print(centroids1)
    # exit()

    centroids1 = centroids1.T
    centroids0 = centroids0.T
    # print(centroids1.shape)
    R,T,thera,A_mean,B_mean = ICP_2D(centroids0,centroids1)
    # exit()
    # print(T)
    # print(thera)
    # T[1] = 0.0
    # exit()
    # print(R,T)
    # thera = math.acos(R[0][0])
    # print(thera)
    # R = cv2.getRotationMatrix2D((0,0),thera,1)
    # R = cv2.getRotationMatrix2D((int(A_mean[0]),int(A_mean[1])),thera,1)
    # R = cv2.getRotationMatrix2D((int(B_mean[0]),int(B_mean[1])),thera,1)
    # return R
    # print(a)
    # exit()
    # R[...,0,1] = R[...,0,1] * H / W
    # R[...,1,0] = R[...,1,0] * W / H
    # T[0] = T[0] / (W)
    # T[1] = T[1] / (H) 
    return np.float32([[R[0][0],R[0][1],T[0]],
                       [R[1][0],R[1][1],T[1]]])