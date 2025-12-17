import numpy as np

c_1 = 0.001
#c_2 = 0.9
c_2 = 0.1

# DEFAULT_JOINT_LIMITS = [
#     (-2*np.pi, 2*np.pi),                        # Joint 1: +/-360°
#     (np.deg2rad(-150), np.deg2rad(150)),        # Joint 2: +/-150°
#     (np.deg2rad(-3.5), np.deg2rad(300)),        # Joint 3: -3.5° to +300°
#     (-2*np.pi, 2*np.pi),                        # Joint 4: +/-360°
#     (np.deg2rad(-124), np.deg2rad(124)),        # Joint 5: +/-124°
#     (-2*np.pi, 2*np.pi),                        # Joint 6: +/-360°
# ]

JOINT_LIMITS_DEG = [
    (360, 360),         # Joint 1
    (150, 150),         # Joint 2
    (-3.5, 300),        # Joint 3
    (360, 360),         # Joint 4
    (124, 124),         # Joint 5
    (360, 360),         # Joint 6
]
DEFAULT_JOINT_LIMITS = np.deg2rad(JOINT_LIMITS_DEG)

def solve_for_final_joints(end_effector_xyz, target_xyz, current_joint_state,
                          forward_kinematics_func, threshold=1e-6, max_iter=100,
                          joint_limits=None, num_restarts=3, max_reach=0.44):
    if joint_limits is None:
        joint_limits = DEFAULT_JOINT_LIMITS

    # cjheck if target is reachable and project onto workspace if needed
    target_distance = np.linalg.norm(target_xyz)
    original_target = target_xyz.copy()

    if target_distance > max_reach: # gotta make sure we are within reach, otherwise it will cause problems!
        safety_margin = 0.02  # 2cm safety margin
        target_xyz = target_xyz * (max_reach - safety_margin) / target_distance
        print(f"CRITICAL: Target distance ({target_distance:.3f}m) exceeds max reach ({max_reach}m)")

    def objective_function(joint_positions):
        ee_pos = forward_kinematics_func(joint_positions)

        diff = ee_pos - target_xyz
        return 0.5 * np.dot(diff, diff)

    def optimize_from_start(x0):
        xk = np.array(x0, dtype=float)
        hk = np.eye(len(x0))
        gk = None 

        def clamp_to_limits(joint_positions):
            clamped = joint_positions.copy()

            for i in range(min(len(clamped), len(joint_limits))):
                clamped[i] = np.clip(clamped[i], joint_limits[i][0], joint_limits[i][1])

            return clamped

        def gradient_fd(f, x, epsilon=1e-5):
            g = np.zeros_like(x)
            f_curr = f(x)

            for i in range(len(x)):
                x_p = x.copy()
                x_p[i] += epsilon
                g[i] = (f(x_p) - f_curr) / epsilon

            return g

        def line_search(phi, a_max, max_iter_ls):
            alpha = a_max
            phi_0 = phi(0)

            # leverage only armijo condition for simple backtracking
            # Armijo: phi(alpha) <= phi(0) + c_1 * alpha * D(0)
            for _ in range(max_iter_ls):
                phi_alpha = phi(alpha)

                if phi_alpha < phi_0:
                    return alpha
                alpha *= 0.5

                if alpha < 1e-6:
                    return 1e-6

            return alpha

        def bfgs_hessian(hk, sk, yk):
            yk_dot_sk = np.dot(yk, sk)

            # check for matrix singularity or negative curvature
            if abs(yk_dot_sk) < 1e-10:
                return hk
        
            pk = 1.0 / yk_dot_sk
            I = np.eye(len(sk))
            a = I - pk * np.outer(sk, yk)
            b = I - pk * np.outer(yk, sk)
            c = pk * np.outer(sk, sk)

            return a @ hk @ b + c

        # quasi-newton optimization loop
        prev_f = float('inf')
        stagnation_count = 0

        for iteration in range(max_iter):
            # compute or use cached gradient
            if gk is None:
                gk = gradient_fd(objective_function, xk)
            g_norm = np.linalg.norm(gk)

            f_current = objective_function(xk)

            # error in meters
            position_error = np.sqrt(2 * f_current)

            if position_error < threshold:
                break

            if abs(prev_f - f_current) < 1e-10:
                stagnation_count += 1
                if stagnation_count > 5:
                    # reset Hessian if stuck
                    hk = np.eye(len(xk))
                    stagnation_count = 0
            else:
                stagnation_count = 0
            prev_f = f_current

            pk = -hk @ gk

            def phi(alpha):
                candidate = clamp_to_limits(xk + alpha * pk)
                return objective_function(candidate)

            alpha = line_search(phi, a_max=1.0, max_iter_ls=20)

            xk_old = xk
            gk_old = gk
            xk_new = xk + alpha * pk

            xk = clamp_to_limits(xk_new)

            gk = gradient_fd(objective_function, xk)

            sk = xk - xk_old
            yk = gk - gk_old

            if np.linalg.norm(sk) > 1e-10: # update only if we moved
                hk = bfgs_hessian(hk, sk, yk)

        return xk, objective_function(xk)

    best_joints, best_error = optimize_from_start(current_joint_state)

    # if error is too large (> 1cm), try random restarts
    if best_error > 0.01**2 / 2 and num_restarts > 0:
        for _ in range(num_restarts):
            # gen random starting point near current position
            random_start = current_joint_state.copy()
            num_joints = min(len(random_start), len(joint_limits))
            for i in range(num_joints):
                # add random perturbation within +/-30 degrees
                perturbation = np.random.uniform(-np.pi/6, np.pi/6)
                random_start[i] = np.clip(
                    random_start[i] + perturbation,
                    joint_limits[i][0],
                    joint_limits[i][1]
                )

            candidate_joints, candidate_error = optimize_from_start(random_start)

            if candidate_error < best_error:
                best_joints = candidate_joints
                best_error = candidate_error

                if best_error < 0.001**2 / 2:  # termination threshold for good soln
                    break

    return best_joints