import controlpy
import control
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


def main():
    r = 0.05
    L = 0.235
    w_ref_l = 3
    w_ref_r = w_ref_l
    v_bar = r/2*(w_ref_l + w_ref_r)

    A = np.array([
        [v_bar, -r/L, r/L],
        [0, -0.1, 0],
        [0, 0, -0.1]
    ])

    print(np.linalg.inv(A))
    B = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    Q = control.ctrb(A, B)      # reachability matrix

    C = np.eye(3)
    D = np.zeros((3, 2))

    # F, G, H, M, dt = sig.cont2discrete((A, B, C, D), .01)
    # print(F)

    Q = np.array([
        [100, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    r_lqr = 0.01
    R = r_lqr*np.eye(2)

    K, S, E = controlpy.synthesis.controller_lqr(A, B, Q, R)
    # K, S, E = controlpy.synthesis.controller_lqr(F, G, Q, R)
    print(K)

    state_space = control.StateSpace(A - B @ K, B, C, D)
    # state_space = control.StateSpace(F - G @ K, G, H, M)

    Z = C @ np.linalg.inv(-A + B @ K) @ B       # Intermediate term
    # Z = H @ np.linalg.inv(-F + G @ K) @ G  # Intermediate term
    F = np.linalg.inv(Z.T @ Z) @ Z.T

    ref = np.array([
        [0],
        [3],
        [3]
    ])
    Fr = F @ ref
    t = np.arange(0, 6, .01)
    Fr_t = Fr * np.ones((2, len(t)))

    T, y_out, x_out = control.forced_response(state_space, t, Fr_t, X0=[.2, 3, 3])   # .2, 3, 3
    fig1, ax1 = plt.subplots()
    ax1.plot(T, y_out[0], T, y_out[1], T, y_out[2])

    T, y_out, x_out = control.forced_response(state_space, t, Fr_t, X0=[0, 0, 0])   # .2, 3, 3
    fig2, ax2 = plt.subplots()
    ax2.plot(T, y_out[0], T, y_out[1], T, y_out[2])

    T, y_out, x_out = control.forced_response(state_space, t, Fr_t, X0=[-.2, 3, 3])   # .2, 3, 3
    fig3, ax3 = plt.subplots()
    ax3.plot(T, y_out[0], T, y_out[1], T, y_out[2])

    plt.show()


if __name__ == "__main__":
    main()
