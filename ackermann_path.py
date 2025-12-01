import math

def interp_line(p0, p1, step):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return [p0]
    n = max(int(L/step), 1)
    return [(x0 + dx*k/n, y0 + dy*k/n) for k in range(1, n+1)]


def curvature_of_fillet(p_prev, p, p_next, R_min):
    """
    Compute the arc that satisfies Ackermann min turning radius.
    """
    v1 = (p[0]-p_prev[0], p[1]-p_prev[1])
    v2 = (p_next[0]-p[0], p_next[1]-p[1])

    L1 = math.hypot(*v1)
    L2 = math.hypot(*v2)
    if L1 < 1e-6 or L2 < 1e-6:
        return None

    u1 = (v1[0]/L1, v1[1]/L1)
    u2 = (v2[0]/L2, v2[1]/L2)

    dot = max(-1,min(1,u1[0]*u2[0] + u1[1]*u2[1]))
    phi = math.acos(dot)
    if phi < 1e-3:
        return None

    # max allowed radius from geometry
    tan_half = math.tan(phi/2)
    R_allowed = min(L1, L2) * 0.5 / tan_half
    R = min(R_min, R_allowed)
    d = R * tan_half

    # tangent points
    t1 = (p[0] - u1[0]*d, p[1] - u1[1]*d)
    t2 = (p[0] + u2[0]*d, p[1] + u2[1]*d)

    # turn direction
    cross = u1[0]*u2[1] - u1[1]*u2[0]
    left = cross > 0

    n1 = (-u1[1], u1[0])
    if left:
        c = (t1[0] + R*n1[0], t1[1] + R*n1[1])
    else:
        c = (t1[0] - R*n1[0], t1[1] - R*n1[1])

    th1 = math.atan2(t1[1]-c[1], t1[0]-c[0])
    th2 = math.atan2(t2[1]-c[1], t2[0]-c[0])

    # sweep direction
    if left:
        while th2 <= th1:
            th2 += 2*math.pi
    else:
        while th2 >= th1:
            th2 -= 2*math.pi

    return t1, t2, c, R, th1, th2
