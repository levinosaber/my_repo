import numpy as np
import cv2
import sys
from numba import njit


@njit
def Census(img_l, img_r, maxdis, box_h=3, box_w=4):
    """
    calculate Census cost
    box_h and box_w is the half size of height and width of box size
    """
    H = img_l.shape[0]
    W = img_l.shape[1]
    census_cost_r = np.zeros(shape=(H, W, maxdis), dtype=np.float32)
    census_cost_l = np.zeros(shape=(H, W, maxdis), dtype=np.float32)
    census_cost_l[:, 0:box_w, :] = maxdis
    census_cost_l[:, (W - box_w) : W, :] = maxdis
    census_cost_l[0:box_h, :, :] = maxdis
    census_cost_l[(H - box_h) : H, :, :] = maxdis
    census_cost_r[:, :, :] = maxdis
    count = 0.0
    for h in range(box_h, H - box_h):
        for w in range(box_w, W - box_w):
            for d in range(0, maxdis):
                if (w - d) < box_w:
                    census_cost_l[h, w, d:] = maxdis
                    break
                else:
                    for x in range(h - box_h, h + box_h + 1):
                        for y in range(w - box_w, w + box_w + 1):
                            if (img_l[x, y] < img_l[h, w]) != (
                                img_r[x, y - d] < img_r[h, w - d]
                            ):
                                count += 1.0
                    census_cost_l[h, w, d] = count
                    census_cost_r[h, w - d, d] = count
                    count = 0
    return census_cost_l, census_cost_r


@njit
def Computer_AD(left_color_img, right_color_img, maxdis):
    """
    calculate AD index
    left and right image both must be image with color(RGB three channel)
    """
    left_color_img = left_color_img.astype(np.float32)
    right_color_img = right_color_img.astype(np.float32)
    left_AD_cost = np.zeros(
        shape=(left_color_img.shape[0], left_color_img.shape[1], maxdis),
        dtype=np.float32,
    )
    right_AD_cost = np.zeros(
        shape=(left_color_img.shape[0], left_color_img.shape[1], maxdis),
        dtype=np.float32,
    )
    left_AD_cost[:, :, :] = 255.0
    right_AD_cost[:, :, :] = 255.0
    for h in range(left_color_img.shape[0]):
        for w in range(left_color_img.shape[1]):
            for d in range(maxdis):
                if w - d >= 0:
                    left_AD_cost[h, w, d] = (
                        np.sum(np.abs(left_color_img[h, w] - right_color_img[h, w - d]))
                        / 3.0
                    )
                    right_AD_cost[h, w - d, d] = left_AD_cost[h, w, d]
    return left_AD_cost, right_AD_cost


@njit
def exp_norm(cost, lamb):
    """
    using a exponential method to do normalization
    """
    return 1 - np.exp(-cost / lamb)


@njit
def AD_Census_cost_comb(Census_cost, AD_cost, lambda1, lambda2, H, W, maxdis):
    """
    normalize the results of Census and AD
    the result should be in the range of [0,2]
    """
    total_cost = np.zeros(shape=(H, W, maxdis), dtype=np.float32)
    total_cost[:, :, :] = 2.0
    Census_cost = Census_cost * (-1 / lambda1)
    AD_cost = AD_cost * (-1 / lambda2)
    total_cost = total_cost - np.exp(Census_cost) - np.exp(AD_cost)
    return total_cost


@njit
def arms_region_check(color_diff, spatial_dis, neighbor_dis, tao1, tao2, L1, L2):
    """
    L2 must be smaller than L1
    tao2 must be smaller than tao1
    """
    if L2 < spatial_dis < L1 and neighbor_dis < tao1 and color_diff < tao2:
        return False
    elif spatial_dis <= L2 and neighbor_dis < tao1 and color_diff < tao1:
        return False
    else:
        return True


@njit
def get_arm_map(img, tao1, tao2, L1, L2):
    """
    L2 < L1
    tao2 < tao1

    return a numpy.ndarray which contains the infos of support region of each pixel,
        it has a shape of (height,width,4). the last dimension whose size is 4 is for stocking the
        numebr of pixels considered in four directions of the center pixels
    """
    H = img.shape[0]
    W = img.shape[1]
    img = img.astype(np.int32)
    arm_map = np.zeros(shape=(H, W, 4), dtype=np.uint8)
    for h in range(0, H):
        for w in range(0, W):
            for r in range(w + 1, W):
                # create right arm of center pixel
                color_diff = max(np.abs(img[h, w] - img[h, r]))
                spatial_dis = r - w
                neighbor_dis = max(np.abs(img[h, r] - img[h, r - 1]))
                if (
                    arms_region_check(
                        color_diff, spatial_dis, neighbor_dis, tao1, tao2, L1, L2
                    )
                    or r == W - 1
                ):
                    arm_map[h, w, 1] = max(spatial_dis, 2)
                    break
            for l in range(w - 1, -1, -1):
                # create left arm of center pixel
                color_diff = max(np.abs(img[h, w] - img[h, l]))
                spatial_dis = w - l
                neighbor_dis = max(np.abs(img[h, l] - img[h, l + 1]))
                if (
                    arms_region_check(
                        color_diff, spatial_dis, neighbor_dis, tao1, tao2, L1, L2
                    )
                    or l == 0
                ):
                    arm_map[h, w, 0] = max(spatial_dis, 1)
                    break
            for up in range(h - 1, -1, -1):
                # create up arm of center pixel
                color_diff = max(np.abs(img[up, w] - img[h, w]))
                spatial_dis = h - up
                neighbor_dis = max(np.abs(img[up, w] - img[up + 1, w]))
                if (
                    arms_region_check(
                        color_diff, spatial_dis, neighbor_dis, tao1, tao2, L1, L2
                    )
                    or up == 0
                ):
                    arm_map[h, w, 2] = max(spatial_dis, 1)
                    break
            for b in range(h + 1, H):
                # create bottom arm of center pixel
                color_diff = max(np.abs(img[h, w] - img[b, w]))
                spatial_dis = b - h
                neighbor_dis = max(np.abs(img[b, w] - img[b - 1, w]))
                if (
                    arms_region_check(
                        color_diff, spatial_dis, neighbor_dis, tao1, tao2, L1, L2
                    )
                    or b == H - 1
                ):
                    arm_map[h, w, 3] = max(spatial_dis, 2)  # ?? 2 可以变吗 ??
                    break
    return arm_map


@njit
def cost_agg(arm_map, cost):  # aggodd
    """
    return:
        agg: of shape (height, width, max_dis)

    Each loop iterates through one central pixel, adding the cost values of the four arms to the cost value of the central pixel
    """
    agg_ver = np.zeros(
        shape=(cost.shape[0], cost.shape[1], cost.shape[2]), dtype=np.float32
    )
    agg_hor = np.zeros(
        shape=(cost.shape[0], cost.shape[1], cost.shape[2]), dtype=np.float32
    )
    # Horizontal direction
    for h in range(cost.shape[0]):
        for w in range(cost.shape[1]):
            # for d in range(CostVolume.shape[2]):
            for l in range(w - arm_map[h, w, 0], w + arm_map[h, w, 1]):
                agg_hor[h, w, :] = agg_hor[h, w, :] + cost[h, l, :]
    # Vertical direction
    for h in range(cost.shape[0]):
        for w in range(cost.shape[1]):
            # for d in range(CostVolume.shape[2]):
            for t in range(h - arm_map[h, w, 2], h + arm_map[h, w, 3]):
                agg_ver[h, w, :] = agg_ver[h, w, :] + agg_hor[t, w, :]
    return agg_ver


def find_min_cost_disp(cost, p_h, p_w, disp):
    min_cost_disp = np.argmin(cost[p_h, p_w, disp], axis=2)
    return min_cost_disp


@njit
def update_penalty(pi1, pi2, taoso, D1, D2, maxdis):
    if D1 < taoso and D2 < taoso:
        p1 = pi1
        p2 = pi2
    elif D1 > taoso and D2 > taoso:
        p1 = pi1 / 10
        p2 = pi2 / 10
    else:
        p1 = pi1 / 4
        p2 = pi2 / 4
    return p1, p2


@njit
def scanline_optimization_horizon(cost, left_color_img, right_color_img, maxdis, paras):
    """
    Take the picture on the left as a reference
    paras:( 
        "taoso"  0
        "pi1"    1
        "pi2"    2
        "lamb"   3
    )
    """
    H = cost.shape[0]
    W = cost.shape[1]
    imgL = left_color_img.astype(np.float32)
    imgR = right_color_img.astype(np.float32)
    aggtwo = np.zeros(shape=(H, W, maxdis), dtype=np.float32)
    aggfour = np.zeros(shape=(H, W, maxdis), dtype=np.float32)
    aggtwo[:, 0, :] = cost[:, 0, :]  # initial values
    aggfour[:, W - 1, :] = cost[:, W - 1, :]  # initial values
    lamb = paras[3]
    for w in range(1, W):
        for h in range(0, H):
            for d in range(maxdis):
                # aggreation from left to right
                if (w - d - 1) >= 0 and d - 1 >= 0 and d + 1 < maxdis:
                    c1_p_d = exp_norm(cost[h, w, d], lamb)
                    term3 = np.argmin(exp_norm(aggtwo[h, w - 1], lamb))
                    cr_pr_d = exp_norm(aggtwo[h, w - 1, d], lamb)
                    D1 = np.max(np.abs(imgL[h, w] - imgL[h, w - 1]))
                    D2 = np.max(np.abs(imgR[h, w - d] - imgR[h, w - d - 1]))
                    P1, P2 = update_penalty(
                        paras[1], paras[2], paras[0], D1, D2, maxdis
                    )
                    cr_pr_d_neg1 = exp_norm(aggtwo[h, w - 1, d - 1], lamb) + P1
                    cr_pr_d_pos1 = exp_norm(aggtwo[h, w - 1, d + 1], lamb) + P1
                    min_disp_cost_pr = np.argmin(exp_norm(aggtwo[h, w - 1], lamb)) + P2
                    cr_p_d = (
                        c1_p_d
                        + min(cr_pr_d, cr_pr_d_neg1, cr_pr_d_pos1, min_disp_cost_pr)
                        - term3
                    )
                    aggtwo[h, w, d] = cr_p_d

    for w in range(W - 2, -1, -1):
        for h in range(0, H):
            for d in range(maxdis):
                # aggreation from right to left
                if (w - d) >= 0 and d - 1 >= 0 and d + 1 < maxdis:
                    c1_p_d = exp_norm(cost[h, w, d], lamb)
                    term3 = np.argmin(exp_norm(aggfour[h, w + 1], lamb))
                    cr_pr_d = exp_norm(aggfour[h, w + 1, d], lamb)
                    D1 = np.max(np.abs(imgL[h, w] - imgL[h, w + 1]))
                    D2 = np.max(np.abs(imgR[h, w - d] - imgR[h, w - d + 1]))
                    P1, P2 = update_penalty(
                        paras[1], paras[2], paras[0], D1, D2, maxdis
                    )
                    cr_pr_d_neg1 = exp_norm(aggfour[h, w + 1, d - 1], lamb) + P1
                    cr_pr_d_pos1 = exp_norm(aggfour[h, w + 1, d + 1], lamb) + P1
                    min_disp_cost_pr = np.argmin(exp_norm(aggfour[h, w - 1], lamb)) + P2
                    cr_p_d = (
                        c1_p_d
                        + min(cr_pr_d, cr_pr_d_neg1, cr_pr_d_pos1, min_disp_cost_pr)
                        - term3
                    )
                    aggfour[h, w, d] = cr_p_d
    return aggtwo, aggfour


@njit
def scanline_optimization_vertical(cost_volume, color_left, color_right, maxdis, paras):
    """
    Take the picture on the left as a reference
    paras:( 
        "taoso"
        "pi1"
        "pi2"
        "lamb"
    )
    """
    H = cost_volume.shape[0]
    W = cost_volume.shape[1]
    imgL = color_left.astype(np.float32)
    imgR = color_right.astype(np.float32)
    aggfirst = np.zeros(shape=(H, W, maxdis), dtype=np.float32)
    aggthird = np.zeros(shape=(H, W, maxdis), dtype=np.float32)
    aggfirst[0, :, :] = cost_volume[0, :, :]  # initial values
    aggthird[H - 1 :, :, :] = cost_volume[H - 1, :, :]  # initial values
    lamb = paras[3]
    for h in range(1, H):
        for w in range(0, W):
            for d in range(1, maxdis):
                # aggreation from up to down
                if d + 1 < maxdis and w - d >= 0:
                    c1_p_d = exp_norm(cost_volume[h, w, d], lamb)
                    term3 = np.argmin(exp_norm(aggfirst[h - 1, w], lamb))
                    cr_pr_d = exp_norm(aggfirst[h - 1, w, d], lamb)
                    D1 = np.max(np.abs(imgL[h, w] - imgL[h - 1, w]))
                    D2 = np.max(np.abs(imgR[h, w - d] - imgR[h - 1, w - d]))
                    P1, P2 = update_penalty(
                        paras[1], paras[2], paras[3], D1, D2, maxdis
                    )
                    cr_pr_d_neg1 = exp_norm(aggfirst[h - 1, w, d - 1], lamb) + P1
                    cr_pr_d_pos1 = exp_norm(aggfirst[h - 1, w, d + 1], lamb) + P1
                    min_disp_cost_pr = (
                        np.argmin(exp_norm(aggfirst[h - 1, w], lamb)) + P2
                    )
                    cr_p_d = (
                        c1_p_d
                        + min(cr_pr_d, cr_pr_d_neg1, cr_pr_d_pos1, min_disp_cost_pr)
                        - term3
                    )
                    aggfirst[h, w, d] = cr_p_d
    for h in range(1, H):
        for w in range(0, W):
            for d in range(1, maxdis):
                # aggreation from down to up
                if d + 1 < maxdis and h + 1 < maxdis:
                    c1_p_d = exp_norm(cost_volume[h, w, d], lamb)
                    term3 = np.argmin(exp_norm(aggthird[h + 1, w], lamb))
                    cr_pr_d = exp_norm(aggfirst[h + 1, w, d], lamb)
                    D1 = np.max(np.abs(imgL[h, w] - imgL[h + 1, w]))
                    D2 = np.max(np.abs(imgR[h, w - d] - imgR[h + 1, w - d]))
                    P1, P2 = update_penalty(
                        paras[1], paras[2], paras[0], D1, D2, maxdis
                    )
                    cr_pr_d_neg1 = exp_norm(aggthird[h + 1, w, d - 1], lamb) + P1
                    cr_pr_d_pos1 = exp_norm(aggthird[h + 1, w, d + 1], lamb) + P1
                    min_disp_cost_pr = (
                        np.argmin(exp_norm(aggthird[h + 1, w], lamb)) + P2
                    )
                    cr_p_d = (
                        c1_p_d
                        + min(cr_pr_d, cr_pr_d_neg1, cr_pr_d_pos1, min_disp_cost_pr)
                        - term3
                    )
                    aggthird[h, w, d] = cr_p_d
    return aggfirst, aggthird


def normalize(volume, maxdisparity):
    return 255.0 * volume / maxdisparity


def select_disparity(aggregation_volume):
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map



class DataLoader:
    def __init__(self, l_img_path, r_img_path):
        """
        all images are already rectified
        """
        self.l_img_path = l_img_path
        self.r_img_path = r_img_path

    def returning_img_data(self):
        try:
            left_color_img = cv2.imread(self.l_img_path)
            right_color_img = cv2.imread(self.r_img_path)
            left_gray_img = cv2.cvtColor(left_color_img, cv2.COLOR_BGR2GRAY)
            right_gray_img = cv2.cvtColor(right_color_img, cv2.COLOR_BGR2GRAY)
        except:
            print("something wrong when read images")
        return left_color_img, right_color_img, left_gray_img, right_gray_img

    def returning_color_img_data(self):
        try:
            left_color_img = cv2.imread(self.l_img_path)
            right_color_img = cv2.imread(self.r_img_path)
        except:
            print("something wrong when read images")
        return left_color_img, right_color_img


class StereoMatcher:
    def __init__(self):
        """
        provide two images rectified to the matcher
        """
        pass

    def verify_data(self, print_image=False):
        try:
            print(f"left image:  shape:{self.img_l.shape}")
            print(f"right image shaoe:{self.img_r.shape}")
        except:
            print("something wrong with printing images infos")
        if print_image == "True":
            cv2.imshow("left image", self.img_l)
            cv2.waitKey(500)
            cv2.imshow("right image", self.img_r)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    def AD_Census_cost(self, a_data_loader, paras):
        """
        return a cost volume of AD-Census
        """
        (
            left_color_img,
            right_color_img,
            left_gray_img,
            right_gray_img,
        ) = a_data_loader.returning_img_data()
        H = left_gray_img.shape[0]
        W = left_gray_img.shape[1]
        maxdis = paras["maxdis"]
        left_census_cost, right_census_cost = Census(
            left_gray_img, right_gray_img, maxdis
        )
        left_AD_cost, right_AD_cost = Computer_AD(
            left_color_img, right_color_img, maxdis
        )
        left_AD_Census_cost = AD_Census_cost_comb(
            left_census_cost, left_AD_cost, 30.0, 10.0, H, W, maxdis
        )
        right_AD_Census_cost = AD_Census_cost_comb(
            right_census_cost, right_AD_cost, 30.0, 10.0, H, W, maxdis
        )
        return left_AD_Census_cost, right_AD_Census_cost

    def AD_Census_cost_with_simple_agg(self, a_data_loader, paras):
        """
        return the AD-Census cost with simple aggreation
        """
        tao1 = paras["tao1"]
        tao2 = paras["tao2"]
        L1 = paras["L1"]
        L2 = paras["L2"]
        target = paras["target"]
        left_AD_Census_cost, right_AD_Census_cost = self.AD_Census_cost(
            a_data_loader, paras
        )
        left_color_img, right_color_img = a_data_loader.returning_color_img_data()
        left_arm_map = get_arm_map(
            left_color_img, tao1, tao2, L1, L2
        )  # determine left image arm map
        right_arm_map = get_arm_map(
            right_color_img, tao1, tao2, L1, L2
        )  # determine right image arm map

        if target == "left":
            left_cost_aggreated = cost_agg(left_arm_map, left_AD_Census_cost)
            return left_cost_aggreated
        elif target == "right":
            right_cost_aggreated = cost_agg(right_arm_map, right_AD_Census_cost)
            return right_cost_aggreated
        else:
            left_cost_aggreated = cost_agg(left_arm_map, left_AD_Census_cost)
            right_cost_aggreated = cost_agg(right_arm_map, right_AD_Census_cost)
            return left_cost_aggreated, right_cost_aggreated

    def AD_Census_cost_with_scanline_opt(self, a_data_loader, paras):
        """
        return the AD-Census aggreated by scanline optimization
        """
        left_AD_Census_cost, right_AD_Census_cost = self.AD_Census_cost(
            a_data_loader, paras
        )
        left_color_img, right_color_img = a_data_loader.returning_color_img_data()
        # calculate AD Census cost with scanline optimization of left image
        agg_two, agg_four = scanline_optimization_horizon(
            left_AD_Census_cost,
            left_color_img,
            right_color_img,
            paras["maxdis"],
            [paras["taoso"], paras["pi1"], paras["pi2"], paras["lamb"]],
        )
        agg_one, agg_three = scanline_optimization_vertical(
            left_AD_Census_cost,
            left_color_img,
            right_color_img,
            paras["maxdis"],
            [paras["taoso"], paras["pi1"], paras["pi2"], paras["lamb"]],
        )
        left_cost_agg_scan_opt = (agg_one + agg_two + agg_three + agg_four) / 4
        return left_cost_agg_scan_opt

    def LRC_check(self, left_disp_map, right_disp_map, thres=230):
        """
        return a occlusion map of left image as reference
        """
        H = left_disp_map.shape[0]
        W = left_disp_map.shape[1]
        occ_map = np.zeros(shape=(H, W), dtype=np.uint8)
        for h in range(H):
            for w in range(W):
                left_point_disp = left_disp_map[h, w]
                right_point_disp = right_disp_map[h, w - left_point_disp]
                if np.abs(left_point_disp - right_point_disp) > thres:
                    # a occlusion point
                    occ_map[h, w] = 0
                else:
                    occ_map[h, w] = 255
        return occ_map

    def occlusion_filling_left(self, left_disp_map, right_disp_map, occ_map):
        """
        filling the occlusion points
        this function may change the value of parameters
        """
        H = left_disp_map.shape[0]
        W = right_disp_map.shape[1]
        for h in range(H):
            for w in range(W):
                if occ_map[h, w] == 0:
                    # a occlusion point
                    find_a_non_occ_left_point = False
                    find_a_non_occ_right_point = False
                    ww = w - 1
                    while ww >= 0:
                        if occ_map[h, ww] == 0:
                            ww -= 1
                            continue
                        else:
                            # first non occlusion point
                            find_a_non_occ_left_point = True
                            break
                    if find_a_non_occ_left_point == True:
                        left_strike = w - ww
                    else:
                        left_strike = np.Inf

                    ww = w + 1
                    while ww < W:
                        if occ_map[h, ww] == 0:
                            ww += 1
                            continue
                        else:
                            # first non occlusion point on right
                            find_a_non_occ_right_point = True
                            break
                    if find_a_non_occ_right_point == True:
                        right_strike = ww - w
                    else:
                        right_strike = np.Inf

                    if (
                        find_a_non_occ_left_point == False
                        and find_a_non_occ_right_point == False
                    ):
                        print(
                            "all points in the same line are occlusion points, impossbile!!"
                        )
                    else:
                        if left_strike < right_strike:
                            ww = w - left_strike
                            left_disp_map[h, w] = left_disp_map[h, ww]
                        else:
                            ww = right_strike + w
                            left_disp_map[h, w] = left_disp_map[h, ww]
        return

    def cost_map_to_color_img(self, cost, maxdis):
        return normalize(np.uint8(np.argmin(cost, axis=2), maxdis))

    def disparity_to_color_img(self, disp_map, maxdis):
        return normalize(np.uint8(disp_map), maxdis)


def main():
    if len(sys.argv) != 4: 
        print("Il faut 3 arguments : stereomatch.py im_gche.png im_dte.png disp_sortie.png")
        return 1
    
    matcher = StereoMatcher()
    loader = DataLoader(sys.argv[1], sys.argv[2])
    paras2 = {
        "tao1": 20,
        "tao2": 6,
        "L1": 34,
        "L2": 17,
        "target": "all",
        "maxdis": 64,
        "taoso": 15,
        "pi1": 1,
        "pi2": 3,
        "lamb": 30,
    }
    loader = DataLoader(sys.argv[1], sys.argv[2])
    left_cost_aggreated, _ = matcher.AD_Census_cost_with_simple_agg(
        loader, paras2
    )
    if cv2.imwrite(
        sys.argv[3], normalize(np.uint8(np.argmin(left_cost_aggreated, axis=2)), 64)
    ):
        return 0
    print("Error in saving output file.")
    return -1


if __name__ == "__main__":
    sys.exit(main())
