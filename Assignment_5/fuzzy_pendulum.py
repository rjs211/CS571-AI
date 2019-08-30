import math
import numpy as np


class FuzzyPendulum:
    def __init__(self, use_gravity=False, pos_theta_eps=None, pos_omega_eps=None, pos_alpha_eps=None):
        self.theta_points = self.get_points(pos_theta_eps)
        self.omega_points = self.get_points(pos_omega_eps)
        self.alpha_points = self.get_points(pos_alpha_eps)
        theta_omega_to_alpha_map = [[2,1,0],
                                         [1,0,-1],
                                         [0,-1,-2]]
        self.theta_omega_to_alpha_map = theta_omega_to_alpha_map[0] + theta_omega_to_alpha_map[1] + \
                                        theta_omega_to_alpha_map[2]
        self.use_gravity = use_gravity

    def get_points(self, pos_points):
        points = [-1 * pos_points[0], 0, 0]
        pos_points = pos_points[1:]
        leftlist = []
        rightlist = []
        for i in range(len(pos_points) // 3):
            bottom_left, top_left, top_right = pos_points[3 * i: 3 * i + 3]
            rightlist += [bottom_left, top_left, top_right]
            leftlist += [top_right + top_left - bottom_left, top_right, top_left]
        leftlist = [-1 * i for i in leftlist]
        points = leftlist + points + rightlist
        # return np.asarray(points, dtype= np.float32)
        return points

    def get_membership(self, value, bottom_left, top_left, top_right):
        #todo
        bottom_right = top_right + top_left - bottom_left
        membership = 0.0
        if bottom_left <= value < top_left:
            pass
        # line eqn1
        elif top_left <= value <= top_right:
            membership = 1.0
        elif top_right < value <= bottom_right:
            pass
        return membership

    def get_center_area_trapezium(self, membership, bottom_left, top_left, top_right):
        bottom_right = top_right + top_left - bottom_left

        top_left = bottom_left + (top_left - bottom_left) * membership
        top_right = bottom_right + (top_right - bottom_right) * membership
        area = top_right + bottom_right - top_left - bottom_right  # sum of parallel sides
        area *= membership  # height
        area /= 2
        center = (top_left + bottom_right) / 2
        return center, area

    def get_alpha_memberships(self, theta_membership, omega_membership):
        theta_mem, omega_mem = np.meshgrid(theta_membership, omega_membership)
        theta_mem, omega_mem = theta_mem.reshape(-1), omega_mem.reshape(-1)
        alpha_mem = np.minimum(theta_mem, omega_mem)
        return alpha_mem

    def get_alpha(self, alpha_membership):
        total_alpha_area = 0
        total_area = 0
        for pos, alpha_mem in enumerate(alpha_membership):
            pos2 = self.theta_omega_to_alpha_map[pos]
            arg = [alpha_mem] + self.alpha_points[3 * pos2:3 * pos2 + 3]
            center, area = self.get_center_area_trapezium(*arg)
            total_alpha_area += (center * area)
            total_area += area
        return total_alpha_area / total_area

    def get_new_theta_omega(self,theta_old, omega_old, time):
        # todo if theta is <=0 or  >= max then make theta and omega zero
        # todo if theta not in range membership then give high opposite alpha.  ie NOT FUZZY
        if -0.7 <= theta_old <= 0.7:
            theta_membership = [self.get_membership(theta_old, *self.theta_points[3 * i:3 * i + 3]) for i in range(len(
                self.theta_points) // 3)]
            omega_membership = [self.get_membership(omega_old, *self.omega_points[3 * i:3 * i + 3]) for i in range(len(
                self.omega_points) // 3)]

            alpha_membership = self.get_alpha_memberships(theta_membership, omega_membership)

            alpha = self.get_alpha(alpha_membership)
        else:
            alpha = -2 if theta_old >0 else 2


        theta_disp = omega_old * time + 0.5*alpha*time*time
        theta_new = theta_old + theta_disp
        omega_new = omega_old + alpha * time

        if self.use_gravity:
            pass
            #todo

        if theta_new <=-1 :
            theta_new = -1
            omega_new = 0
        elif theta_new >= 1:
            theta_new = 1
            omega_new = 0









