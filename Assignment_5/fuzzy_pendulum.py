import math
import numpy as np


class FuzzyPendulum:
    """
    Class for defining the physics of the pendulum bob.
    """
    def __init__(self, use_gravity = False, theta_epsilons = None,
                 omega_epsilons = None, alpha_epsilons = None):
        """
        Function for initializing the fuzzy pendulum class.
        """
        self.theta_points = self.get_points(theta_epsilons)
        self.omega_points = self.get_points(omega_epsilons)
        self.alpha_points = self.get_points(alpha_epsilons)
        theta_omega_to_alpha_map = [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
        self.theta_omega_to_alpha_map = theta_omega_to_alpha_map[0] + \
                                        theta_omega_to_alpha_map[1] + \
                                        theta_omega_to_alpha_map[2]
        self.theta_omega_to_alpha_map = [i + 2 for i in self.theta_omega_to_alpha_map]
        self.use_gravity = use_gravity
        self.g = 10

    def get_points(self, positive_points):
        """
        Function for generating the membership profile points.
        """
        points = [-1 * positive_points[0], 0, 0]
        positive_points = positive_points[1 :]
        left_list = []
        right_list = []

        for i in range(len(positive_points) // 3):
            bottom_left, top_left, top_right = positive_points[3 * i : 3 * i + 3]
            right_list += [bottom_left, top_left, top_right]
            left_list += [top_right + top_left - bottom_left, top_right, top_left]

        left_list = [-1 * i for i in left_list]
        points = left_list + points + right_list

        return points

    def get_membership(self, value, bottom_left, top_left, top_right):
        """
        Function for .
        """
        bottom_right = top_right + top_left - bottom_left
        membership = 0.0

        if bottom_left <= value < top_left:
            membership = (value - bottom_left) / (top_left - bottom_left)
        elif top_left <= value <= top_right:
            membership = 1.0
        elif top_right < value <= bottom_right:
            membership = (bottom_right - value) / (bottom_right - top_right)

        return membership

    def get_center_area_trapezium(self, membership, bottom_left,
                                  top_left, top_right):
        """
        Function for finding the area of the trapezium under
        a particular profile.
        """
        bottom_right = top_right + top_left - bottom_left
        top_left = bottom_left + (top_left - bottom_left) * membership
        top_right = bottom_right + (top_right - bottom_right) * membership
        area = top_right + bottom_right - top_left - bottom_right  # sum of parallel sides
        area *= membership  # height
        area /= 2
        center = (top_left + bottom_right) / 2

        return center, area

    def get_alpha_membership(self, theta_membership, omega_membership):
        """
        Function for generating the membership profile
        for the current applied.
        """
        new_theta_membership, new_omega_membership = np.meshgrid(theta_membership, omega_membership)
        new_theta_membership, new_omega_membership = new_theta_membership.reshape(-1), new_omega_membership.reshape(-1)
        alpha_membership = np.minimum(new_theta_membership, new_omega_membership)

        return alpha_membership

    def get_alpha(self, alpha_membership):
        """
        Function for finding the right value of current
        to be applied.
        """
        total_alpha_area = 0
        total_area = 0

        for pos, alpha_mem in enumerate(alpha_membership):
            new_pos = self.theta_omega_to_alpha_map[pos]
            arg = [alpha_mem] + self.alpha_points[3 * new_pos : 3 * new_pos + 3]
            center, area = self.get_center_area_trapezium(*arg)
            total_alpha_area += (center * area)
            total_area += area

        return total_alpha_area / total_area

    def get_new_theta_omega(self, theta_old, omega_old, time):
        """
        Function for calculating the changes in the angle with
        the vertical and the angular velocity of the pendulum.
        """
        # Done if theta is <= 0 or >= max then make theta and omega zero.
        # Done if theta not in range membership then give high opposite alpha, i.e. NOT FUZZY.
        if -0.3 * math.pi * 0.5 <= theta_old <= 0.3 * math.pi * 0.5:
            theta_membership = [self.get_membership(theta_old, *self.theta_points[3 * i : 3 * i + 3]) for i in range(len(
                self.theta_points) // 3)]
            omega_membership = [self.get_membership(omega_old, *self.omega_points[3 * i : 3 * i + 3]) for i in range(len(
                self.omega_points) // 3)]
            alpha_membership = self.get_alpha_membership(theta_membership, omega_membership)
            alpha = self.get_alpha(alpha_membership)
        else:
            alpha = -12 if theta_old > 0 else 12

        theta_displacement = omega_old * time + 0.5 * alpha * time * time
        theta_new = theta_old + theta_displacement
        omega_new = omega_old + alpha * time

        if self.use_gravity:
            my_angle = theta_old
            g_component = math.sin(my_angle) * self.g
            theta_gravity_displacement = 0.5 * g_component * time * time
            theta_new += theta_gravity_displacement
            omega_new += g_component * time

        if theta_new <=-1 * math.pi * 0.5 :
            theta_new = -1 * math.pi * 0.5
            omega_new = 0
        elif theta_new >= 1 * math.pi * 0.5:
            theta_new = 1 * math.pi * 0.5
            omega_new = 0

        return theta_new, omega_new
