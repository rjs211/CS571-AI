import math
import numpy as np


class FuzzyPendulum:

    def __init__(self,pos_theta_eps, pos_omega_eps, pos_alpha_eps):
        self.theta_points = np.asarray([], dtype=np.float32)
        self.omega_points = np.asarray([], dtype=np.float32)
        self.alpha_points = np.asarray([], dtype=np.float32)

    def generate_trapezium_point_indices(self,):
    	

	def get_membership(self, value, bottom_left, top_left, top_right):
	    bottom_right = top_right + top_left - bottom_left
	    membership = 0.0
	    if bottom_left <= value < top_left:
	        pass
	        # line eqn1
	    elif top_left <= value <= top_right :
	        membership = 1.0
	    elif top_right < value <= bottom_right
	        pass

	    return membership

	def get_membership(self, ):

		pass

	def get_center_area_trapezium(self, membership, bottom_left, top_left, top_right):
	    bottom_right = top_right + top_left - bottom_left

	    top_left = bottom_left + (top_left - bottom_left) * membership
	    top_right = bottom_right + (top_right - bottom_right) * membership
	    area = top_right + bottom_right - top_left - bottom_right #  sum of parallel sides
	    area *= membership # height
	    area /= 2
	    center = (top_left + bottom_right) /2
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
			arg = [alpha_mem] + self.alpha_points[alpha_inds[pos]].tolist()
			center, area = self.get_center_area_trapezium(*arg)
			total_alpha_area += (center*area)
			total_area += area
		return  total_alpha_area / total_area




def get_alpha(theta_t,omega_t):

    """

    :param theta_t:
    :param omega_t:
    :return:

     1) get the memberships for theta_t and omega_t
     2) transform it alpha memberships
     using alpha memberships, get value
    """