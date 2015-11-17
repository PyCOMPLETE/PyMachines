from __future__ import division

import numpy as np
from scipy.constants import c

from PyHEADTAIL.general.element import Element

try:
	from PyHEADTAIL.trackers.transverse_tracking_cython import TransverseMap
	from PyHEADTAIL.trackers.detuners_cython import (Chromaticity,
                                                     AmplitudeDetuning)
except ImportError as e:
	print ("*** Warning: could not import cython variants of trackers, "
           "did you cythonize (use the following command)?\n"
           "$ ./install \n"
           "Falling back to (slower) python version.")
	from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
	from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning

from PyHEADTAIL.trackers.simple_long_tracking import LinearMap, RFSystems

class BasicSynchrotron(Element):
	def __init__(self, optics_mode='smooth', circumference=None, n_segments=None, s=None,
		alpha_x=None, beta_x=None, D_x=None, alpha_y=None, beta_y=None, D_y=None, 
		accQ_x=None, accQ_y=None, Qp_x=0, Qp_y=0, app_x=0, app_y=0, app_xy=0, 
		alpha_mom_compaction=None, longitudinal_mode=None, Q_s=None, 
		h_RF=None, V_RF=None, dphi_RF=None, p0=None, p_increment=None, 
		charge, mass, **kwargs):
		
		self.optics_mode = optics_mode
		self.longitudinal_mode = longitudinal_mode
		self.p0 = p0
		self.charge = charge
		self.mass = mass
		
		detuners = []
		if Qp_x != 0 or Qp_y != 0:
			detuners.append(Chromaticity(Qp_x, Qp_y))
		if app_x != 0 or app_y != 0 or app_xy != 0:
			detuners.append(AmplitudeDetuning(app_x, app_y, app_xy))
	
		if optics_mode == 'smooth':
			if circumference is None:
				raise ValueError('circumference has to be specified if optics_mode = "smooth"')
	
			if  n_segments is None:
				raise ValueError('n_segments has to be specified if optics_mode = "smooth"')
	
			if s is not None:
				raise ValueError('s vector cannot be provided if optics_mode = "smooth"')
			
			
			s = (np.arange(0, n_segments + 1)
				  * circumference / n_segments)
			
			self.transverse_map = TransverseMap(s=s,
             alpha_x=0.*s, 
             beta_x=0.*s+beta_x, 
             D_x=0.*s+D_x,
             alpha_y=0.*s, 
             beta_y=0.*s+beta_y, 
             D_y=0.*s+D_y,
             accQ_x=accQ_x, accQ_y=accQ_y, detuners=detuners)
			
	
		elif optics_mode == 'non-smooth':
			if circumference is not None:
				raise ValueError('circumference cannot be provided if optics_mode = "non-smooth"')
	
			if  n_segments is not None:
				raise ValueError('n_segments cannot be provided if optics_mode = "non-smooth"')
	
			if s is None:
				raise ValueError('s has to be specified if optics_mode = "smooth"')
			
			self.transverse_map = TransverseMap(s=s,
             alpha_x=alpha_x, 
             beta_x=beta_x, 
             D_x=D_x,
             alpha_y=alpha_y, 
             beta_y=beta_y, 
             D_y=D_y,
             accQ_x=accQ_x, accQ_y=accQ_y, detuners=detuners)
	
		else:
			raise ValueError('optics_mode not recognized')

		self.circumference = s[-1]
		
		
		# compute the index of the element before which to insert
		# the longitudinal map
		if longitudinal_mode is not None:
			for insert_before, si in enumerate(s):
				if si > 0.5 * self.circumference:
					break
		
		if longitudinal_mode == 'linear':
			self.longitudinal_map = LinearMap(
				np.atleast_1d(alpha_mom_compaction),
				self.circumference, Q_s,
				D_x=self.transverse_map.D_x[insert_before],
				D_y=self.transverse_map.D_y[insert_before]
			)
		elif longitudinal_mode == 'non-linear':
			self.longitudinal_map = RFSystems(
				self.circumference, np.atleast_1d(h_RF),
				np.atleast_1d(V_RF), np.atleast_1d(dphi_RF),
				np.atleast_1d(alpha_mom_compaction), self.gamma, self.p_increment,
				D_x=self.transverse_map.D_x[insert_before],
				D_y=self.transverse_map.D_y[insert_before]
			)
		else:
			raise NotImplementedError(
				'Something wrong with longitudinal_mode')
			
	@property
	def gamma(self):
		return self._gamma
	@gamma.setter
	def gamma(self, value):
		self._gamma = value
		self._beta = np.sqrt(1 - self.gamma**-2)
		self._betagamma = np.sqrt(self.gamma**2 - 1)
		self._p0 = self.betagamma * self.mass * c

	@property
	def beta(self):
		return self._beta
	@beta.setter
	def beta(self, value):
		self.gamma = 1. / np.sqrt(1-value**2)

	@property
	def betagamma(self):
		return self._betagamma
	@betagamma.setter
	def betagamma(self, value):
		self.gamma = np.sqrt(value**2 + 1)

	@property
	def p0(self):
		return self._p0
	@p0.setter
	def p0(self, value):
		self.gamma = value / (self.mass * self.beta * c)

	
	