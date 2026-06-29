#!/usr/bin/python3

import pytest
from orbit import RVToCOE
from config import constants
import numpy as np

# class RVToCOETestClass:
class Test_RVToCOE:
    
    r_1 = np.array([-3174.89878864632, -992.843201535607, -6922.86051061034])
    v_1 = np.array([5.29933974840672, 3.41395143650654, -2.9199470061499])
    
    e_1 = np.array([0.028935594161872, 0.009048637408142, 0.063094005670550])
    n_1 = np.array([4.595712608928938e+04, 2.653335911883275e+04, 0])
    
    LOW_TOL = 1e-4
    
    def test_calculate_semi_major_axis(self):

        # Note: This method uses an eccentricity vector to preemptively throw an error if the orbit is parabolic
        a = RVToCOE.calculate_semi_major_axis(self.r_1, self.v_1, constants.EARTH_GRAVITATIONAL_PARAMETER, [0.0289, 0.0090, 0.0631]) 
        
        expected_a = 7.178138e3

        assert a == pytest.approx(expected_a, rel=self.LOW_TOL)
        
    def test_calculate_eccentricity(self):
        
        e = RVToCOE.calculate_eccentricity(self.r_1, self.v_1, constants.EARTH_GRAVITATIONAL_PARAMETER)
        
        assert e == pytest.approx(self.e_1, rel=self.LOW_TOL)
        
    def test_calculate_inclination(self):
        
        i = RVToCOE.calculate_inclination(self.r_1, self.v_1)
        
        expected_i = np.rad2deg(1.675516081914556)
        
        assert i == pytest.approx(expected_i)
        
    def test_calculate_longitude_of_ascending_node(self):
        
        lan = RVToCOE.calculate_longitude_of_ascending_node(self.r_1, self.v_1)

        expected_lan = np.rad2deg(0.523598775598298)
        
        assert lan == pytest.approx(expected_lan)
        
    def test_calculate_argument_of_periapsis(self):
        
        aop = RVToCOE.calculate_argument_of_periapsis(self.e_1, self.n_1)

        expected_aop = np.rad2deg(1.134464013796315)
        assert aop == pytest.approx(expected_aop)
    
    def test_calculate_true_anomaly(self):
    
        true_anom = RVToCOE.calculate_true_anomaly(self.r_1, self.v_1, self.e_1)
        
        expected_anom = np.rad2deg(3.141592653589793)
        assert true_anom == pytest.approx(expected_anom)
        
    def test_calculate_line_of_nodes(self):
        
        n = RVToCOE.calculate_line_of_nodes(self.r_1, self.v_1)

        expected_n = self.n_1
        assert n == pytest.approx(expected_n)
        
    def test_calculate_specific_angular_momentum(self):
        
        h = RVToCOE.calculate_specific_angular_momentum(self.r_1, self.v_1)
        
        expected_h = np.array([2.653335911883275e+04, -4.595712608928938e+04, -5.577536838428952e+03])
        assert h == pytest.approx(expected_h)