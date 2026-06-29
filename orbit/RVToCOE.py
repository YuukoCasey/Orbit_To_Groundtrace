#!/usr/bin/python3

import numpy as np
# from config import constants

def rv_to_coe(r: np.array, v: np.array, mu: float) -> np.array:
    
    e_vector = calculate_eccentricity(r, v, mu)
    e = np.linalg.norm(e_vector)
    a = calculate_semi_major_axis(r, v, mu, e_vector)
    i = calculate_inclination(r, v)
    
    lan = calculate_longitude_of_ascending_node(r, v)
    n_vector = calculate_line_of_nodes(r, v)
    arg_peri = calculate_argument_of_periapsis(e_vector, n_vector)
    true_anom = calculate_true_anomaly(r, v, e_vector)
    
    ret_COE = [a, e, i, lan, arg_peri, true_anom]
    
    return ret_COE

def calculate_semi_major_axis(r: np.array, v: np.array, mu: float, e: np.array) -> float:
    # Uses the Vis-viva equation to calculate the semi-major axis
    # Note: Can calculate as infinity if orbit is parabolic
    
    # If eccentricity magnitude is 1, then it is parabolic and will have an infinite SMA
    e_mag = np.linalg.norm(e)
    if (e_mag == 1):
        raise ZeroDivisionError("Parabolic orbits have an infinite semi-major axis")
    
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    a = (r_mag*mu) / ( (2*mu) - ((v_mag**2)*r_mag) )
    # a = (-mu*r_mag)/( ((v_mag**2)*r_mag) - (2*mu) )
    return a

def calculate_specific_angular_momentum(r: np.array, v: np.array) -> np.array:
    h = np.cross(r, v)
    return h

def calculate_eccentricity(r: np.array, v: np.array, mu: float) -> np.array:
    h = calculate_specific_angular_momentum(r, v)
    
    ecc_vector = (np.cross(v,h))/mu - (r/np.linalg.norm(r))
    return ecc_vector

def calculate_inclination(r: np.array, v: np.array) -> float:

    # Calculate the inclination of an orbit in degrees

    h_vector = calculate_specific_angular_momentum(r, v)
    
    h_z = h_vector.item(2)
    h = np.linalg.norm(h_vector)
    
    i = np.arccos(h_z / h)
    i = np.rad2deg(i)
    
    return i

def calculate_line_of_nodes(r: np.array, v: np.array) -> np.array:
    
    k = np.array([0,0,1])
    h = calculate_specific_angular_momentum(r, v)
    n = np.cross(k, h)
    return n

def calculate_longitude_of_ascending_node(r: np.array, v: np.array) -> float:

    # Calculate the longitude of ascending node in degrees

    n_vector = calculate_line_of_nodes(r, v)
    n_y = n_vector.item(1)
    n_x = n_vector.item(0)
    n = np.linalg.norm(n_vector)
    
    lan = 0.0
    if (n_y >= 0):
        lan = np.arccos(n_x/n)
    else:
        lan = (2*np.pi) - ( np.arccos(n_x/n) )
        
    lan = np.rad2deg(lan)
        
    return lan

def calculate_argument_of_periapsis(e: np.array, n: np.array) -> float:

    e_z = e.item(2)
    
    arg_peri = 0
    
    e_mag = np.linalg.norm(e)
    n_mag = np.linalg.norm(n)
    
    arg_peri = np.arccos( (np.dot(n,e))/(n_mag*e_mag) )
    
    if (e_z < 0):
        arg_peri = (2*np.pi) - arg_peri
        
    arg_peri = np.rad2deg(arg_peri)
    return arg_peri

def calculate_true_anomaly(r: np.array, v: np.array, e: np.array) -> float:
    
    # First, you need to know whether the object has a positive or negative radial velocity
    
    r_hat = r / np.linalg.norm(r)
    v_r = np.dot(v, r_hat)

    r_mag = np.linalg.norm(r)
    e_mag = np.linalg.norm(e)

    true_anom = 0

    if v_r >= 0: # If moving towards the earth or what is being orbited
        true_anom = np.arccos( np.dot(r, e)/(r_mag * e_mag) )
    else:
        true_anom = (2*np.pi) - np.arccos( np.dot(r, e)/(r_mag * e_mag) )
    
    true_anom = np.rad2deg(true_anom)

    return true_anom
