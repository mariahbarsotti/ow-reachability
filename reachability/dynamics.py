''' 5-state oblique-wing longitudinal dynamics
State vector:
    x[0] = theta   - pitch angle [rad]
    x[1] = q       - body-axis pitch rate [rad/s]
    x[2] = lam     - sweep angle [rad]   (using 'lam' since 'lambda' is a Python keyword)
    x[3] = lam_d   - sweep rate [rad/s] 
    x[4] = alpha   - angle of attack [rad]

Control vector:
    u[0] = delta_e - elevator deflection [rad], bounded
    u[1] = lam_dot - commanded sweep rate [rad/s], bounded

Disturbance vector:
    d[0] = w_g     - vertical gust velocity [m/s], adversarial bound

Dynamics are affine in control and disturbance:
    x_dot = f(x,t) + g(x,t) @ u + h(x,t) @ d    
'''

import jax.numpy as jnp


def f(x, t):
    '''drift: what the system does without control or distrbance'''
    # inputs: state x and time t
    # outputs: vector f(x,t)
    theta, q, lambda, lambda_dot, alpha = x
    '''f(x,t) without control or disturbance'''
    f = jnp.array([q, ..., ...])
    return f

def g(x, t):
    '''control jacobian: controller affine dynamics'''
    # inputs: state x and time t
    # outputs: matrix g(x,t)
    theta, q, lambda, lambda_dot, alpha = x
    g = jnp.array([q, ..., ...])
    return g

def h(x, t):
    '''disturbance jacobian: disturbance affine dynamics'''
    # inputs: state x and time t
    # outputs: matrix h(x,t)
    theta, q, lambda, lambda_dot, alpha = x
    h = jnp.array([q, ..., ...])
    return h
