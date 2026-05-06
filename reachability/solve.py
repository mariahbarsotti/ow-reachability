import hj_reachability
from .dynamics import open_loop_dynamics, control_jacobian, disturbance_jacobian

class OWDynamics(hj_reachability.Dynamics):
    def __init__(self, control_space, disturbance_space):
        super.__init__(
            control_mode = 'min',
            disturbance_mode = 'max',
            control_space = control_space
            disturbance_space = disturbance_space,
        )

    def f(self, state, time):
        return f(state, time)
    
    def g(self, state, time):
        return g(state, time)
    
    def h(self, state, time):
        return h(state, time)