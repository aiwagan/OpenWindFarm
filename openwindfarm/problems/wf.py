from dolfin_adjoint import Constant
from problem import Problem
from ..boundary_conditions import BoundaryConditionSet
from ..helpers import FrozenClass
from .. import finite_elements


class WFProblemParameters(FrozenClass):
    """ A set of parameters for a :class:`WFProblem`.
    """

    # Time parameters
    theta = Constant(0.5)
    dt = 0.05
    start_time = 0.0
    finish_time = 1.0
    rho = 1.0

    # Functional time integration parameters
    functional_final_time_only = True

    viscosity = Constant(2.e-5)

    # Re_tau = 178.12
    Re_tau = 392.24

    velocity_degree = 2
    pressure_degree = 1

    max_inner_iters = 300
    max_error = 1e-7

    # Finite element settings
    finite_element = staticmethod(finite_elements.p2p1)

    # Initial condition
    initial_condition = Constant((1e-16, 0, 0))

    # Wind farm
    wind_farm = None

    # Domain
    domain = None

    initial_condition = Constant((1., 0, 0))
    initial_condition_u = Constant((1e-16, 0))
    initial_condition_p = Constant((0))

    # Boundary conditions
    bcs = BoundaryConditionSet()

class WFProblem(Problem):

    def __init__(self, parameters):
        """ Instantiates a new :class:`WFProblem` object.

            :parameter parameters: A :class:`WFProblemParameters`
                object containing the parameters of the problem.
        """

        if not isinstance(parameters, WFProblemParameters):
            raise TypeError("parameters must be of type \
WFProblemParameters.")

        self.parameters = parameters

    @property
    def _is_transient(self):
        return True

    @staticmethod
    def default_parameters():
        ''' Returns a :class:`WFProblemParameters` with default
            parameters. '''

        return WFProblemParameters()
