import os.path

from dolfin import *
from dolfin_adjoint import *
from solver import Solver
from .. import finite_elements
from ..problems import WFProblem
from ..helpers import StateWriter, FrozenClass

class CoupledWFSolverParameters(FrozenClass):
    dolfin_solver = {"newton_solver": {}}
    dump_period = 1
    print_individual_turbine_power = False

    omega = 1.0

    # If we're printing individual turbine information, the solver needs
    # the helper functional instantiated in the reduced_functional which will live here
    output_writer = None

    # Output settings
    output_dir = os.curdir
    output_turbine_power = False

    # Performance settings
    cache_forward_state = True
    quadrature_degree = -1
    cpp_flags = ["-O3", "-ffast-math", "-march=native"]
    revolve_parameters = None  # (strategy,
                               # snaps_on_disk,
                               # snaps_in_ram,
                               # verbose)

    def __init__(self):

        linear_solver = 'mumps' if ('mumps' in map(lambda x: x[0], linear_solver_methods())) else 'default'
        preconditioner = 'default'

        self.dolfin_solver["newton_solver"]["linear_solver"] = linear_solver
        self.dolfin_solver["newton_solver"]["preconditioner"] = preconditioner
        self.dolfin_solver["newton_solver"]["maximum_iterations"] = 20


class CoupledWFSolver(Solver):

    def __init__(self, problem, solver_params):

        if not isinstance(problem, WFProblem):
            raise TypeError, "problem must be of type Problem"

        if not isinstance(solver_params, CoupledWFSolverParameters):
            raise TypeError, "solver_params must be of type CoupledWFSolverParameters."

        self.problem = problem
        self.parameters = solver_params

        self.state_cache = {}
        self.current_state = None

        self.mesh = problem.parameters.domain.mesh
        V, H = self.problem.parameters.finite_element(self.mesh)
        self.function_space = MixedFunctionSpace([V, H])
        self.optimisation_iteration = 0

    @staticmethod
    def default_parameters():
        """ Return the default parameters for the :class:`CoupledWFSolver`.
        """
        return CoupledWFSolverParameters()

    def _finished(self, current_time, finish_time):
        return float(current_time - finish_time) >= - 1e3*DOLFIN_EPS

    def _generate_bcs(self):

        bcs = self.problem.parameters.bcs
        facet_ids = self.problem.parameters.domain.facet_ids
        fs = self.function_space

        # Generate velocity boundary conditions
        bcs_u = []
        for _, expr, facet_id, _ in bcs.filter("u", "strong_dirichlet"):
            bc = DirichletBC(fs.sub(0), expr, facet_ids, facet_id)
            bcs_u.append(bc)

        # # Generate free-surface boundary conditions
        # bcs_p = []
        # for _, expr, facet_id, _ in bcs.filter("p", "strong_dirichlet"):
        #     bc = DirichletBC(fs.sub(1), expr, facet_ids, facet_id)
        #     bcs_p.append(bc)

        return bcs_u 
    
    def solve(self, annotate=True):
        '''Solve the coupled NSE equations'''

        problem_params = self.problem.parameters
        solver_params = self.parameters
        farm = problem_params.wind_farm

        # problem_params = self.problem.parameters
        # farm = problem_params.wind_farm
        # turbine_friction = farm.turbine_cache["turbine_field"]
        # mesh = problem_params.domain.mesh

        # Performance settings
        parameters['form_compiler']['quadrature_degree'] = \
            solver_params.quadrature_degree
        parameters['form_compiler']['cpp_optimize_flags'] = \
            " ".join(solver_params.cpp_flags)
        parameters['form_compiler']['cpp_optimize'] = True
        parameters['form_compiler']['optimize'] = True

        # Get domain measures
        ds = problem_params.domain.ds

        log(INFO, "Solve a steady-state NSE problem")

        # Get temporal settings
        theta = Constant(1.)
        dt = Constant(1.)
        finish_time = Constant(0.5)

        t = Constant(0.)

        include_time_term = False

        nu = problem_params.viscosity
        # f_u = problem_params.f_u
        # if f_u is None:
        #     f_u = Constant((0, 0))

         # Get boundary conditions
        bcs = problem_params.bcs

        cache_forward_state = solver_params.cache_forward_state


        u_dg = "Discontinuous" in str(self.function_space.split()[0])

        up = TrialFunction(self.function_space)
        u, p = split(up)
        v, q = TestFunctions(self.function_space)

        up_ = Function(self.function_space, name="next_iteration")
        up_1 = Function(self.function_space, name="last_iteration")

        #get initial conditions, project for better convergence
        ic = project(problem_params.initial_condition, self.function_space)
        up_1.assign(ic, annotate=False)

        u_, p_ = split(up_)
        u_1, p_1 = split(up_1)

        up_.assign(ic,annotate=annotate)

        if not farm:
            tf = Constant(0)
        elif type(farm.friction_function) == list:
            tf = Function(farm.friction_function[0], name="turbine_friction", annotate=annotate)
        else:
            tf = Function(farm.friction_function, name="turbine_friction", annotate=annotate)

        if farm:
            F_ad = tf*dot(u_,u_)**0.5*inner(u_, v)*farm.site_dx

        # F_ad = Constant((0.,0.))*dx
        # setup equations
        F = inner(dot(grad(u_), u_), v)*dx + nu*inner(grad(u_), grad(v))*dx \
        - inner(p_, div(v))*dx - inner(q, div(u_))*dx + F_ad

        # a=lhs(F)
        # L=rhs(F)

        # J = derivative(F, up_, up)
        # A = Matrix()

        # Generate the scheme specific strong boundary conditions
        facet_ids = self.problem.parameters.domain.facet_ids
        fs = self.function_space
        bcs_u = [DirichletBC(fs.sub(0), Constant((2.0,0.)),"on_boundary && near(x[0], 0)")]


        # #assemble
        # A = assemble(a, annotate=annotate)
        # for bc in bcs_u:
        #     bc.apply(A, annotate=annotate)

        ############################### Perform the simulation ###########################

        if solver_params.dump_period > 0:
            writer = StateWriter(solver=self)
            if type(self.problem) == WFProblem:
                log(INFO, "Writing state to disk...")
                writer.write(up_)

        yield({"time": t,
               "u": u_1,
               "p": p_1,
               "tf": tf,
               "state": up_1,
               "is_final": self._finished(t, finish_time)})

        log(INFO, "Start of time loop")
        adjointer.time.start(t)
        timestep = 0
        while not self._finished(t, finish_time):
            # Update timestep
            timestep += 1
            t = Constant(t + dt)

            # b = assemble(L, annotate=annotate)
            # [bc.apply(b) for bc in bcs_u]

            # # Update bc's
            # t_theta = Constant(t - (1.0 - theta) * dt)
            # bcs.update_time(t, only_type=["strong_dirichlet"])
            # bcs.update_time(t_theta, exclude_type=["strong_dirichlet"])

            # Update source term
            # f_u.t = Constant(t_theta)

            # Set the initial guess for the solve
            if cache_forward_state and self.state_cache.has_key(float(t)):
                log(INFO, "Read initial guess from cache for t=%f." % t)
                # Load initial guess for solver from cache
                up_.assign(self.state_cache[float(t)], annotate=False)

            elif not include_time_term:
                log(INFO, "Set the initial guess for the nonlinear solver to the initial condition.")
                # Reset the initial guess after each timestep
                ic = problem_params.initial_condition
                up_.assign(ic, annotate=False)

            #solve
            log(INFO, "Solve steady NS equations.")

            solve(F==0, up_, bcs=bcs_u,
                  solver_parameters=solver_params.dolfin_solver,
                  annotate=annotate)
                  

            # up_1.vector().zero()
            # solve(A, up_1.vector(), up, annotate=annotate)
            # up_.vector().axpy(-omega, up_1.vector())
            # up = assemble(F, tensor=up, annotate=annotate)
            # for bc in strong_bcs:
            #     bc.apply(up.vector(), up_.vector())

            # After the timestep solve, update state
            up_1.assign(up_, annotate=annotate)

            if cache_forward_state:
                # Save state for initial guess cache
                log(INFO, "Cache solution t=%f as next initial guess." % t)
                if not self.state_cache.has_key(float(t)):
                    self.state_cache[float(t)] = Function(self.function_space)
                self.state_cache[float(t)].assign(up_, annotate=False)

            # Set the control function for the upcoming timestep.
            if farm:
                if type(farm.friction_function) == list:
                    tf.assign(farm.friction_function[timestep])
                else:
                    tf.assign(farm.friction_function)

            if (solver_params.dump_period > 0 and
                timestep % solver_params.dump_period == 0):
                log(INFO, "Write state to disk...")
                writer.write(up_1)

            # Increase the adjoint timestep
            adj_inc_timestep(time=float(t), finished=self._finished(t,
                finish_time))

            yield({"time": t,
                   "u": u_,
                   "p": p_,
                   "tf": tf,
                   "state": up_,
                   "is_final": self._finished(t, finish_time)})

        # If we're outputting the individual turbine power
        if self.parameters.print_individual_turbine_power:
            self.parameters.output_writer.individual_turbine_power(self)

        log(INFO, "End of time loop.")


