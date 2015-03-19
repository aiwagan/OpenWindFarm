import os.path

from dolfin import *
from dolfin_adjoint import *
from solver import Solver
from .. import finite_elements
from ..problems import WFProblem
from ..helpers import FrozenClass

class IPCSWFSolverParameters(FrozenClass):
    dolfin_solver = {"newton_solver": {}}

    # Large eddy simulation
    les_model = False
    les_parameters = {'smagorinsky_coefficient': 1e-2}

    # Performance settings
    quadrature_degree = -1
    cpp_flags = ["-O3", "-ffast-math", "-march=native"]

    def __init__(self):

        linear_solver = 'mumps' if ('mumps' in map(lambda x: x[0], linear_solver_methods())) else 'default'
        preconditioner = 'default'

        self.dolfin_solver["newton_solver"]["linear_solver"] = linear_solver
        self.dolfin_solver["newton_solver"]["preconditioner"] = preconditioner
        self.dolfin_solver["newton_solver"]["maximum_iterations"] = 20


class IPCSWFSolver(Solver):

    def __init__(self, problem, parameters):

        if not isinstance(problem, WFProblem):
            raise TypeError, "problem must be of type WFProblem"

        if not isinstance(parameters, IPCSWFSolverParameters):
            raise TypeError, "parameters must be of type IPCSWFSolverParameters."

        self.problem = problem
        self.parameters = parameters
        self.optimisation_iteration = 0

        # If cache_for_nonlinear_initial_guess is true, then we store all
        # intermediate state variables in this dictionary to be used for the
        # next solve
        self.state_cache = {}

        self.mesh = problem.parameters.domain.mesh
        self.V, self.Q = self.problem.parameters.finite_element(self.mesh)

        #from dummy solver
        # self.tf = None  
        # self.current_state = None

    @staticmethod
    def default_parameters():
        return IPCSWFSolverParameters()

    def _finished(self, current_time, finish_time):
        return float(current_time - finish_time) >= - 1e3*DOLFIN_EPS

    def _generate_strong_bcs(self, dgu):

        if dgu:
            bcu_method = "geometric"
        else:
            bcu_method = "topological"

        bcs = self.problem.parameters.bcs
        facet_ids = self.problem.parameters.domain.facet_ids

        # Generate velocity boundary conditions
        bcs_u = []
        for _, expr, facet_id, _ in bcs.filter("u", "strong_dirichlet"):
            bc = DirichletBC(self.V, expr, facet_ids, facet_id, method=bcu_method)
            bcs_u.append(bc)

        # Generate free-surface boundary conditions
        bcs_p = []
        for _, expr, facet_id, _ in bcs.filter("p", "strong_dirichlet"):
            bc = DirichletBC(self.Q, expr, facet_ids, facet_id)
            bcs_p.append(bc)

        return bcs_u, bcs_p

    # def setup(self, W, turbine_field, annotate=True):
    #     (v, q) = TestFunctions(W)
    #     (u, h) = TrialFunctions(W)

    #     # Mass matrices
    #     self.M = inner(v, u) * dx
    #     self.M += inner(q, h) * dx

    #     self.A = (1.0 + turbine_field) * inner(v, u) * dx
    #     self.A += inner(q, h) * dx

    #     self.tf = Function(turbine_field, name="turbine_friction", annotate=annotate)

    #     self.annotate = annotate

    def solve(self, annotate=True):
        '''
        IPCS solution to the Navier Stokes Equations
        '''

        # Initialise solver settings
        if not type(self.problem) == WFProblem:
            raise TypeError("Do not know how to solve problem of type %s." %
                type(self.problem))

        # Get parameters
        problem_params = self.problem.parameters
        solver_params = self.parameters
        farm = problem_params.wind_farm
        if farm:
            turbine_friction = farm.turbine_cache["turbine_field"]
        mesh = problem_params.domain.mesh
        max_inner_iters = 300
        max_error = 1e-7

        # Performance settings
        parameters['form_compiler']['quadrature_degree'] = solver_params.quadrature_degree
        parameters['form_compiler']['cpp_optimize_flags'] = " ".join(solver_params.cpp_flags)
        parameters['form_compiler']['cpp_optimize'] = True
        parameters['form_compiler']['optimize'] = True

        # Get domain measures
        ds = problem_params.domain.ds
        dx = problem_params.domain.dx

        # Get temporal settings
        theta = Constant(problem_params.theta)
        dt = Constant(problem_params.dt)
        finish_time = Constant(problem_params.finish_time)
        t = Constant(problem_params.start_time)

        nu = problem_params.viscosity
        # f_u = problem_params.f_u
        # if f_u is None:
        #     f_u = Constant((0, 0))

        include_les = solver_params.les_model

        # Get boundary conditions
        bcs = problem_params.bcs

        # Get function spaces
        V, Q = self.V, self.Q
        dgu = "Discontinuous" in str(V)

        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)

        #define functions
        u0 = Function(V)  # velocity at previous time step t=n-1
        u1 = Function(V)  # velocity two time steps back t=n-2
        u_tent = Function(V)  # tentative velocity
        u_ = Function(V)         # current velocity at t=n
        p0 = Function(Q)   # previous pressure
        p_ = Function(Q)         # current pressure

        # Load initial condition
        # Projection is necessary to obtain 2nd order convergence
        uIC = project(problem_params.initial_condition_u, self.V)
        pIC = project(problem_params.initial_condition_p, self.Q)

        u0.assign(uIC, annotate=False)
        u1.assign(uIC, annotate=False)

        p0.assign(pIC, annotate=False)
        p_.assign(pIC, annotate=False)

        # Large eddy model
        if include_les:
            les_V = FunctionSpace(problem_params.domain.mesh, "CG", 1)
            les = LES(les_V, u0,
                    solver_params.les_parameters['smagorinsky_coefficient'])
            eddy_viscosity = les.eddy_viscosity
            nu += eddy_viscosity
        else:
            eddy_viscosity = None

        #setup equations
        f_ad = Constant((0., 0.))
        f = Constant((0., 0.))

        #convection w/ adams-bashforth projection
        nonlinearity = inner(dot(1.5*u0 - 0.5*u1, nabla_grad(0.5*(u + u0))), v)*dx

        # #tentative velocity
        F_tent = (1./dt)*inner(u - u0, v)*dx + nonlinearity\
        + nu*inner(grad(0.5*(u + u0)), grad(v))*dx + inner(grad(p0), v)*dx\
        - inner(f, v)*dx     # grad(p0) or grad(p_)?
        #- inner(Ct*f_ad, v)*dx

        # Pressure Projection
        F_pres = inner(grad(p - p0), grad(q))*dx + (1./dt)*q*div(u_)*dx #u_ or u?

        # Velocity Update
        F_up = inner(u - u_, v)*dx + dt*inner(grad(p_ - p0), v)*dx #grad(p_ - p0) or grad(p_)?

        bcu, bcp = self._generate_strong_bcs(dgu)

        #assemble equations
        a_tent, L_tent = system(F_tent)
        a_pres, L_pres = system(F_pres)
        a_up, L_up = system(F_up)
        A_tent = assemble(a_tent)
        A_pres = assemble(a_pres)
        A_up = assemble(a_up)

        for bc in bcu: bc.apply(A_up)

        # solver_tent = KrylovSolver('gmres', 'ilu')

        # solver_pres = KrylovSolver('cg', 'petsc_amg')

        solver_tent = KrylovSolver('bicgstab', 'additive_schwarz')

        solver_pres = KrylovSolver('cg', 'petsc_amg')


        solver_up = KrylovSolver('cg', 'additive_schwarz')

        # Get the nullspace if there are no pressure boundary conditions
        foo = Function(Q)     # auxiliary vector for setting pressure nullspace
        if not bcp:
            null_vec = Vector(foo.vector())
            Q.dofmap().set(null_vec, 1.0)
            null_vec *= 1.0/null_vec.norm('l2')
            null_space = VectorSpaceBasis([null_vec])
            solver_pres.set_nullspace(null_space)
            solver_pres.set_transpose_nullspace(null_space)
            solver_pres.null_space = null_space

        yield({"time": t,
               "u": u0,
               "p": p0,
               "eddy_viscosity": eddy_viscosity,
               "is_final": self._finished(t, finish_time)})

        log(INFO, "Start of time loop")
        adjointer.time.start(t)
        timestep = 0

        # De/activate annotation
        annotate_orig = parameters["adjoint"]["stop_annotating"]
        parameters["adjoint"]["stop_annotating"] = not annotate

        while not self._finished(t, finish_time):
            # Update timestep
            timestep += 1
            t = Constant(t + dt)

            inner_iter = 0
            udiff = 1e8

            # Update bc's
            t_theta = Constant(t - (1.0 - theta) * dt)
            bcs.update_time(t, only_type=["strong_dirichlet"])
            bcs.update_time(t_theta, exclude_type=["strong_dirichlet"])

            # # Update source term
            # if f_u is not None:
            #     f_u.t = Constant(t_theta)

            # if include_les:
            #     log(PROGRESS, "Compute eddy viscosity.")
            #     les.solve()

            # Compute tentative velocity step
            log(PROGRESS, "Iterative solve for tentative velocity and pressure correction.")
            while udiff > max_error and inner_iter < max_inner_iters:     # print inner_iter
                inner_iter += 1
                # tentative vel
                u_tmp = u_.vector()[:]
                b = assemble(L_tent)
                for bc in bcu: bc.apply(A_tent,b)
                # [bc.apply(A_tent, b) for bc in bcs_u]
                solver_tent.solve(A_tent, u_.vector(), b)
                uout=u_.vector()[:]
                udiff = norm(u_tmp - uout)

                # pressure correction
                b = assemble(L_pres)
                for bc in bcp: bc.apply(A_pres,b)
                if bcp == []:
                    solver_pres.null_space.orthogonalize(b)
                solver_pres.solve(A_pres, p_.vector(), b)

            #velocity update
            log(PROGRESS, "Solve for velocity update.")
            b = assemble(L_up)
            for bc in bcu: bc.apply(A_up,b)
            # [bc.apply(A_up, b) for bc in bcs_u]
            solver_up.solve(A_up, u_.vector(), b) 

            u1.assign(u0)
            u0.assign(u_)
            p0.assign(p_)

            # Increase the adjoint timestep
            adj_inc_timestep(time=float(t), finished=self._finished(t,
                finish_time))

            yield({"time": t,
                   "u": u0,
                   "p": p0,
                   "eddy_viscosity": eddy_viscosity,
                   "is_final": self._finished(t, finish_time)})

        # Reset annotation flag
        parameters["adjoint"]["stop_annotating"] = annotate_orig

        log(INFO, "End of time loop.")



       