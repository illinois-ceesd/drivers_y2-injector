"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
import sys
import yaml
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import math
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial
from pytools.obj_array import make_obj_array
from mirgecom.fluid import make_conserved

from arraycontext import thaw, freeze
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DTAG_BOUNDARY
#from grudge.op import nodal_max, nodal_min
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    set_sim_state
)

from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    generate_and_distribute_mesh,
    check_step,
    write_visfile,
    check_naninf_local,
    check_range_local,
    get_sim_timestep
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)

from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    #IsothermalNoSlipBoundary,
    IsothermalWallBoundary,
    AdiabaticSlipBoundary,
    OutflowBoundary,
    DummyBoundary
)
import cantera
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from mirgecom.gas_model import GasModel, make_fluid_state


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


from grudge.trace_pair import TracePair
from mirgecom.inviscid import inviscid_flux_rusanov
from mirgecom.viscous import viscous_flux_central


class IsothermalWallBoundaryLocal(PrescribedFluidBoundary):
    r"""Isothermal viscous wall boundary.
    This class implements an isothermal wall by:
    Pescribed flux, *not* with Riemann solver
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self._wall_temp = wall_temperature
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.isothermal_wall_state,
            inviscid_flux_func=self.inviscid_wall_flux,
            viscous_flux_func=self.viscous_wall_flux,
            boundary_temperature_func=self.temperature_bc,
            boundary_gradient_cv_func=self.grad_cv_bc
        )

    def isothermal_wall_state(self, discr, btag, gas_model, state_minus, **kwargs):
        """Return state with 0 velocities and energy(Twall)."""
        temperature_wall = self._wall_temp + 0*state_minus.mass_density
        mom_plus = state_minus.mass_density*0.*state_minus.velocity
        #mom_plus = -state_minus.momentum_density
        mass_frac_plus = state_minus.species_mass_fractions

        internal_energy_plus = gas_model.eos.get_internal_energy(
            temperature=temperature_wall, species_mass_fractions=mass_frac_plus)

        total_energy_plus = state_minus.mass_density*internal_energy_plus

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density, energy=total_energy_plus,
            momentum=mom_plus, species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def inviscid_wall_flux(self, discr, btag, gas_model, state_minus,
            numerical_flux_func=inviscid_flux_rusanov, **kwargs):
        """Return Riemann flux using state with mom opposite of interior state."""
        wall_cv = make_conserved(dim=state_minus.dim,
                                 mass=state_minus.mass_density,
                                 momentum=-state_minus.momentum_density,
                                 energy=state_minus.energy_density,
                                 species_mass=state_minus.species_mass_density)
        wall_state = make_fluid_state(cv=wall_cv, gas_model=gas_model,
                                      temperature_seed=state_minus.temperature)
        state_pair = TracePair(btag, interior=state_minus, exterior=wall_state)

        from mirgecom.inviscid import inviscid_facial_flux
        return self._boundary_quantity(
            discr, btag,
            inviscid_facial_flux(discr, gas_model=gas_model, state_pair=state_pair,
                                 numerical_flux_func=numerical_flux_func,
                                 local=True),
            **kwargs)

    def temperature_bc(self, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        # return 2*self._wall_temp - state_minus.temperature
        return 0.*state_minus.temperature + self._wall_temp

    def grad_cv_bc(self, state_minus, grad_cv_minus, normal, **kwargs):
        from mirgecom.fluid import species_mass_fraction_gradient
        grad_y_plus = species_mass_fraction_gradient(state_minus.cv, grad_cv_minus)
        for i in range(state_minus.nspecies):
            grad_y_plus[i] = grad_y_plus[i] - (np.dot(grad_y_plus[i], normal)*normal)
        grad_cv = make_conserved(grad_cv_minus.dim,
                                 mass=grad_cv_minus.mass,
                                 energy=grad_cv_minus.energy,
                                 momentum=grad_cv_minus.momentum,
                                 species_mass=state_minus.mass_density*grad_y_plus)
        return grad_cv

    def viscous_wall_flux(self, discr, btag, gas_model, state_minus,
                                           grad_cv_minus, grad_t_minus,
                                           numerical_flux_func=viscous_flux_central,
                                           **kwargs):
        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = thaw(discr.normal(btag), actx)

        state_plus = self.isothermal_wall_state(discr=discr, btag=btag,
                                                gas_model=gas_model,
                                                state_minus=state_minus, **kwargs)
        grad_cv_plus = self.grad_cv_bc(state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       normal=normal, **kwargs)

        grad_t_plus = self._bnd_grad_temperature_func(
            discr=discr, btag=btag, gas_model=gas_model,
            state_minus=state_minus, grad_cv_minus=grad_cv_minus,
            grad_t_minus=grad_t_minus)

        f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                             grad_t=grad_t_plus)

        return self._boundary_quantity(
            discr, btag,
            quantity=f_ext@normal)


def get_mesh(use_gmsh=True):
    """Get the mesh."""
    left_boundary_loc = -0.1
    right_boundary_loc = 0.1
    bottom_boundary_loc = 0.0
    top_boundary_loc = 0.02
    if use_gmsh:
        from meshmode.mesh.io import (
            generate_gmsh,
            ScriptSource
        )

        # for 2D, the line segments/surfaces need to be specified clockwise to
        # get the correct facing (right-handed) surface normals
        my_string = (f"""
                size=0.001;
                Point(1) = {{ {left_boundary_loc},  {bottom_boundary_loc}, 0, size}};
                Point(2) = {{ {right_boundary_loc}, {bottom_boundary_loc}, 0, size}};
                Point(3) = {{ {right_boundary_loc}, {top_boundary_loc},    0, size}};
                Point(4) = {{ {left_boundary_loc},  {top_boundary_loc},    0, size}};
                Line(1) = {{1, 2}};
                Line(2) = {{2, 3}};
                Line(3) = {{3, 4}};
                Line(4) = {{4, 1}};
                Line Loop(1) = {{-4, -3, -2, -1}};
                Plane Surface(1) = {{1}};
                Physical Surface('domain') = {{1}};
                Physical Curve('Bottom') = {{1}};
                Physical Curve('Right') = {{2}};
                Physical Curve('Top') = {{3}};
                Physical Curve('Left') = {{4}};

                // Create distance field from curves, excludes cavity
                Field[1] = Distance;
                Field[1].CurvesList = {{3,4}};
                Field[1].NumPointsPerCurve = 100000;

                //Create threshold field that varrries element size near boundaries
                Field[2] = Threshold;
                Field[2].InField = 1;
                Field[2].SizeMin = size / 3;
                Field[2].SizeMax = size;
                Field[2].DistMin = 0.0002;
                Field[2].DistMax = 0.005;
                Field[2].StopAtDistMax = 1;

                //  background mesh size
                Field[3] = Box;
                Field[3].XMin = 0.;
                Field[3].XMax = 1.0;
                Field[3].YMin = -1.0;
                Field[3].YMax = 1.0;
                Field[3].VIn = size;
                Field[3].VOut = 2 * size

                // take the minimum of all defined meshing fields
                Field[100] = Min;
                Field[100].FieldsList = {{2, 3}};
                Background Field = 100;

                Mesh.MeshSizeExtendFromBoundary = 0;
                Mesh.MeshSizeFromPoints = 0;
                Mesh.MeshSizeFromCurvature = 0;
        """)

        #print(my_string)
        generate_mesh = partial(generate_gmsh, ScriptSource(my_string, "geo"),
                                force_ambient_dim=2, dimensions=2, target_unit="M")
    else:
        char_len_x = 0.002
        char_len_y = 0.002
        box_ll = (left_boundary_loc, bottom_boundary_loc)
        box_ur = (right_boundary_loc, top_boundary_loc)
        num_elements = (int((box_ur[0]-box_ll[0])/char_len_x),
                            int((box_ur[1]-box_ll[1])/char_len_y))

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh, a=box_ll, b=box_ur,
                                n=num_elements,
                                boundary_tag_to_face={
                                    "Left": ["-x"],
                                    "Right": ["+x"],
                                    "Top": ["+y"],
                                    "Bottom": ["-y"]}
                                )

    return generate_mesh


class InitBox:
    r"""Solution initializer

    create a uniform (no-flow) field with a linear variation in species mass fraction
    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, nspecies=0,
            pressure, temperature, mass_frac=None
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        P0: float
            stagnation pressure
        T0: float
            stagnation temperature
        """

        if mass_frac is None:
            if nspecies > 0:
                mass_frac = np.zeros(shape=(nspecies,))

        self._dim = dim
        self._nspecies = nspecies
        self._pressure = pressure
        self._temperature = temperature
        self._mass_frac = mass_frac

    def __call__(self, discr, x_vec, eos, *, time=0.0):
        """Create the solution state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        xpos = x_vec[0]
        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]
        actx = xpos.array_context
        zeros = 0*xpos
        ones = zeros + 1.0


        pressure = ones*self._pressure
        temperature = ones*self._temperature

        y = ones*self._mass_frac

        # mass fraction is a linear function of x
        x0 = -0.1
        x1 = 0.1
        y0 = [0.4, 0.6]
        y1 = [0.8, 0.2]
        for i in range(self._nspecies):
            y[i] = y0[i] + (xpos - x0)*(y1[i] - y0[i])/(x1 - x0)

        #make rho a linear function in x
        temp0 = 1000
        temp1 = 300
        temperature = temp0 + (xpos - x0)*(temp1 - temp0)/(x1 - x0)
        mass = eos.get_density(pressure=pressure, temperature=temperature,
                               species_mass_fractions=y)

        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        velocity = ones*np.zeros(self._dim, dtype=object)
        mom = mass*velocity

        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, restart_filename=None,
         use_profiling=False, use_logmgr=True, user_input_file=None,
         use_overintegration=False, actx_class=None, lazy=False, casename=None):
    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    log_path = "log_data/"
    logname = log_path + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # default i/o junk frequencies
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 1

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-8
    t_final = 1e-7
    current_t = 0
    current_step = 0
    current_cfl = 0.5
    constant_cfl = False

    # default health status bounds
    health_pres_min = 1.0e-1
    health_pres_max = 2.0e6
    health_temp_min = 1.0
    health_temp_max = 4000
    health_mass_frac_min = -10
    health_mass_frac_max = 10

    # discretization and model control
    order = 1
    dim = 2

    # material properties
    mu = 1.0e-5
    spec_diff = 1e-4
    mu_override = False  # optionally read in from input
    nspecies = 2

    # ACTII flow properties
    total_pres = 101325
    total_temp = 300

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        input_data = comm.bcast(input_data, root=0)
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            mu_input = float(input_data["mu"])
            mu_override = True
        except KeyError:
            pass
        try:
            spec_diff = float(input_data["spec_diff"])
        except KeyError:
            pass
        try:
            nspecies = int(input_data["nspecies"])
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            dim = int(input_data["dimen"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            health_pres_min = float(input_data["health_pres_min"])
        except KeyError:
            pass
        try:
            health_pres_max = float(input_data["health_pres_max"])
        except KeyError:
            pass
        try:
            health_temp_min = float(input_data["health_temp_min"])
        except KeyError:
            pass
        try:
            health_temp_max = float(input_data["health_temp_max"])
        except KeyError:
            pass
        try:
            health_mass_frac_min = float(input_data["health_mass_frac_min"])
        except KeyError:
            pass
        try:
            health_mass_frac_max = float(input_data["health_mass_frac_max"])
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print(f"\tTime integration {integrator}")
        print("#### Simluation control data: ####\n")

    if rank == 0:
        print("\n#### Simluation setup data: ####")
        print(f"\ttotal_pres = {total_pres}")
        print(f"\ttotal_temp = {total_temp}")
        print("#### Simluation setup data: ####")

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step

    # }}}
    # working gas: O2/N2 #
    #   O2 mass fraction 0.273
    #   gamma = 1.4
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma = 1.4
    mw_o2 = 15.999*2
    mw_n2 = 14.0067*2
    mf_o2 = 0.273
    # viscosity @ 400C, Pa-s
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu_mix = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    r = 8314.59/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75

    if mu_override:
        mu = mu_input
    else:
        mu = mu_mix

    # thermal conductivity
    kappa = cp*mu/Pr
    init_temperature = 300.0

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")
        print(f"\tnspecies = {nspecies}")
        print(f"\tspecies diffusivity = {spec_diff}")

    #spec_diffusivity = 0. * np.ones(nspecies)
    spec_diffusivity = spec_diff * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

    # initialize eos and species mass fractions
    y = np.zeros(nspecies)
    y_fuel = np.zeros(nspecies)
    y[0] = 1
    y_fuel[1] = 1
    species_names = ["air", "fuel"]

    eos = IdealSingleGas(gamma=gamma, gas_const=r)
    gas_model = GasModel(eos=eos, transport=transport_model)

    pressure = total_pres
    temperature = total_temp

    bulk_init = InitBox(dim=dim, pressure=pressure, temperature=temperature,
                        nspecies=nspecies)

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:06d}-{rank:04d}.pkl"
    )
    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])
        # will use this later
        #restart_nspecies = int(restart_data["nspecies"])

        assert restart_data["num_parts"] == nparts
        assert restart_data["nspecies"] == nspecies
    else:
        local_mesh, global_nelements = generate_and_distribute_mesh(
            comm, get_mesh(use_gmsh=False))
        local_nelements = local_mesh.nelements

    if rank == 0:
        logging.info("Making discretization")

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
          default_simplex_group_factory, QuadratureSimplexGroupFactory

    discr = EagerDGDiscretization(
        actx, local_mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=local_mesh.dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1)
        },
        mpi_communicator=comm
    )

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    if rank == 0:
        logging.info("Done making discretization")

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "step walltime: {value:6g} s")
            #("t_log.max", "log walltime: {value:6g} s")
        ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    if rank == 0:
        logging.info("Before restart/init")

    def get_fluid_state(cv, temperature_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature_seed)

    create_fluid_state = actx.compile(get_fluid_state)

    if restart_filename:
        if rank == 0:
            logging.info("Restarting soln.")
        current_cv = restart_data["cv"]
        temperature_seed = restart_data["temperature_seed"]
        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol")
            )
            current_cv = connection(restart_data["cv"])
            temperature_seed = connection(restart_data["temperature_seed"])

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        current_cv = bulk_init(discr=discr, x_vec=thaw(discr.nodes(), actx),
                               eos=eos, time=0)
        temperature_seed = init_temperature

    current_state = create_fluid_state(current_cv, temperature_seed)
    temperature_seed = current_state.temperature

    # eventually we want to read in a reference (target) state on a restart
    target_cv = bulk_init(discr=discr, x_vec=thaw(discr.nodes(), actx),
                                 eos=eos, time=0)
    target_state = create_fluid_state(target_cv, init_temperature)

    # set the boundary conditions
    def _ref_state_func(discr, btag, gas_model, ref_state, **kwargs):
        from mirgecom.gas_model import project_fluid_state
        from grudge.dof_desc import DOFDesc, as_dofdesc
        dd_base_vol = DOFDesc("vol")
        return project_fluid_state(discr, dd_base_vol,
                                   as_dofdesc(btag).with_discr_tag(quadrature_tag),
                                   ref_state, gas_model)

    _ref_boundary_state_func = partial(_ref_state_func, ref_state=target_state)

    ref_state = PrescribedFluidBoundary(boundary_state_func=_ref_boundary_state_func)
    isothermal_wall = IsothermalWallBoundary()
    dummy = DummyBoundary()
    #adiabatic_wall = AdiabaticSlipBoundary()

    boundaries = {
        DTAG_BOUNDARY("Left"): ref_state,
        DTAG_BOUNDARY("Right"): isothermal_wall,
        DTAG_BOUNDARY("Top"): dummy,
        DTAG_BOUNDARY("Bottom"): dummy
    }

    visualizer = make_visualizer(discr)

    #    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    # some utility functions
    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(discr, "vol", x))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(discr, "vol", x))[()]

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(discr, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(discr, "vol", x))[()]

    def my_write_status(cv, dv, dt, cfl):
        status_msg = f"-------- dt = {dt:1.3e}, cfl = {cfl:1.4f}"
        temperature = thaw(freeze(dv.temperature, actx), actx)
        pressure = thaw(freeze(dv.pressure, actx), actx)
        p_min = vol_min(pressure)
        p_max = vol_max(pressure)
        t_min = vol_min(temperature)
        t_max = vol_max(temperature)

        from pytools.obj_array import obj_array_vectorize
        y_min = obj_array_vectorize(lambda x: vol_min(x),
                                      cv.species_mass_fractions)
        y_max = obj_array_vectorize(lambda x: vol_max(x),
                                      cv.species_mass_fractions)

        dv_status_msg = (
            f"\n-------- P (min, max) (Pa) = ({p_min:1.9e}, {p_max:1.9e})")
        dv_status_msg += (
            f"\n-------- T (min, max) (K)  = ({t_min:7g}, {t_max:7g})")
        for i in range(nspecies):
            dv_status_msg += (
                f"\n-------- y_{species_names[i]} (min, max) = "
                f"({y_min[i]:1.3e}, {y_max[i]:1.3e})")
        status_msg += dv_status_msg
        status_msg += "\n"

        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, cv, dv, ts_field,
                     rhs=None, grad_cv=None, grad_t=None, grad_v=None,
                     grad_y=None):
        mach = cv.speed / dv.speed_of_sound
        viz_fields = [("cv", cv),
                      ("dv", dv),
                      ("mach", mach),
                      ("velocity", cv.velocity),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], cv.species_mass_fractions[i])
            for i in range(nspecies))
        if rhs is not None:
            viz_ext = [("rhs", rhs), ("grad_temperature", grad_t),
                       ("grad_v_x", grad_v[0]), ("grad_v_y", grad_v[1])]
            viz_ext.extend(("grad_Y_"+species_names[i], grad_y[i])
                           for i in range(nspecies))
            from mirgecom.inviscid import inviscid_flux
            fluid_state = create_fluid_state(cv=cv, temperature_seed=init_temperature)
            inv_flux = inviscid_flux(state=fluid_state)
            viz_ext2 = [("inviscid_flux_mass", inv_flux.mass)]
            viz_ext2.extend(("inviscid_flux_"+species_names[i],
                             inv_flux.species_mass[i]) for i in range(nspecies))
            viz_fields.extend(viz_ext2)
            from mirgecom.viscous import viscous_flux
            vis_flux = viscous_flux(state=fluid_state, grad_cv=grad_cv, grad_t=grad_t)
            viz_ext3 = [("viscous_flux_mass", vis_flux.mass)]
            viz_ext3.extend(("viscous_flux_"+species_names[i],
                             vis_flux.species_mass[i]) for i in range(nspecies))
            viz_fields.extend(viz_ext3)

        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, temperature_seed):
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "temperature_seed": temperature_seed,
                "nspecies": nspecies,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

    def my_health_check(cv, dv):
        health_error = False
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(discr, "vol", dv.pressure,
                                     health_pres_min, health_pres_max),
                                     op="lor"):
            health_error = True
            p_min = vol_min(dv.pressure)
            p_max = vol_max(dv.pressure)
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        if global_reduce(check_range_local(discr, "vol", dv.temperature,
                                     health_temp_min, health_temp_max),
                                     op="lor"):
            health_error = True
            t_min = vol_min(dv.temperature)
            t_max = vol_max(dv.temperature)
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        for i in range(nspecies):
            if global_reduce(check_range_local(discr, "vol",
                                               cv.species_mass_fractions[i],
                                         health_mass_frac_min, health_mass_frac_max),
                                         op="lor"):
                health_error = True
                y_min = vol_min(cv.species_mass_fractions[i])
                y_max = vol_max(cv.species_mass_fractions[i])
                logger.info(f"Species mass fraction range violation. "
                            f"{species_names[i]}: ({y_min=}, {y_max=})")

        return health_error

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            from mirgecom.viscous import get_viscous_timestep
            ts_field = current_cfl * get_viscous_timestep(discr, state=state)
            from grudge.op import nodal_min
            dt = actx.to_numpy(nodal_min(discr, "vol", ts_field))
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            ts_field = get_viscous_cfl(discr, dt=dt, state=state)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(discr, "vol", ts_field))

        return ts_field, cfl, min(t_remaining, dt)

    def my_pre_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = create_fluid_state(cv=cv, temperature_seed=tseed)
        dv = fluid_state.dv

        try:
            if logmgr:
                logmgr.tick_before()

            ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.warning("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_status:
                my_write_status(dt=dt, cfl=cfl, dv=dv, cv=cv)

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv, temperature_seed=tseed)

            if do_viz:
                from mirgecom.fluid import (
                    velocity_gradient,
                    species_mass_fraction_gradient
                )
                ns_rhs, grad_cv, grad_t = \
                    ns_operator(discr, state=fluid_state, time=t,
                                boundaries=boundaries, gas_model=gas_model,
                                return_gradients=True)
                grad_v = velocity_gradient(cv, grad_cv)
                grad_y = species_mass_fraction_gradient(cv, grad_cv)
                my_write_viz(step=step, t=t, cv=cv, dv=dv,
                             ts_field=ts_field,
                             rhs=ns_rhs, grad_cv=grad_cv, grad_t=grad_t,
                             grad_v=grad_v, grad_y=grad_y)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, cv=cv, dv=dv, ts_field=ts_field)
            my_write_restart(step=step, t=t, cv=cv, temperature_seed=tseed)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = create_fluid_state(cv=cv, temperature_seed=tseed)
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, cv, gas_model.eos)
            logmgr.tick_after()
        return make_obj_array([fluid_state.cv, fluid_state.temperature]), dt

    def my_rhs(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        cv_rhs = (
            ns_operator(discr, state=fluid_state, time=t, boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag)
        )
        return make_obj_array([cv_rhs, 0*tseed])

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt,
                      t=current_t, t_final=t_final,
                      state=make_obj_array([current_state.cv, temperature_seed]))
    current_cv, tseed = stepper_state
    current_state = make_fluid_state(current_cv, gas_model, tseed)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = current_state.dv
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt,
                                        state=current_state)
    my_write_status(dt=dt, cfl=cfl, cv=current_state.cv, dv=final_dv)

    from mirgecom.fluid import (
        velocity_gradient,
        species_mass_fraction_gradient
    )
    ns_rhs, grad_cv, grad_t = \
        ns_operator(discr, state=current_state, time=current_t,
                    boundaries=boundaries, gas_model=gas_model,
                    return_gradients=True)
    grad_v = velocity_gradient(current_state.cv, grad_cv)
    grad_y = species_mass_fraction_gradient(current_state.cv, grad_cv)
    my_write_viz(step=current_step, t=current_t, cv=current_state.cv, dv=final_dv,
                 ts_field=ts_field,
                 rhs=ns_rhs, grad_cv=grad_cv, grad_t=grad_t,
                 grad_v=grad_v, grad_y=grad_y)

    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")

    args = parser.parse_args()
    lazy = args.lazy

    # for writing output
    casename = "species_box"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    if args.profile:
        if args.lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Ignoring user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    main(restart_filename=restart_filename, user_input_file=input_file,
         use_profiling=args.profile, use_logmgr=args.log,
         use_overintegration=args.overintegration, lazy=lazy,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
