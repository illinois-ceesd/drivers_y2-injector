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
from mirgecom.artificial_viscosity import \
    av_laplacian_operator, smoothness_indicator
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
    IsothermalWallBoundary,
    OutflowBoundary
)
import cantera
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
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


def get_mesh(dim, read_mesh=True):
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh_filename = "data/isolator.msh"
    #mesh = read_gmsh(mesh_filename, force_ambient_dim=dim)
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)
    #mesh = read_gmsh(mesh_filename)

    return mesh


def getIsentropicPressure(mach, P0, gamma):
    pressure = (1. + (gamma - 1.)*0.5*mach**2)
    pressure = P0*pressure**(-gamma / (gamma - 1.))
    return pressure


def getIsentropicTemperature(mach, T0, gamma):
    temperature = (1. + (gamma - 1.)*0.5*mach**2)
    temperature = T0/temperature
    return temperature


def getMachFromAreaRatio(area_ratio, gamma, mach_guess=0.01):
    error = 1.0e-8
    nextError = 1.0e8
    g = gamma
    M0 = mach_guess
    while nextError > error:
        R = (((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))/M0
            - area_ratio)
        dRdM = (2*((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               / (2*g - 2)*(g - 1)/(2/(g + 1) + ((g - 1)/(g + 1)*M0*M0)) -
               ((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               * M0**(-2))
        M1 = M0 - R/dRdM
        nextError = abs(R)
        M0 = M1

    return M1


class InitACTII:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, nspecies=0,
            P0, T0, temp_wall, temp_sigma, vel_sigma, gamma_guess,
            mass_frac=None,
            inj_pres, inj_temp, inj_vel, inj_mass_frac=None,
            inj_gamma_guess,
            inj_temp_sigma, inj_vel_sigma,
            inj_ytop, inj_ybottom,
            inj_mach
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
        gamma_guess: float
            guesstimate for gamma
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        geom_top: numpy.ndarray
            coordinates for the top wall
        geom_bottom: numpy.ndarray
            coordinates for the bottom wall
        """

        # check number of points in the geometry
        #top_size = geom_top.size
        #bottom_size = geom_bottom.size

        if mass_frac is None:
            if nspecies > 0:
                mass_frac = np.zeros(shape=(nspecies,))

        if inj_mass_frac is None:
            if nspecies > 0:
                inj_mass_frac = np.zeros(shape=(nspecies,))

        if inj_vel is None:
            inj_vel = np.zeros(shape=(dim,))

        self._dim = dim
        self._nspecies = nspecies
        self._P0 = P0
        self._T0 = T0
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._gamma_guess = gamma_guess
        # TODO, calculate these from the geometry files
        self._throat_height = 3.61909e-3
        self._x_throat = 0.283718298
        self._mass_frac = mass_frac

        self._inj_P0 = inj_pres
        self._inj_T0 = inj_temp
        self._inj_vel = inj_vel
        self._inj_gamma_guess = inj_gamma_guess

        self._temp_sigma_injection = inj_temp_sigma
        self._vel_sigma_injection = inj_vel_sigma
        self._inj_mass_frac = inj_mass_frac
        self._inj_ytop = inj_ytop
        self._inj_ybottom = inj_ybottom
        self._inj_mach = inj_mach

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

        # initialize the bulk to P0, T0, and quiescent
        pressure = ones*self._P0
        temperature = ones*self._T0

        y = ones*self._mass_frac
        mass = eos.get_density(pressure=pressure, temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        velocity = ones*np.zeros(self._dim, dtype=object)
        mom = mass*velocity

        # fuel stream initialization
        # initially in pressure/temperature equilibrium with the cavity
        #inj_left = 0.71
        # even with the bottom corner
        inj_left = 0.70163
        # even with the top corner
        #inj_left = 0.7074
        #inj_left = 0.65
        inj_right = 0.73
        inj_top = -0.0226
        inj_bottom = -0.025
        inj_fore = 0.035/2. + 1.59e-3
        inj_aft = 0.035/2. - 1.59e-3
        xc_left = zeros + inj_left
        xc_right = zeros + inj_right
        yc_top = zeros + inj_top
        yc_bottom = zeros + inj_bottom
        zc_fore = zeros + inj_fore
        zc_aft = zeros + inj_aft

        yc_center = zeros - 0.0283245 + 4e-3 + 1.59e-3/2.
        zc_center = zeros + 0.035/2.
        inj_radius = 1.59e-3/2.
        #inj_bl_thickness = inj_radius/3.
        inj_bl_thickness = -1000

        if self._dim == 2:
            radius = actx.np.sqrt((ypos - yc_center)**2)
        else:
            radius = actx.np.sqrt((ypos - yc_center)**2 + (zpos - zc_center)**2)

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        top_edge = actx.np.less(ypos, yc_top)
        aft_edge = ones
        fore_edge = ones
        if self._dim == 3:
            aft_edge = actx.np.greater(zpos, zc_aft)
            fore_edge = actx.np.less(zpos, zc_fore)
        inside_injector = (left_edge*right_edge*top_edge*bottom_edge *
                           aft_edge*fore_edge)

        inj_y = ones*self._inj_mass_frac

        inj_velocity = zeros*np.zeros(self._dim, dtype=object)
        inj_velocity[0] = self._inj_vel[0]

        inj_mach = self._inj_mach*ones

        # smooth out the injection profile
        # relax to the cavity temperature/pressure/velocity
        inj_x0 = 0.712
        #inj_x0 = 100
        # the entrace to the injector
        #inj_fuel_x0 = 0.7085
        #inj_fuel_x0 = 0.705
        # back inside the injector
        # behind the shock
        #inj_fuel_x0 = inj_x0 + 0.002
        # infront of the shock
        #inj_fuel_x0 = inj_x0 - 0.002
        inj_fuel_x0 = 0.712 - 0.002
        inj_sigma = 1500
        inj_sigma_y = 10000
        #gamma_guess_inj = gamma

        # left extent
        inj_tanh = inj_sigma*(inj_fuel_x0 - xpos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        for i in range(self._nspecies):
            inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        # transition the fuel from 1 at the centerline to 0 at the injector boundary
        # radial extent
        inj_tanh = inj_sigma_y*(radius - (inj_radius-inj_bl_thickness))
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        for i in range(self._nspecies):
            inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        # transition the mach number from 0 (cavity) to 1 (injection)
        inj_tanh = inj_sigma*(inj_x0 - xpos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        inj_mach = inj_weight*inj_mach

        # assume a smooth transition in gamma, could calculate it
        inj_gamma = (self._gamma_guess +
            (self._inj_gamma_guess - self._gamma_guess)*inj_weight)

        inj_pressure = getIsentropicPressure(
            mach=inj_mach,
            P0=self._inj_P0,
            gamma=inj_gamma
        )
        inj_temperature = getIsentropicTemperature(
            mach=inj_mach,
            T0=self._inj_T0,
            gamma=inj_gamma
        )

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(temperature=inj_temperature,
                                                      species_mass_fractions=inj_y)

        inj_velocity = zeros*np.zeros(self._dim, dtype=object)
        inj_mom = inj_mass*inj_velocity

        # the velocity magnitude
        inj_cv = make_conserved(dim=self._dim, mass=inj_mass, momentum=inj_mom,
                                energy=inj_energy, species_mass=inj_mass*inj_y)

        inj_velocity[0] = -inj_mach*eos.sound_speed(inj_cv, inj_temperature)

        # relax the pressure at the cavity/injector interface
        inj_pressure = pressure + (inj_pressure - pressure)*inj_weight
        inj_temperature = (temperature +
            (inj_temperature - temperature)*inj_weight)

        # we need to calculate the velocity from a prescribed mass flow rate
        # this will need to take into account the velocity relaxation at the
        # injector walls
        #inj_velocity[0] = velocity[0] + (self._inj_vel[0] - velocity[0])*inj_weight

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma_injection
        wall_temperature = self._temp_wall
        smoothing_radius = actx.np.tanh(sigma*(actx.np.abs(radius - inj_radius)))
        inj_temperature = (wall_temperature +
            (inj_temperature - wall_temperature)*smoothing_radius)

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(temperature=inj_temperature,
                                                  species_mass_fractions=inj_y)

        # modify the velocity in the near-wall region to have a tanh profile
        # this approximates the BL velocity profile
        sigma = self._vel_sigma_injection
        smoothing_radius = actx.np.tanh(sigma*(actx.np.abs(radius - inj_radius)))
        inj_velocity[0] = inj_velocity[0]*smoothing_radius

        for i in range(self._nspecies):
            y[i] = actx.np.where(inside_injector, inj_y[i], y[i])
            #y[i] = inj_y[i]

        # recompute the mass and energy (outside the injector) to account for
        # the change in mass fraction
        mass = eos.get_density(pressure=pressure,
                               temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        mass = actx.np.where(inside_injector, inj_mass, mass)
        velocity[0] = actx.np.where(inside_injector, inj_velocity[0], velocity[0])
        energy = actx.np.where(inside_injector, inj_energy, energy)

        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))
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
    # verbosity for what gets written to viz dumps, increase for more stuff
    viz_level = 1
    # control the time interval for writing viz dumps
    viz_interval_type = 0

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-8
    t_final = 1e-6
    t_viz_interval = 1.e-7
    current_t = 0
    t_start = 0
    current_step = 0
    current_cfl = 0.5
    constant_cfl = 0

    # default health status bounds
    health_pres_min = 1.0e-1
    health_pres_max = 2.0e6
    health_temp_min = 1.0
    health_temp_max = 4000
    health_mass_frac_min = -10
    health_mass_frac_max = 10

    # discretization and model control
    order = 1
    alpha_sc = 0.3
    s0_sc = -5.0
    kappa_sc = 0.5
    dim = 2

    # material properties
    mu = 1.0e-5
    spec_diff = 1e-4
    mu_override = False  # optionally read in from input
    nspecies = 0

    # ACTII flow properties
    total_pres_inflow = 5000
    total_temp_inflow = 300

    # injection flow properties
    total_pres_inj = 50400
    total_temp_inj = 300.0
    mach_inj = 1.0

    # parameters to adjust the shape of the initialization
    #vel_sigma = 2000
    #temp_sigma = 2500
    vel_sigma = 1000
    temp_sigma = 1250
    # adjusted to match the mass flow rate
    vel_sigma_inj = 5000
    temp_sigma_inj = 5000
    temp_wall = 300

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
            t_viz_interval = float(input_data["t_viz_interval"])
        except KeyError:
            pass
        try:
            viz_interval_type = int(input_data["viz_interval_type"])
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
            constant_cfl = int(input_data["constant_cfl"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            current_cfl = float(input_data["current_cfl"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            alpha_sc = float(input_data["alpha_sc"])
        except KeyError:
            pass
        try:
            kappa_sc = float(input_data["kappa_sc"])
        except KeyError:
            pass
        try:
            s0_sc = float(input_data["s0_sc"])
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
            total_pres_inj = float(input_data["total_pres_inj"])
        except KeyError:
            pass
        try:
            total_temp_inj = float(input_data["total_temp_inj"])
        except KeyError:
            pass
        try:
            mach_inj = float(input_data["mach_inj"])
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
        try:
            vel_sigma = float(input_data["vel_sigma"])
        except KeyError:
            pass
        try:
            temp_sigma = float(input_data["temp_sigma"])
        except KeyError:
            pass
        try:
            vel_sigma_inj = float(input_data["vel_sigma_inj"])
        except KeyError:
            pass
        try:
            temp_sigma_inj = float(input_data["temp_sigma_inj"])
        except KeyError:
            pass
        try:
            viz_level = int(input_data["viz_level"])
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if viz_interval_type > 2:
        error_message = "Invalid value for viz_interval_type [0-2]"
        raise RuntimeError(error_message)

    s0_sc = np.log10(1.0e-4 / np.power(order, 4))
    if rank == 0:
        print(f"Shock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if constant_cfl == 1:
            print(f"\tConstant cfl mode, current_cfl = {current_cfl}")
        else:
            print(f"\tConstant dt mode, current_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print(f"\tTime integration {integrator}")
        print("#### Simluation control data: ####")

    if rank == 0:
        print("\n#### Visualization setup: ####")
        if viz_level >= 0:
            print("\tBasic visualization output enabled.")
            print("\t(cv, dv, cfl)")
        if viz_level >= 1:
            print("\tExtra visualization output enabled for derived quantities.")
            print("\t(velocity, mass_fractions, etc.)")
        if viz_level >= 2:
            print("\tNon-dimensional parameter visualization output enabled.")
            print("\t(Re, Pr, etc.)")
        if viz_level >= 3:
            print("\tDebug visualization output enabled.")
            print("\t(rhs, grad_cv, etc.)")
        if viz_interval_type == 0:
            print(f"\tWriting viz data every {nviz} steps.")
        if viz_interval_type == 1:
            print(f"\tWriting viz data roughly every {t_viz_interval} seconds.")
        if viz_interval_type == 2:
            print(f"\tWriting viz data exactly every {t_viz_interval} seconds.")
        print("#### Visualization setup: ####")

    if rank == 0:
        print("\n#### Simluation setup data: ####")
        print(f"\ttotal_pres_injection = {total_pres_inj}")
        print(f"\ttotal_temp_injection = {total_temp_inj}")
        print(f"\tvel_sigma = {vel_sigma}")
        print(f"\ttemp_sigma = {temp_sigma}")
        print(f"\tvel_sigma_injection = {vel_sigma_inj}")
        print(f"\ttemp_sigma_injection = {temp_sigma_inj}")
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
    mf_c2h4 = 0.5
    mf_h2 = 0.5

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
        if nspecies == 0:
            print("\tno passive scalars, uniform ideal gas eos")
        elif nspecies == 2:
            print("\tpassive scalars to track air/fuel mixture, ideal gas eos")
        else:
            print("\tfull multi-species initialization with pyrometheus eos")

    #spec_diffusivity = 0. * np.ones(nspecies)
    spec_diffusivity = spec_diff * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

    #
    # isentropic expansion based on the area ratios between the inlet (r=54e-3m) and
    # the throat (r=3.167e-3)
    #
    vel_injection = np.zeros(shape=(dim,))

    # initialize eos and species mass fractions
    y = np.zeros(nspecies)
    y_fuel = np.zeros(nspecies)
    if nspecies == 2:
        y[0] = 1
        y_fuel[1] = 1
        species_names = ["air", "fuel"]
    elif nspecies > 2:
        from mirgecom.mechanisms import get_mechanism_cti
        mech_cti = get_mechanism_cti("uiuc")

        cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
        cantera_nspecies = cantera_soln.n_species
        if nspecies != cantera_nspecies:
            if rank == 0:
                print(f"specified {nspecies=}, but cantera mechanism"
                      f" needs nspecies={cantera_nspecies}")
            raise RuntimeError()

        i_c2h4 = cantera_soln.species_index("C2H4")
        i_h2 = cantera_soln.species_index("H2")
        i_ox = cantera_soln.species_index("O2")
        i_di = cantera_soln.species_index("N2")
        # Set the species mass fractions to the free-stream flow
        y[i_ox] = mf_o2
        y[i_di] = 1. - mf_o2
        # Set the species mass fractions to the free-stream flow
        y_fuel[i_c2h4] = mf_c2h4
        y_fuel[i_h2] = mf_h2

        cantera_soln.TPY = init_temperature, 101325, y

    # make the eos
    if nspecies < 3:
        eos = IdealSingleGas(gamma=gamma, gas_const=r)
    else:
        from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
        pyro_mech = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)
        eos = PyrometheusMixture(pyro_mech, temperature_guess=init_temperature)
        species_names = pyro_mech.species_names

    gas_model = GasModel(eos=eos, transport=transport_model)

    # injection mach number
    if nspecies < 3:
        gamma_inj = gamma
    else:
        gamma_inj = 0.5*(1.24 + 1.4)

    pres_injection = getIsentropicPressure(mach=mach_inj,
                                           P0=total_pres_inj,
                                           gamma=gamma_inj)
    temp_injection = getIsentropicTemperature(mach=mach_inj,
                                              T0=total_temp_inj,
                                              gamma=gamma_inj)

    if nspecies < 3:
        rho_injection = pres_injection/temp_injection/r
        sos = math.sqrt(gamma*pres_injection/rho_injection)
    else:
        cantera_soln.TPY = temp_injection, pres_injection, y_fuel
        rho_injection = cantera_soln.density
        gamma_loc = cantera_soln.cp_mass/cantera_soln.cv_mass
        sos = math.sqrt(gamma_loc*pres_injection/rho_injection)
        if rank == 0:
            print(f"injection gamma guess {gamma_inj} cantera gamma {gamma_loc}")

    vel_injection[0] = -mach_inj*sos

    if rank == 0:
        print("")
        print(f"\tinjector Mach number {mach_inj}")
        print(f"\tinjector temperature {temp_injection}")
        print(f"\tinjector pressure {pres_injection}")
        print(f"\tinjector rho {rho_injection}")
        print(f"\tinjector velocity {vel_injection[0]}")
        print("#### Simluation initialization data: ####\n")

    inj_ymin = -0.0243245
    inj_ymax = -0.0227345
    bulk_init = InitACTII(dim=dim,
                          P0=total_pres_inflow, T0=total_temp_inflow,
                          temp_wall=temp_wall, temp_sigma=temp_sigma,
                          vel_sigma=vel_sigma, nspecies=nspecies,
                          mass_frac=y, gamma_guess=gamma, inj_gamma_guess=gamma_inj,
                          inj_pres=total_pres_inj,
                          inj_temp=total_temp_inj,
                          inj_vel=vel_injection, inj_mass_frac=y_fuel,
                          inj_temp_sigma=temp_sigma_inj,
                          inj_vel_sigma=vel_sigma_inj,
                          inj_ytop=inj_ymax, inj_ybottom=inj_ymin,
                          inj_mach=mach_inj)

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
        t_start = current_t
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
            comm, get_mesh(dim=dim))
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
    outflow = OutflowBoundary(boundary_pressure=total_pres_inflow)
    wall = IsothermalWallBoundary()

    boundaries = {
        #DTAG_BOUNDARY("outflow"): ref_state,
        DTAG_BOUNDARY("outflow"): outflow,
        DTAG_BOUNDARY("injection"): ref_state,
        DTAG_BOUNDARY("wall"): wall
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

    def my_write_viz(step, t, fluid_state, ts_field, alpha_field):

        if rank == 0:
            print(f"******** Writing Visualization File at {step}, "
                  f"sim time {t:1.6e} s ********")

        cv = fluid_state.cv
        dv = fluid_state.dv

        # basic viz quantities, things here are difficult (or impossible) to compute
        # in post-processing
        viz_fields = [("cv", cv),
                      ("dv", dv),
                      ("dt" if constant_cfl else "cfl", ts_field)]

        # extra viz quantities, things here are often used for post-processing
        if viz_level > 0:
            mach = cv.speed / dv.speed_of_sound
            tagged_cells = smoothness_indicator(discr, cv.mass, s0=s0_sc,
                                                kappa=kappa_sc)
            viz_ext = [("mach", mach),
                       ("velocity", cv.velocity),
                       ("alpha", alpha_field),
                       ("tagged_cells", tagged_cells)]
            viz_fields.extend(viz_ext)
            # species mass fractions
            viz_fields.extend(
                ("Y_"+species_names[i], cv.species_mass_fractions[i])
                for i in range(nspecies))

        # additional viz quantities, add in some non-dimensional numbers
        if viz_level > 1:
            from grudge.dt_utils import characteristic_lengthscales
            char_length = characteristic_lengthscales(cv.array_context, discr)
            cell_Re = cv.mass*cv.speed*char_length/fluid_state.viscosity
            cp = gas_model.eos.heat_capacity_cp(cv, fluid_state.temperature)
            alpha_heat = fluid_state.thermal_conductivity/cp/fluid_state.viscosity
            cell_Pe_heat = char_length*cv.speed/alpha_heat
            from mirgecom.viscous import get_local_max_species_diffusivity
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )
            cell_Pe_mass = char_length*cv.speed/d_alpha_max
            # these are useful if our transport properties
            # are not constant on the mesh
            # prandtl
            # schmidt_number
            # damkohler_number

            viz_ext = [("Re", cell_Re),
                       ("Pe_mass", cell_Pe_mass),
                       ("Pe_heat", cell_Pe_heat)]
            viz_fields.extend(viz_ext)
        # debbuging viz quantities, things here are used for diagnosing run issues
        if viz_level > 2:
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

            viz_ext = [("rhs", ns_rhs), ("grad_temperature", grad_t),
                       ("grad_v_x", grad_v[0]), ("grad_v_y", grad_v[1])]
            viz_ext.extend(("grad_Y_"+species_names[i], grad_y[i])
                           for i in range(nspecies))
            viz_fields.extend(viz_ext)

        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

        if rank == 0:
            print("******** Done Writing Visualization File ********\n")

    def my_write_restart(step, t, cv, temperature_seed):
        if rank == 0:
            print(f"******** Writing Restart File at {step=}, "
                  f"sim time {t:1.6e} s ********")

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

        if rank == 0:
            print("******** Done Writing Restart File ********\n")

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

    def my_get_viscous_timestep(discr, state, alpha):
        """Routine returns the the node-local maximum stable viscous timestep.

        Parameters
        ----------
        discr: grudge.eager.EagerDGDiscretization
            the discretization to use
        state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """
        from grudge.dt_utils import characteristic_lengthscales

        length_scales = characteristic_lengthscales(state.array_context, discr)

        nu = 0
        d_alpha_max = 0

        if state.is_viscous:
            nu = state.viscosity/state.mass_density
            # this appears to break lazy for whatever reason
            from mirgecom.viscous import get_local_max_species_diffusivity
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    state.array_context,
                    state.species_diffusivity
                )

        return(
            length_scales / (state.wavespeed
            + ((nu + d_alpha_max + alpha) / length_scales))
        )

    def my_get_viscous_cfl(discr, dt, state, alpha):
        """Calculate and return node-local CFL based on current state and timestep.

        Parameters
        ----------
        discr: :class:`grudge.eager.EagerDGDiscretization`
            the discretization to use
        dt: float or :class:`~meshmode.dof_array.DOFArray`
            A constant scalar dt or node-local dt
        state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The CFL at each node.
        """
        return dt / my_get_viscous_timestep(discr, state=state, alpha=alpha)

    def my_get_timestep(t, dt, state, alpha):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            ts_field = current_cfl * my_get_viscous_timestep(discr, state=state,
                                                             alpha=alpha)
            from grudge.op import nodal_min
            dt = actx.to_numpy(nodal_min(discr, "vol", ts_field))
            cfl = current_cfl
        else:
            ts_field = my_get_viscous_cfl(discr, dt=dt, state=state, alpha=alpha)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(discr, "vol", ts_field))

        return ts_field, cfl, min(t_remaining, dt)

    def my_get_alpha(discr, state, alpha):
        """ Scale alpha by the element characteristic length """
        from grudge.dt_utils import characteristic_lengthscales
        array_context = state.array_context
        length_scales = characteristic_lengthscales(array_context, discr)

        #from mirgecom.fluid import compute_wavespeed
        #wavespeed = compute_wavespeed(eos, state)

        vmag = array_context.np.sqrt(np.dot(state.velocity, state.velocity))
        #alpha_field = alpha*wavespeed*length_scales
        alpha_field = alpha*vmag*length_scales
        #alpha_field = wavespeed*0 + alpha*current_step
        #alpha_field = state.mass

        return alpha_field

    def check_time(time, cfl, dt, interval, interval_type):
        toler = 1.e-6
        status = False

        dumps_so_far = math.floor((time-t_start)/interval)

        # dump if we just passed a dump interval
        if interval_type == 2:
            time_till_next = (dumps_so_far + 1)*interval - time
            steps_till_next = math.floor(time_till_next/dt)

            # reduce the timestep going into a dump to avoid a big variation in dt
            dt_new = dt
            if steps_till_next < 5:
                extra_time = time_till_next - steps_till_next*dt
                if abs(extra_time/dt) > toler:
                    dt_new = time_till_next/(steps_till_next + 1)

            if steps_till_next < 1:
                dt_new = time_till_next

            # adjust cfl and dt accordingly
            cfl = cfl/dt
            dt = dt_new
            cfl = cfl*dt

            time_from_last = time - t_start - (dumps_so_far)*interval
            if abs(time_from_last/dt) < toler:
                status = True
        else:
            time_from_last = time - t_start - (dumps_so_far)*interval
            if time_from_last < dt:
                status = True

        return status, cfl, dt

    def my_pre_step(step, t, dt, state):

        cv, tseed = state
        fluid_state = create_fluid_state(cv=cv, temperature_seed=tseed)
        dv = fluid_state.dv

        try:
            if logmgr:
                logmgr.tick_before()

            alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
            dt_last = dt
            ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state, alpha_field)

            if viz_interval_type == 1:
                do_viz, cfl, dt = check_time(time=t, cfl=cfl, dt=dt_last,
                                             interval=t_viz_interval,
                                             interval_type=viz_interval_type)
            elif viz_interval_type == 2:
                do_viz, cfl, dt = check_time(time=t, cfl=cfl, dt=dt,
                                             interval=t_viz_interval,
                                             interval_type=viz_interval_type)
            else:
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
                my_write_viz(step=step, t=t, fluid_state=fluid_state,
                             ts_field=ts_field, alpha_field=alpha_field)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, fluid_state=fluid_state,
                         ts_field=ts_field, alpha_field=alpha_field)
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

    def my_rhs_without_combustion(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
        cv_rhs = (
            ns_operator(discr, state=fluid_state, time=t, boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag)
            + av_laplacian_operator(discr, fluid_state=fluid_state,
                                    boundaries=boundaries,
                                    boundary_kwargs={"time": t,
                                                     "gas_model": gas_model},
                                    alpha=alpha_field, s0=s0_sc, kappa=kappa_sc,
                                    quadrature_tag=quadrature_tag)
        )
        return make_obj_array([cv_rhs, 0*tseed])

    def my_rhs_with_combustion(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
        #from mirgecom.inviscid import inviscid_flux_hll
        cv_rhs = (
            ns_operator(discr, state=fluid_state, time=t, boundaries=boundaries,
                        #inviscid_numerical_flux_func=inviscid_flux_hll,
                        gas_model=gas_model, quadrature_tag=quadrature_tag)
            + eos.get_species_source_terms(cv,
                                           temperature=fluid_state.temperature)
            + av_laplacian_operator(discr, fluid_state=fluid_state,
                                    boundaries=boundaries,
                                    boundary_kwargs={"time": t,
                                                     "gas_model": gas_model},
                                    alpha=alpha_field, s0=s0_sc, kappa=kappa_sc,
                                    quadrature_tag=quadrature_tag)
        )
        return make_obj_array([cv_rhs, 0*tseed])

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    my_rhs = my_rhs_without_combustion
    if nspecies > 2:
        my_rhs = my_rhs_with_combustion

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
    alpha_field = my_get_alpha(discr, current_state, alpha_sc)
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt,
                                        state=current_state, alpha=alpha_field)
    my_write_status(dt=dt, cfl=cfl, cv=current_state.cv, dv=final_dv)

    dump_step = current_step
    my_write_viz(step=dump_step, t=current_t, fluid_state=current_state,
                 ts_field=ts_field, alpha_field=alpha_field)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())


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
    casename = "isolator"
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
