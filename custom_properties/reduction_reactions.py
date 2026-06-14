
"""
Multi stage reduction reaction property package for iron.

Reactions:

(1) 3Fe2O3 + H2  ->  2Fe3O4 + H2O            irreversible, Step I
(2) Fe3O4  + H2  <-> 3FeO   + H2O            reversible, Step II, T > 860 K
(3) FeO    + H2  <-> Fe     + H2O            reversible, Step III, T > 860 K
(4) 1/4Fe3O4 + H2 <-> 3/4Fe  + H2O           reversible, lumped, T < 860 K


References:
A. Abad et al., CORAL Deliverable D3.1 (2025) 

"""


from pyomo.environ import (
    Constraint,
    exp,
    log,
    Param,
    Reals,
    Set,
    sqrt,
    value,
    Var,
    units as pyunits,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.common.config import ConfigBlock, ConfigValue, Bool

from idaes.core import (
    declare_process_block_class,
    MaterialFlowBasis,
    ReactionParameterBlock,
    ReactionBlockDataBase,
    ReactionBlockBase,
)
from idaes.core.util.misc import add_object_reference
from idaes.core.util.initialization import (
    fix_state_vars,
    revert_state_vars,
    solve_indexed_blocks,
)
from idaes.core.util.model_statistics import (
    number_unfixed_variables_in_activated_equalities,
)
from idaes.core.util.config import (
    is_state_block,
    is_physical_parameter_block,
    is_reaction_parameter_block,
)
from idaes.core.util.constants import Constants
import idaes.logger as idaeslog
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver

__author__ = "Custom reduction package based on IDAES by Chinedu Okoli and CORAL Deliverable D3.1 "

_log = idaeslog.getLogger(__name__)

# Reactant stoichiometric coefficients
reactant_abs = {"R1": 3.0, "R2": 1.0, "R3": 1.0, "R4": 0.25}

# Smoothing parameter for Avrami
smooth_avr = 1e-8

# Reaction steps
rxn_step = {"R1": "I", "R2": "II", "R3": "III", "R4": "II"}
step_rxn={v: k for k, v in rxn_step.items()}


# Reactants of the reactions
reactants = {"R1": "Fe2O3", "R2": "Fe3O4", "R3": "FeO", "R4": "Fe3O4"}




@declare_process_block_class("ReductionReactionParameterBlock")
class ReactionParameterData(ReactionParameterBlock):


    CONFIG = ConfigBlock()
    CONFIG.declare(
        "gas_property_package",
        ConfigValue(
            description="Reference to associated PropertyPackageParameter "
            "object for the gas phase.",
            domain=is_physical_parameter_block,
        ),
    )
    CONFIG.declare(
        "solid_property_package",
        ConfigValue(
            description="Reference to associated PropertyPackageParameter "
            "object for the solid phase.",
            domain=is_physical_parameter_block,
        ),
    )
    CONFIG.declare(
        "default_arguments",
        ConfigBlock(
            description="Default arguments to use with Property Package", implicit=True
        ),
    )

    def build(self):
        super(ReactionParameterBlock, self).build()
        self._reaction_block_class = ReductionReactionBlock

        self.rate_reaction_idx = Set(initialize=["R1", "R2", "R3", "R4"])
        self.reversible_rxn_idx = Set(initialize=["R2", "R3", "R4"])
        self.step_idx = Set(initialize=["I", "II", "III"])

        self.rate_reaction_stoichiometry = {

            # R1: 3Fe2O3 + H2 -> 2Fe3O4 + H2O

            ("R1", "Vap", "O2"): 0, 
            ("R1", "Vap", "CO2"): 0, 
            ("R1", "Vap", "H2"): -1,
            ("R1", "Sol", "Fe2O3"): -3, 
            ("R1", "Sol", "FeO"): 0, 
            ("R1", "Sol", "Al2O3"): 0,
            ("R1", "Vap", "N2"): 0,
            ("R1", "Vap", "H2O"): 1,
            ("R1", "Sol", "Fe3O4"): 2,
            ("R1", "Sol", "Fe"): 0,


            # R2: Fe3O4 + H2 <-> 3FeO + H2O

            ("R2", "Vap", "O2"): 0, 
            ("R2", "Vap", "CO2"): 0, 
            ("R2", "Vap", "H2"): -1,
            ("R2", "Sol", "Fe2O3"): 0, 
            ("R2", "Sol", "FeO"): 3, 
            ("R2", "Sol", "Al2O3"): 0,
            ("R2", "Vap", "N2"): 0,
            ("R2", "Vap", "H2O"): 1,
            ("R2", "Sol", "Fe3O4"): -1,
            ("R2", "Sol", "Fe"): 0,


            # R3: FeO + H2 <-> Fe + H2O

            ("R3", "Vap", "O2"): 0, 
            ("R3", "Vap", "CO2"): 0, 
            ("R3", "Vap", "H2"): -1,
            ("R3", "Sol", "Fe2O3"): 0, 
            ("R3", "Sol", "FeO"): -1, 
            ("R3", "Sol", "Al2O3"): 0,
            ("R3", "Vap", "N2"): 0,
            ("R3", "Vap", "H2O"): 1,
            ("R3", "Sol", "Fe3O4"): 0,
            ("R3", "Sol", "Fe"): 1,


            # R4: 0.25Fe3O4 + H2 <-> 0.75Fe + H2O

            ("R4", "Vap", "O2"): 0, 
            ("R4", "Vap", "CO2"): 0, 
            ("R4", "Vap", "H2"): -1,
            ("R4", "Sol", "Fe2O3"): 0, 
            ("R4", "Sol", "FeO"): 0, 
            ("R4", "Sol", "Al2O3"): 0,
            ("R4", "Vap", "N2"): 0,
            ("R4", "Vap", "H2O"): 1,
            ("R4", "Sol", "Fe3O4"): -0.25,
            ("R4", "Sol", "Fe"): 0.75,


        }


        # Standard Heat of Reaction - J/mol_rxn- ref: NIST webbook 
        # R1: dH = 2(-1120.894) + 1(-241.8264) -1(0) -3(-825.5032) = -7.1048e3 
        # R2: dH = 3(-272.04) + 1(-241.8264) - 1(-1120.894) - 1(0) = +62.94766e3 
        # R3: dH = 1(0) + 1(-241.8264) - 1(-272.04) - 1(0) = +30.2136e3 
        # R4: dH = 0.75(0) + 1(-241.8264) - 0.25(-1120.894) - 1(0) = +38.3971e3 

        dh_rxn_dict = {
            "R1": -7104.8, 
            "R2": 62947.6, 
            "R3": 30213.6, 
            "R4": 38397.1,
        }
        self.dh_rxn = Param(
            self.rate_reaction_idx, initialize=dh_rxn_dict,
            doc="Standard heat of reaction [J/mol]",
            units=pyunits.J / pyunits.mol,
        )

        self.eps = Param(
            mutable=True, default=1e-8,
            doc="Smoothing factor",
            units=pyunits.mol / pyunits.m**3,
        )

        self.k0_rxn = Param(
            self.rate_reaction_idx,
            initialize={"R1": 0.58, "R2": 1.35, "R3": 1.35, "R4": 1.35},
            doc="Pre-exponential factor [m3/mol/s]",
            units=pyunits.m**3 / pyunits.mol / pyunits.s, mutable=True,
        )

        self.energy_activation = Param(
            self.rate_reaction_idx,
            initialize={"R1": 35.6e3, "R2": 49.2e3, "R3": 49.2e3, "R4": 49.2e3},
            doc="Activation energy [J/mol]",
            units=pyunits.J / pyunits.mol,
        )

        self.rxn_order = Param(
            self.rate_reaction_idx,
            initialize={"R1": 1.1, "R2": 1.1, "R3": 1.1, "R4": 1.1},
            doc="Reaction order for H2 [-]",
            units=pyunits.dimensionless,
        )

        self.rxn_steam_order = Param(
            self.rate_reaction_idx,
            initialize={"R1": 0.0, "R2": 0.3, "R3": 2.4, "R4": 2.2},
            doc="Steam driving force order [-]",
            units=pyunits.dimensionless,
        )

        self.particle_size_exp = Param(
            self.rate_reaction_idx,
            initialize={"R1": 0.0, "R2": 0.23, "R3": 0.23, "R4": 0.23},
            doc="Particle size exponent [-]",
            units=pyunits.dimensionless,
        )

        # Sintering: Fe = 1.0 (T < 873 K), 0.7 (T > 873 K)
        self.f_Fe_high = Param(default=1.0, mutable=True,
            doc="Active iron fraction below sintering temperature [-]")
        self.f_Fe_low = Param(default=0.7, mutable=True,
            doc="Active iron fraction above sintering temperature [-]")
        self.T_sinter = Param(default=873, mutable=True,
            doc="Sintering temperature [K]",
            units=pyunits.K)
        self.f_Fe_delta = Param(default=30.0, mutable=True,
            doc="Smoothing for sintering transition [K]",
            units=pyunits.K)


        self.rp_ref = Param(
            default=30e-6, mutable=True,
            doc="Reference particle radius [m]", units=pyunits.m,
        )

        # Equilibrium constants
        # R1 is irreversible, so no constant for it
        self.Keq_A = Param(
            self.reversible_rxn_idx,
            initialize={"R2": 6.6567, "R3": 0.8513, "R4": 1.5131},
            doc="Coefficient A [-]", units=pyunits.dimensionless, mutable=True,
        )
        self.Keq_B = Param(
            self.reversible_rxn_idx,
            initialize={"R2": 6476.4, "R3": 1395.5, "R4": 1394.9},
            doc="Coefficient B [K]", units=pyunits.K, mutable=True,
        )
        self.Keq_C = Param(
            self.reversible_rxn_idx,
            initialize={"R2": 181141.0, "R3": 253791.0, "R4": 745520.0},
            doc="Coefficient C [K^2]", units=pyunits.K**2, mutable=True,
        )

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "k_rxn": {"method": "_k_rxn"},
                "Keq_red": {"method": "_Keq_red"},
                "xi_red": {"method": "_xi_red"},
                "kc_full": {"method": "_kc_full"},
                "X_conv_I": {"method": "_X_conv_step"},
                "X_conv_II": {"method": "_X_conv_step"},
                "X_conv_III": {"method": "_X_conv_step"},
                "reaction_rate": {"method": "_reaction_rate"},
                "tau_residence": {"method": "_reaction_rate"},
                "C0_Fe2O3": {"method": "_reaction_rate"},
                "C0_Fe3O4": {"method": "_reaction_rate"},
                "C0_FeO": {"method": "_reaction_rate"},
            }
        )
        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m, 
                "mass": pyunits.kg,
                "amount": pyunits.mol, 
                "temperature": pyunits.K,
            }
        )


class _ReductionReactionBlock(ReactionBlockBase):

    """

    Methods applied to Reaction Blocks as a whole.

    """

    def initialize(blk, outlvl=idaeslog.NOTSET, optarg=None, solver="ipopt"):

        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="reactions")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="reactions")
        init_log.info_high("Starting initialization")

        for k in blk.keys():
            rep_key = k
            break

        state_var_flags = fix_state_vars(blk[rep_key].config.solid_state_block)

        Cflag = {} # Gas concentration flag
        Dflag = {} # Solid density flag

        for k, b in blk.items():
            for j in b.gas_state_ref.params.component_list:
                if b.gas_state_ref.dens_mol_comp[j].fixed is True:
                    Cflag[k, j] = True
                else:
                    Cflag[k, j] = False
                    b.gas_state_ref.dens_mol_comp[j].fix(
                        b.gas_state_ref.dens_mol_comp[j].value
                    )
            if b.solid_state_ref.dens_mass_skeletal.fixed is True:
                Dflag[k] = True
            else:
                Dflag[k] = False
                b.solid_state_ref.dens_mass_skeletal.fix(
                    b.solid_state_ref.dens_mass_skeletal.value
                )

        for k in blk.values():
            # Arrhenius
            if hasattr(k, "rate_constant_eqn"):
                for j in k.params.rate_reaction_idx:
                    calculate_variable_from_constraint(
                        k.k_rxn[j], k.rate_constant_eqn[j]
                    )

            # Equilibrium constants
            if hasattr(k, "Keq_red_eqn"):
                for j in k.params.reversible_rxn_idx:
                    calculate_variable_from_constraint(
                        k.Keq_red[j], k.Keq_red_eqn[j]
                    )

            # Equilibrium driving force
            if hasattr(k, "xi_red_eqn"):
                for j in k.params.reversible_rxn_idx:
                    calculate_variable_from_constraint(
                        k.xi_red[j], k.xi_red_eqn[j]
                    )

            # Full kc 
            if hasattr(k, "kc_full_eqn"):
                for s in k.params.step_idx:
                    calculate_variable_from_constraint(
                        k.kc_full[s], k.kc_full_eqn[s]
                    )

            # Reaction rates 
            if hasattr(k, "gen_rate_expression"):
                for j in k.params.rate_reaction_idx:
                    calculate_variable_from_constraint(
                        k.reaction_rate[j], k.gen_rate_expression[j]
                    )

        
        free_vars = 0
        for k in blk.values():
            free_vars += number_unfixed_variables_in_activated_equalities(k)

        if free_vars > 0:
            opt = get_solver(solver, optarg)
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solve_indexed_blocks(opt, [blk], tee=slc.tee)
        else:
            res = ""
        init_log.info_high(
            "reactions initialization complete {}.".format(idaeslog.condition(res))
        )

        
        for k in blk.values():
            revert_state_vars(k.config.solid_state_block, state_var_flags)
        for k, b in blk.items():
            for j in b.gas_state_ref.params.component_list:
                if Cflag[k, j] is False:
                    b.gas_state_ref.dens_mol_comp[j].unfix()
            if Dflag[k] is False:
                b.solid_state_ref.dens_mass_skeletal.unfix()

        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="reactions")
        init_log.info_high("States released.")


@declare_process_block_class("ReductionReactionBlock", block_class=_ReductionReactionBlock)
class ReactionBlockData(ReactionBlockDataBase):
    """
    Multi stage reduction reaction package for iron

    """

    CONFIG = ConfigBlock()
    CONFIG.declare(
        "parameters",
        ConfigValue(
            domain=is_reaction_parameter_block,
            description="Reference to Reaction Parameter Block.",
        ),
    )
    CONFIG.declare(
        "solid_state_block",
        ConfigValue(domain=is_state_block,
            description="Reference to solid StateBlock."),
    )
    CONFIG.declare(
        "gas_state_block",
        ConfigValue(domain=is_state_block,
            description="Reference to gas StateBlock."),
    )
    CONFIG.declare(
        "has_equilibrium", ConfigValue(default=False, domain=Bool,
            description="Equilibrium reaction flag."),
    )

    def build(self):
        super(ReactionBlockDataBase, self).build()
        add_object_reference(self, "_params", self.config.parameters)
        add_object_reference(
            self, "solid_state_ref",
            self.config.solid_state_block[self.index()]
        )
        add_object_reference(
            self, "gas_state_ref",
            self.config.gas_state_block[self.index()]
        )
        add_object_reference(
            self, "rate_reaction_stoichiometry",
            self.config.parameters.rate_reaction_stoichiometry,
        )
        add_object_reference(self, "dh_rxn", self.config.parameters.dh_rxn)

    
    # k_rxn: Arrhenius * f_Fe
    
    def _k_rxn(self):
        self.k_rxn = Var(
            self.params.rate_reaction_idx, domain=Reals, initialize=0.01,
            doc="Constant ki [m3/mol/s]",
            units=pyunits.m**3 / pyunits.mol / pyunits.s,
        )

        def rate_constant_eqn(b, j):
            T = b.solid_state_ref.temperature
            T_s = b.params.T_sinter
            delta = b.params.f_Fe_delta

            # Sigmoid
            f_Fe = (
                b.params.f_Fe_low
                + (b.params.f_Fe_high - b.params.f_Fe_low)
                / (1.0 + exp(-(T_s - T) / delta))
            )
            return b.k_rxn[j] == (
                b.params.k0_rxn[j]
                * exp(
                    -b.params.energy_activation[j]
                    / (
                        pyunits.convert(
                            Constants.gas_constant,
                            to_units=pyunits.J / pyunits.mol / pyunits.K,
                        )
                        * T
                    )
                )
                * f_Fe
            )

        try:
            self.rate_constant_eqn = Constraint(
                self.params.rate_reaction_idx, rule=rate_constant_eqn
            )
        except AttributeError:
            self.del_component(self.k_rxn)
            self.del_component(self.rate_constant_eqn)
            raise

    
    # Keq: equilibrium constant
    
    def _Keq_red(self):
        self.Keq_red = Var(
            self.params.reversible_rxn_idx, domain=Reals, initialize=1.0,
            doc="Equilibrium constant [-]",
            units=pyunits.dimensionless,
        )

        def Keq_red_eqn(b, j):
            T = b.solid_state_ref.temperature
            return b.Keq_red[j] == exp(
                b.params.Keq_A[j]
                - b.params.Keq_B[j] / T
                - b.params.Keq_C[j] / T**2
            )

        try:
            self.Keq_red_eqn = Constraint(
                self.params.reversible_rxn_idx, rule=Keq_red_eqn
            )
        except AttributeError:
            self.del_component(self.Keq_red)
            self.del_component(self.Keq_red_eqn)
            raise

    
    # xi: equilibrium driving force
    
    def _xi_red(self):
        self.xi_red = Var(
            self.params.reversible_rxn_idx, domain=Reals, initialize=0.9,
            doc="Equilibrium driving force [-]",
            units=pyunits.dimensionless,
        )

        def xi_red_eqn(b, j):
            C_H2 = b.gas_state_ref.dens_mol_comp["H2"]
            C_H2O = b.gas_state_ref.dens_mol_comp["H2O"]
            eps = b.params.eps
            Keq = b.Keq_red[j]
            ratio = (                    #Pseudo Huber smoothing
                sqrt(C_H2O**2 + eps**2)
                / (sqrt(C_H2**2 + eps**2) * Keq)
            )
            xi_raw = 1.0 - ratio
            return b.xi_red[j] == (
                0.5 * (xi_raw - eps + sqrt((xi_raw - eps) ** 2 + eps**2)) + eps
            )

        try:
            self.xi_red_eqn = Constraint(
                self.params.reversible_rxn_idx, rule=xi_red_eqn
            )
        except AttributeError:
            self.del_component(self.xi_red)
            self.del_component(self.xi_red_eqn)
            raise

    
    # kc_full: global apparent kinetic constant

    def _kc_full(self):
        self.kc_full = Var(
            self.params.step_idx, domain=Reals, initialize=1e-3,
            doc="Global apparent kc [s^-1] ",
            units=pyunits.s**(-1),
        )

        
        def kc_full_eqn(b, step):
            r = step_rxn[step]
            s_exp = b.params.rxn_steam_order[r]
            if r in b.params.reversible_rxn_idx and value(s_exp) > 0:
                eq_term = b.xi_red[r] ** s_exp
            else:
                eq_term = 1.0

            return b.kc_full[step] == (
                b.k_rxn[r]
                * (
                    sqrt(
                        b.gas_state_ref.dens_mol_comp["H2"] ** 2
                        + b.params.eps**2
                    )
                )
                ** b.params.rxn_order[r]
                * eq_term
                * (
                    b.params.rp_ref
                    / (b.solid_state_ref.params.particle_dia / 2)
                )
                ** b.params.particle_size_exp[r]
            )

        try:
            self.kc_full_eqn = Constraint(
                self.params.step_idx, rule=kc_full_eqn
            )
        except AttributeError:
            self.del_component(self.kc_full)
            self.del_component(self.kc_full_eqn)
            raise

    
    # X_conv: CORAL algebraic per-step conversions
    
    def _X_conv_step(self):

        """
        
        The constraint is added by the reactor script after initialization
        to avoid stiffness in the solvers.

        """

        self.X_conv_I = Var(
            domain=Reals, initialize=0.01, bounds=(0, 1),
            doc="CORAL Step I conversion (Fe2O3->Fe3O4) [-]",
        )
        self.X_conv_II = Var(
            domain=Reals, initialize=0.001, bounds=(0, 1),
            doc="CORAL Step II conversion (Fe3O4->FeO) [-]",
        )
        self.X_conv_III = Var(
            domain=Reals, initialize=0.001, bounds=(0, 1),
            doc="CORAL Step III conversion (FeO->Fe) [-]",
        )

    
    # reaction_rate [mol/m3/s]
    # rate_r = (2/|v|) * C0 * kc * (1-X) * sqrt(-ln(1-X))
    
    def _reaction_rate(self):
        self.reaction_rate = Var(
            self.params.rate_reaction_idx, domain=Reals, initialize=0,
            doc="Reaction rate [mol/m3/s]",
            units=pyunits.mol / pyunits.m**3 / pyunits.s,
        )
        self.tau_residence = Var(
            domain=Reals, initialize=100.0, bounds=(0.1, 1e5),
            doc="Solid residence time [s]",
            units=pyunits.s,
        )

        # C0 = (1 - phi) * rho_skel / MW
        # phi: intraparticle porosity [-]
        # rho_skel: skeletal density [kg/m^3]
        # MW: molecular weight of oxide[kg/mol]
        # ref:Pubchem
        #C0_Fe2O3 = (1 - 0.27) * 5250.0 / 0.15969 =23999
        #C0_Fe3O4 = (1 - 0.27) * 5170.0 / 0.23153 =16300
        #C0_FeO   = (1 - 0.27) * 5700.0 / 0.07184 =57920

        self.C0_Fe2O3 = Var(
            domain=Reals, initialize=23999,
            doc="Reference inlet Fe2O3 conc [mol/m3_particle] ",
            units=pyunits.mol / pyunits.m**3,
        )
        self.C0_Fe3O4 = Var(
            domain=Reals, initialize=16300,
            doc="Reference max Fe3O4 conc [mol/m3_particle]",
            units=pyunits.mol / pyunits.m**3,
        )
        self.C0_FeO = Var(
            domain=Reals, initialize=57920,
            doc="Reference max FeO conc [mol/m3_particle]",
            units=pyunits.mol / pyunits.m**3,
        )

        eps = smooth_avr

        def rate_rule(b, r):
            step = rxn_step[r]
            C0 = getattr(b, f"C0_{reactants[r]}")
            kc = b.kc_full[step]
            X_step = getattr(b, f"X_conv_{step}")
            avr = sqrt(eps + (-log(1.0 - X_step + eps)))

            return b.reaction_rate[r] == (
                (2.0 / reactant_abs[r])
                * C0
                * kc
                * (1.0 - X_step)
                * avr
            )

        try:
            self.gen_rate_expression = Constraint(
                self.params.rate_reaction_idx, rule=rate_rule
            )
        except AttributeError:
            self.del_component(self.reaction_rate)
            self.del_component(self.gen_rate_expression)
            raise

    def get_reaction_rate_basis(b):
        return MaterialFlowBasis.molar

    def model_check(blk):
        if value(blk.temperature) < blk.temperature.lb:
            _log.error("{} Temperature set below lower bound.".format(blk.name))
        if value(blk.temperature) > blk.temperature.ub:
            _log.error("{} Temperature set above upper bound.".format(blk.name))

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        def _set_default(v, s):
            for i in v:
                if iscale.get_scaling_factor(v[i]) is None:
                    iscale.set_scaling_factor(v[i], s)

        _set_default(self.k_rxn, 1e2)
        _set_default(self.Keq_red, 1e0)
        _set_default(self.xi_red, 1e0)
        _set_default(self.kc_full, 1e3)
        _set_default(self.reaction_rate, 1e2)

        # Scalar Vars
        for v, s in [
            (self.X_conv_I, 1e2), (self.X_conv_II, 1e2), (self.X_conv_III, 1e2),
            (self.tau_residence, 1e-2),
            (self.C0_Fe2O3, 1e-4), (self.C0_Fe3O4, 1e-4), (self.C0_FeO, 1e-4),
        ]:
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, s)

        if self.is_property_constructed("rate_constant_eqn"):
            for i, c in self.rate_constant_eqn.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.k_rxn[i]), overwrite=False
                )
        if self.is_property_constructed("Keq_red_eqn"):
            for i, c in self.Keq_red_eqn.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.Keq_red[i]), overwrite=False
                )
        if self.is_property_constructed("xi_red_eqn"):
            for i, c in self.xi_red_eqn.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.xi_red[i]), overwrite=False
                )
        if self.is_property_constructed("kc_full_eqn"):
            for i, c in self.kc_full_eqn.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.kc_full[i]), overwrite=False
                )
        if self.is_property_constructed("gen_rate_expression"):
            for i, c in self.gen_rate_expression.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.reaction_rate[i]),
                    overwrite=False,
                )
