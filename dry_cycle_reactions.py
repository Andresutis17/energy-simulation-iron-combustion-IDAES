"""
Multi stage oxidation reaction property package for iron energy CLC air reactor.

Reactions:
(1) 2Fe + O2     -> 2FeO     
(2) 6FeO + O2    -> 2Fe3O4   
(3) 4Fe3O4 + O2  -> 6Fe2O3   

Kinetic data from:
A. Abad et al., Chem. Eng. Sci. 62 (2007) 533-549.
H. Leion et al., Fuel 87 (2008) 2037-2047.
S. Jiménez et al., Energy Fuels (2026).
"""


from pyomo.environ import (
    Constraint,
    exp,
    Param,
    Reals,
    Set,
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


__author__ = "Custom package based on IDAES by Chinedu Okoli"

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("DryCycleReactionParameterBlock")
class ReactionParameterData(ReactionParameterBlock):

    """
    Reaction Parameter Block for multi-stage oxidation reactions.

    """

    
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

        self._reaction_block_class = DryCycleReactionBlock

        # Reaction Index
        self.rate_reaction_idx = Set(initialize=["R1", "R2", "R3"])


        # Reaction Stoichiometry
        # (1) 2Fe + O2     -> 2FeO 
        # (2) 6FeO + O2    -> 2Fe3O4 
        # (3) 4Fe3O4 + O2  -> 6Fe2O3 

        self.rate_reaction_stoichiometry = {
            ("R1", "Vap", "O2"): -1,
            ("R1", "Vap", "N2"): 0,
            ("R1", "Vap", "CO2"): 0,
            ("R1", "Vap", "H2O"): 0,
            ("R1", "Vap", "H2"): 0,
            ("R1", "Sol", "Fe2O3"): 0,
            ("R1", "Sol", "Fe3O4"): 0,
            ("R1", "Sol", "FeO"): 2,
            ("R1", "Sol", "Fe"): -2,
            ("R1", "Sol", "Al2O3"): 0,

            ("R2", "Vap", "O2"): -1,
            ("R2", "Vap", "N2"): 0,
            ("R2", "Vap", "CO2"): 0,
            ("R2", "Vap", "H2O"): 0,
            ("R2", "Vap", "H2"): 0,
            ("R2", "Sol", "Fe2O3"): 0,
            ("R2", "Sol", "Fe3O4"): 2,
            ("R2", "Sol", "FeO"): -6,
            ("R2", "Sol", "Fe"): 0,
            ("R2", "Sol", "Al2O3"): 0,

            ("R3", "Vap", "O2"): -1,
            ("R3", "Vap", "N2"): 0,
            ("R3", "Vap", "CO2"): 0,
            ("R3", "Vap", "H2O"): 0,
            ("R3", "Vap", "H2"): 0,
            ("R3", "Sol", "Fe2O3"): 6,
            ("R3", "Sol", "Fe3O4"): -4,
            ("R3", "Sol", "FeO"): 0,
            ("R3", "Sol", "Fe"): 0,
            ("R3", "Sol", "Al2O3"): 0,

        }

        # Standard Heat of Reaction - J/mol_rxn- ref: NIST webbook
        # R1: dH = 2*(-272.04) - 0 = -544.08e3 J/mol
        # R2: dH = 2*(-1120.89) - 6*(-272.04) = -609.54e3 J/mol
        # R3: dH = 6*(-825.50) - 4*(-1120.89) = -469.44e3 J/mol

        dh_rxn_dict = {
            "R1": -544.09e3 ,
            "R2": -609.52e3,
            "R3": -469.44e3,
        }
        self.dh_rxn = Param(
            self.rate_reaction_idx,
            initialize=dh_rxn_dict,
            doc="Heat of reaction [J/mol]",
            units=pyunits.J / pyunits.mol,
        )

        # Smoothing factor
        self.eps = Param(
            mutable=True,
            default=1e-8,
            doc="Smoothing Factor",
            units=pyunits.mol / pyunits.m**3,
        )
        # Reaction rate scale factor
        self._scale_factor_rxn = Param(
            mutable=True,
            default=1,
            doc="Scale Factor for reaction eqn.",
        )

        # Reaction properties that can be estimated

        # Particle grain radius within OC particle
        self.grain_radius = Var(
            domain=Reals,
            initialize=2.6e-7,
            doc="Representative grain radius [m]",
            units=pyunits.m,
        )
        self.grain_radius.fix()

        # Molar density OC particle
        self.dens_mol_sol = Var(
            domain=Reals,
            initialize=22472,
            doc="Molar density of OC particle [mol/m^3]",
            units=pyunits.mol / pyunits.m**3,
        )
        self.dens_mol_sol.fix()

        # Available volume for reaction
        self.a_vol = Var(
            domain=Reals,
            initialize=0.28,
            doc="Available reaction vol. per vol. of OC",
            units=pyunits.dimensionless,
        )
        self.a_vol.fix()

        # Activation Energy [J/mol] - ref: S. Jiménez et al., Energy Fuels (2026)
        self.energy_activation = Var(
            self.rate_reaction_idx,
            domain=Reals,
            initialize=1.4e4,
            doc="Activation energy [J/mol]",
            units=pyunits.J / pyunits.mol,
        )
        self.energy_activation["R1"].fix(12e3)
        self.energy_activation["R2"].fix(40e3)
        self.energy_activation["R3"].fix(12e3)

        # Reaction order in gas species
        self.rxn_order = Var(
            self.rate_reaction_idx,
            domain=Reals,
            initialize=1.0,
            doc="Reaction order [-]",
            units=pyunits.dimensionless,
        )
        self.rxn_order["R1"].fix(1.0)
        self.rxn_order["R2"].fix(1.0)
        self.rxn_order["R3"].fix(1.0)

        # Pre-exponential factor [m/s] - ref: S. Jiménez et al., Energy Fuels (2026)
        # An unit convertion was calculated
        self.k0_rxn = Var(
            self.rate_reaction_idx,
            domain=Reals,
            initialize=3.1e-4,
            doc="Pre-exponential factor [m/s]",
            units=pyunits.m / pyunits.s,
        )
        self.k0_rxn["R1"].fix(9.1e-5)
        self.k0_rxn["R2"].fix(1.9e-4)
        self.k0_rxn["R3"].fix(1.4e-6)

        # Reactant solid species for each reaction (used in rate expression)
        self._reactant_solid = {
            "R1": "Fe",
            "R2": "FeO",
            "R3": "Fe3O4",
        }

        # Stoichiometric coefficient of reactant solid
        self._reactant_stoich = {
            "R1": -2,
            "R2": -6,
            "R3": -4,
        }

    @classmethod
    def define_metadata(cls, obj):
        obj.define_custom_properties(
            {
                "OC_conv": {"method": "_OC_conv", "units": None},
                "OC_conv_temp": {"method": "_OC_conv_temp", "units": None},
            }
        )
        obj.add_properties(
            {
                "k_rxn": {"method": "_k_rxn"},
                "reaction_rate": {"method": "_reaction_rate"},
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


class _DryCycleReactionBlock(ReactionBlockBase):
    """
    Methods applied to Reaction Blocks as a whole.

    """

    def initialize(blk, outlvl=idaeslog.NOTSET, optarg=None, solver="ipopt"):
        """
        Initialisation routine for reaction package.

        Keyword ArgumO$ents:
            outlvl : sets output level of initialization routine
                 * 0 = Use default idaes.init logger setting
                 * 1 = Maximum output
                 * 2 = Include solver output
                 * 3 = Return solver state for each step in subroutines
                 * 4 = Return solver state for each step in routine
                 * 5 = Final initialization status and exceptions
                 * 6 = No output
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = "ipopt")
        Returns:
            None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="reactions")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="reactions")

        init_log.info_high("Starting initialization")

        
        for k in blk.keys():
            rep_key = k
            break

        # Fix state variables of the primary (solid) state block
        state_var_flags = fix_state_vars(blk[rep_key].config.solid_state_block)

       
        Cflag = {}  # Gas concentration flag
        Dflag = {}  # Solid density flag

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
            if hasattr(k, "OC_conv_eqn"):
                calculate_variable_from_constraint(k.OC_conv, k.OC_conv_eqn)

            if hasattr(k, "OC_conv_temp_eqn"):
                calculate_variable_from_constraint(k.OC_conv_temp, k.OC_conv_temp_eqn)

            for j in k.params.rate_reaction_idx:
                if hasattr(k, "rate_constant_eqn"):
                    calculate_variable_from_constraint(
                        k.k_rxn[j], k.rate_constant_eqn[j]
                    )

                if hasattr(k, "gen_rate_expression"):
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


@declare_process_block_class("DryCycleReactionBlock", block_class=_DryCycleReactionBlock)
class ReactionBlockData(ReactionBlockDataBase):
    """

    Multi stage oxidation reaction package for iron-energy CLC air reactor.

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
        ConfigValue(
            domain=is_state_block,
            description="Reference to solid phase StateBlock.",
        ),
    )
    CONFIG.declare(
        "gas_state_block",
        ConfigValue(
            domain=is_state_block,
            description="Reference to gas phase StateBlock.",
        ),
    )
    CONFIG.declare(
        "has_equilibrium",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Equilibrium reaction construction flag",
            doc="""
        Indicates whether terms for equilibrium controlled reactions
        should be constructed,
        **default** - True.
        **Valid values:** {
        **True** - include equilibrium reaction terms,
        **False** - exclude equilibrium reaction terms.}
        """,
        ),
    )

    def build(self):

        """
        Callable method for Block construction

        """
        super(ReactionBlockDataBase, self).build()

        
        add_object_reference(self, "_params", self.config.parameters)

        add_object_reference(
            self, "solid_state_ref", self.config.solid_state_block[self.index()]
        )
        add_object_reference(
            self, "gas_state_ref", self.config.gas_state_block[self.index()]
        )

        
        add_object_reference(
            self,
            "rate_reaction_stoichiometry",
            self.config.parameters.rate_reaction_stoichiometry,
        )

        
        add_object_reference(self, "dh_rxn", self.config.parameters.dh_rxn)

    def _k_rxn(self):
        self.k_rxn = Var(
            self.params.rate_reaction_idx,
            domain=Reals,
            initialize=1,
            doc="Rate constant [m/s]",
            units=pyunits.m / pyunits.s,
        )

        def rate_constant_eqn(b, j):
            return self.k_rxn[j] == (
                self.params.k0_rxn[j]
                * exp(
                    -self.params.energy_activation[j]
                    / (
                        pyunits.convert(
                            Constants.gas_constant,
                            to_units=pyunits.J / pyunits.mol / pyunits.K,
                        )
                        * self.solid_state_ref.temperature
                    )
                )
            )

        try:
            self.rate_constant_eqn = Constraint(
                self.params.rate_reaction_idx, rule=rate_constant_eqn
            )
        except AttributeError:
            self.del_component(self.k_rxn)
            self.del_component(self.rate_constant_eqn)
            raise

    def _OC_conv(self):
        self.OC_conv = Var(
            domain=Reals,
            initialize=0.0,
            doc="Global conversion based on O/Fe ratio",
            units=pyunits.dimensionless,
        )

        def OC_conv_eqn(b):
            mw = b.solid_state_ref.params.mw_comp
            x = b.solid_state_ref.mass_frac_comp

            # Total O atoms from iron oxides per kg of solid
            # Fe:0, FeO:1, Fe3O4:4, Fe2O3:3
            O_atoms = (
                x["FeO"] / mw["FeO"] * 1
                + x["Fe3O4"] / mw["Fe3O4"] * 4
                + x["Fe2O3"] / mw["Fe2O3"] * 3
            )

            # Total Fe atoms per kg of solid
            # Fe:1, FeO:1, Fe3O4:3, Fe2O3:2
            Fe_atoms = (
                x["Fe"] / mw["Fe"] * 1
                + x["FeO"] / mw["FeO"] * 1
                + x["Fe3O4"] / mw["Fe3O4"] * 3
                + x["Fe2O3"] / mw["Fe2O3"] * 2
            )

            
            # O/Fe in Fe2O3 = 3/2 so the factor = 2/3
            return b.OC_conv == (2.0 / 3.0) * O_atoms / Fe_atoms

        try:
            self.OC_conv_eqn = Constraint(rule=OC_conv_eqn)
        except AttributeError:
            self.del_component(self.OC_conv)
            self.del_component(self.OC_conv_eqn)

    def _OC_conv_temp(self):
        self.OC_conv_temp = Var(
            domain=Reals,
            initialize=1.0,
            doc="Reformulation term for X to help scaling",
            units=pyunits.dimensionless,
        )

        def OC_conv_temp_eqn(b):
            return b.OC_conv_temp**3 == (1 - b.OC_conv) ** 2

        try:
            self.OC_conv_temp_eqn = Constraint(rule=OC_conv_temp_eqn)
        except AttributeError:
            self.del_component(self.OC_conv_temp)
            self.del_component(self.OC_conv_temp_eqn)

    # General rate of reaction method
    def _reaction_rate(self):
        self.reaction_rate = Var(
            self.params.rate_reaction_idx,
            domain=Reals,
            initialize=0,
            doc="Gen. rate of reaction [mol_rxn/m3.s]",
            units=pyunits.mol / pyunits.m**3 / pyunits.s,
        )

        def rate_rule(b, r):
            # Get the reactant solid species
            reactant = b.params._reactant_solid[r]
            nu_reactant = b.params._reactant_stoich[r]

            return b.reaction_rate[
                r
            ] == b.params._scale_factor_rxn * (
                b.solid_state_ref.mass_frac_comp[reactant]
                * (1 - b.solid_state_ref.particle_porosity)
                * b.solid_state_ref.dens_mass_skeletal
                * (b.params.a_vol / (b.solid_state_ref.params.mw_comp[reactant]))
                * 3
                * -nu_reactant
                * b.k_rxn[r]
                * (
                    (
                        (b.gas_state_ref.dens_mol_comp["O2"] ** 2 + b.params.eps**2)
                        ** 0.5
                    )
                    ** b.params.rxn_order[r]
                )
                * b.OC_conv_temp
                / (b.params.dens_mol_sol * b.params.grain_radius)
                / (-nu_reactant)
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

        # Set default scaling
        def _set_default_factor(v, s):
            for i in v:
                if iscale.get_scaling_factor(v[i]) is None:
                    iscale.set_scaling_factor(v[i], s)

        _set_default_factor(self.k_rxn, 1e6)
        _set_default_factor(self.OC_conv, 1e6)
        _set_default_factor(self.OC_conv_temp, 1e3)
        _set_default_factor(self.reaction_rate, 1e4)

        if self.is_property_constructed("OC_conv_eqn"):
            iscale.constraint_scaling_transform(
                self.OC_conv_eqn,
                iscale.get_scaling_factor(self.OC_conv),
                overwrite=False,
            )

        if self.is_property_constructed("OC_conv_temp_eqn"):
            iscale.constraint_scaling_transform(
                self.OC_conv_temp_eqn,
                iscale.get_scaling_factor(self.OC_conv_temp),
                overwrite=False,
            )

        if self.is_property_constructed("rate_constant_eqn"):
            for i, c in self.rate_constant_eqn.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.k_rxn[i]), overwrite=False
                )

        if self.is_property_constructed("gen_rate_expression"):
            for i, c in self.gen_rate_expression.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.reaction_rate[i]), overwrite=False
                )