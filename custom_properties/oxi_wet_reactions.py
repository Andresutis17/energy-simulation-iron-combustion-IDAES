
"""

Overall wet oxidation reactions package for iron
OW1: Fe    + H2O  ->  FeO   + H2
OW2: 3FeO  + H2O  ->  Fe3O4 + H2


References:
Kuhn et al., Applications in Energy and Combustion Science (2022)

"""

from pyomo.environ import (
    Constraint,
    exp,
    Param,
    Reals,
    Set,
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

__author__ = "Custom wet oxidation package based on Iron as recyclable energy carrier by Kuhn et al."

_log = idaeslog.getLogger(__name__)

# Mapping reaction to solid reactant component
_SOLID_REACTANT = {"OW1": "Fe", "OW2": "FeO"}


@declare_process_block_class("OxiWetReactionParameterBlock")
class OxiWetParameterData(ReactionParameterBlock):

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
        self._reaction_block_class = OxiWetReactionBlock

        self.rate_reaction_idx = Set(initialize=["OW1", "OW2"])
        self.reversible_rxn_idx = Set(initialize=[])

        # OW1: Fe + H2O -> FeO + H2
        # OW2: 3FeO + H2O -> Fe3O4 + H2
        self.rate_reaction_stoichiometry = {

            # OW1
            ("OW1", "Vap", "O2"): 0, 
            ("OW1", "Vap", "N2"): 0,
            ("OW1", "Vap", "CO2"): 0, 
            ("OW1", "Vap", "H2O"): -1,
            ("OW1", "Vap", "H2"): 1,
            ("OW1", "Sol", "Fe2O3"): 0, 
            ("OW1", "Sol", "Fe3O4"): 0,
            ("OW1", "Sol", "FeO"): 1, 
            ("OW1", "Sol", "Fe"): -1,
            ("OW1", "Sol", "Al2O3"): 0,


            # OW2
            ("OW2", "Vap", "O2"): 0, 
            ("OW2", "Vap", "N2"): 0,
            ("OW2", "Vap", "CO2"): 0, 
            ("OW2", "Vap", "H2O"): -1,
            ("OW2", "Vap", "H2"): 1,
            ("OW2", "Sol", "Fe2O3"): 0, 
            ("OW2", "Sol", "Fe3O4"): 1,
            ("OW2", "Sol", "FeO"): -3, 
            ("OW2", "Sol", "Fe"): 0,
            ("OW2", "Sol", "Al2O3"): 0,
        }

        
        # Standard Heat of Reaction - J/mol_rxn- ref: NIST webbook 
        # OW1: dH = 1(-272.04) + 1(0) - 1(0) - 1(-241.826) = -30.214e3
        # OW2: dH = 1(-1120.894) + 1(0) - 3(-272.04) - 1(-241.86) = -62.914e3
        dh_rxn_dict = {"OW1": -30.214e3, "OW2": -62.914e3}
        self.dh_rxn = Param(
            self.rate_reaction_idx,
            initialize=dh_rxn_dict,
            doc="Standard heat of reaction [J/mol]",
            units=pyunits.J / pyunits.mol,
        )

        # Smoothing factor
        self.eps = Param(
            mutable=True, default=1e-8,
            doc="Smoothing factor for concentration ratios",
            units=pyunits.mol / pyunits.m**3,
        )

        
        # Kinetic parameters 
        self.Ak_rxn = Param(
            self.rate_reaction_idx,
            initialize={"OW1": 6.5e7, "OW2": 2.2e7},
            doc="Pre-exponential factor [m3/mol/s] ",
            units=pyunits.m**3 / pyunits.mol / pyunits.s,
            mutable=True,
        )

        self.energy_activation = Param(
            self.rate_reaction_idx,
            initialize={"OW1": 231.0e3, "OW2": 235.0e3},
            doc="Activation energy [J/mol] ",
            units=pyunits.J / pyunits.mol,
        )

        self.rxn_order_solid = Param(
            self.rate_reaction_idx,
            initialize={"OW1": 1.0, "OW2": 1.0},
            doc="Solid reactant reaction order [-] ",
            units=pyunits.dimensionless,
        )

        self.rxn_order_H2O = Param(
            self.rate_reaction_idx,
            initialize={"OW1": 1.2, "OW2": 1.0},
            doc="H2O reaction order [-] ",
            units=pyunits.dimensionless,
        )

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "k_rxn": {"method": "_k_rxn"},
                "reaction_rate": {"method": "_reaction_rate"},
            }
        )
        obj.add_default_units(
            {
                "time": pyunits.s, "length": pyunits.m, "mass": pyunits.kg,
                "amount": pyunits.mol, "temperature": pyunits.K,
            }
        )


class _OxiWetReactionBlock(ReactionBlockBase):

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
            # Trigger lazy construction of reaction properties
            _ = k.k_rxn          # builds rate_constant_eqn
            _ = k.reaction_rate  # builds gen_rate_expression

            # Arrhenius constant
            if hasattr(k, "rate_constant_eqn"):
                for j in k.params.rate_reaction_idx:
                    calculate_variable_from_constraint(
                        k.k_rxn[j], k.rate_constant_eqn[j]
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


@declare_process_block_class("OxiWetReactionBlock", block_class=_OxiWetReactionBlock)
class OxiWetReactionBlockData(ReactionBlockDataBase):

    """

    Overall wet oxidation reaction package for iron

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
            description="Reference to solid StateBlock.",
        ),
    )
    CONFIG.declare(
        "gas_state_block",
        ConfigValue(
            domain=is_state_block,
            description="Reference to gas StateBlock.",
        ),
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

    
    
    # k_rxn = Ak * exp(-Ea/(Rg*T))

    def _k_rxn(self):
        self.k_rxn = Var(
            self.params.rate_reaction_idx, domain=Reals, initialize=0.01,
            doc="Rate constant [m3/mol/s]",
            units=pyunits.m**3 / pyunits.mol / pyunits.s,
        )

        def rate_constant_eqn(b, j):
            return b.k_rxn[j] == (
                b.params.Ak_rxn[j]
                * exp(
                    -b.params.energy_activation[j]
                    / (
                        pyunits.convert(
                            Constants.gas_constant,
                            to_units=pyunits.J / pyunits.mol / pyunits.K,
                        )
                        * b.solid_state_ref.temperature
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

    
    # rate = k_rxn * C_solid^ns * C_H2O^ng
    #ns: solid reaction order
    #ng: gas reaction order

    def _reaction_rate(self):
        self.reaction_rate = Var(
            self.params.rate_reaction_idx,
            domain=Reals, initialize=0,
            doc="Reaction rate [mol/m3/s] ",
            units=pyunits.mol / pyunits.m**3 / pyunits.s,
        )

        def rate_rule(b, r):
            ns = b.params.rxn_order_solid[r]
            ng = b.params.rxn_order_H2O[r]
            comp = _SOLID_REACTANT[r]
            C_solid = (
                (b.solid_state_ref.mass_frac_comp[comp]
                 / b.solid_state_ref.params.mw_comp[comp])
                * (1.0 - b.solid_state_ref.particle_porosity)
                * b.solid_state_ref.dens_mass_skeletal
            )
            C_H2O = b.gas_state_ref.dens_mol_comp["H2O"]

            return b.reaction_rate[r] == b.k_rxn[r] * C_solid**ns * C_H2O**ng

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
        _set_default(self.reaction_rate, 1e2)

        if self.is_property_constructed("rate_constant_eqn"):
            for i, c in self.rate_constant_eqn.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.k_rxn[i]), overwrite=False,
                )
        if self.is_property_constructed("gen_rate_expression"):
            for i, c in self.gen_rate_expression.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.reaction_rate[i]),
                    overwrite=False,
                )
