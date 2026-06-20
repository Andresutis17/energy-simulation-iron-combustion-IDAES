
"""

Overall lumped oxidation reaction package for iron

2 Fe + 3/2 O2 -> Fe2O3

References:
A. Abad et al., CORAL Deliverable D3.1 (2025) 

"""

from pyomo.environ import (
    Constraint,
    exp,
    Param,
    Reals,
    Set,
    tanh,
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

__author__ = "Custom oxidation package based on IDAES by Chinedu Okoli and CORAL Deliverable D3.1"

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("DryOxidationReactionParameterBlock")
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

        self._reaction_block_class = ReactionBlock

        self.rate_reaction_idx = Set(initialize=["R1"])

        # R1: 2Fe + 3/2 O2 -> Fe2O3 
        self.rate_reaction_stoichiometry = {
            ("R1", "Vap", "O2"): -1.5,
            ("R1", "Vap", "N2"): 0,
            ("R1", "Vap", "CO2"): 0,
            ("R1", "Vap", "H2O"): 0,
            ("R1", "Vap", "H2"): 0,
            ("R1", "Sol", "Fe2O3"): 1,
            ("R1", "Sol", "Fe"): -2,
            ("R1", "Sol", "Fe3O4"): 0,
            ("R1", "Sol", "FeO"): 0,
            ("R1", "Sol", "Al2O3"): 0,
        }

        # Standard Heat of Reaction - J/mol_rxn- ref: NIST webbook 
        # R1: dH = 1(-825.5032) - 2(0) - 3/2(0) = -825.5032e3
        self.dh_rxn = Param(
            self.rate_reaction_idx,
            initialize={"R1": -825.5032e3},
            doc="Standard heat of reaction [J/mol]",
            units=pyunits.J / pyunits.mol,
        )

        # Smoothing factors
        self.eps = Param(
            mutable=True, default=1e-8,
            doc="Smoothing factor for O2 concentration [mol/m^3]",
            units=pyunits.mol / pyunits.m**3,
        )
        self.eps_x = Param(
            mutable=True, default=1e-8,
            doc="Dimensionless smoothing factor for conversion [-]",
            units=pyunits.dimensionless,
        )
        self._scale_factor_rxn = Param(
            mutable=True, default=1,
            doc="Scale factor for reaction eqn",
        )

       
        self.rp_ref = Param(
            default=30e-6, mutable=True,
            doc="Reference particle radius [m]",
            units=pyunits.m,
        )

        
        # Step I: Chemical reaction control 
        
        
        self.k_chr_0 = Param(
            initialize=5.61e-1, mutable=True,
            doc="Pre exponential factor [m3/mol/s] ",
            units=pyunits.m**3 / pyunits.mol / pyunits.s,
        )
        self.E_chr = Param(
            initialize=45500, mutable=False,
            doc="Activation energy [J/mol] ",
            units=pyunits.J / pyunits.mol,
        )
        self.n_chr = Param(
            initialize=1.0, mutable=True,
            doc="Reaction order [-]",
            units=pyunits.dimensionless,
        )
        self.n_k_rp = Param(
            initialize=0.63, mutable=True,
            doc="Particle size exponent [-]",
            units=pyunits.dimensionless,
        )

        


        #Step II: Diffusion control
        
        
        self.D_g_0 = Param(
            initialize=2.0e-6, mutable=True,
            doc="Gas diffusion pre-exponential factor [m3/mol/s]",
            units=pyunits.m**3 / pyunits.mol / pyunits.s,
        )
        self.E_g = Param(
            initialize=10000, mutable=False,
            doc="Gas diffusion activation energy [J/mol] ",
            units=pyunits.J / pyunits.mol,
        )
        self.D_s_0 = Param(
            initialize=7.28e13, mutable=True,
            doc="Solid diffusion pre exponential [m3/mol/s]",
            units=pyunits.m**3 / pyunits.mol / pyunits.s,
        )
        self.E_s = Param(
            initialize=367300, mutable=False,
            doc="Solid diffusion activation energy [J/mol] ",
            units=pyunits.J / pyunits.mol,
        )
        self.n_dif = Param(
            initialize=1.0, mutable=True,
            doc="Reaction order [-] ",
            units=pyunits.dimensionless,
        )
        self.n_D_rp = Param(
            initialize=2.0, mutable=True,
            doc="Particle size exponent [-]",
            units=pyunits.dimensionless,
        )



        
        # Transition point X_chr
        
        
        self.Xchr_O2_a = Param(
            initialize=8.351e-1, mutable=True,
            doc="XchrO2 pre-exponential [-]",
            units=pyunits.dimensionless,
        )
        self.Xchr_O2_b = Param(
            initialize=-2073.0, mutable=False,
            doc="XchrO2 temperature exponent [K]",
            units=pyunits.K,
        )
        self.a_O2_0 = Param(
            initialize=2.127e-4, mutable=True,
            doc="aO2 correction factor [-] ",
            units=pyunits.dimensionless,
        )
        self.a_O2_a = Param(
            initialize=4.28e-9, mutable=True,
            doc="aO2 temperature pre exponential factor [-]",
            units=pyunits.dimensionless,
        )
        self.a_O2_b = Param(
            initialize=11560.0, mutable=False,
            doc="aO2 temperature exponent [K] ",
            units=pyunits.K,
        )
        self.b_O2_a = Param(
            initialize=40.0, mutable=True,
            doc="bO2 pre-exponential [m3/mol] ",
            units=pyunits.m**3 / pyunits.mol,
        )
        self.b_O2_b = Param(
            initialize=-3060.0, mutable=False,
            doc="bO2 temperature exponent [K] ",
            units=pyunits.K,
        )
        self.n_X_rp = Param(
            initialize=0.6, mutable=True,
            doc="Particle size exponent for Xchr [-] ",
            units=pyunits.dimensionless,
        )

        # Width of smooth transition 
        self.delta_smooth = Param(
            mutable=True, default=0.02,
            doc="Width of smooth sigmoid zone for SCM/ZLT [-]",
            units=pyunits.dimensionless,
        )

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "reaction_rate": {"method": "_reaction_rate"},
            }
        )
        obj.define_custom_properties(
            {
                "OC_conv": {"method": "_OC_conv", "units": None},
                "k_chr": {"method": "_k_chr"},
                "D_eff": {"method": "_D_eff"},
                "X_chr": {"method": "_X_chr"},
                "C_O2_smooth": {"method": "_rate_components"},
                "dXdt_I": {"method": "_rate_components"},
                "dXdt_II": {"method": "_rate_components"},
                "sigmoid_w": {"method": "_rate_components"},
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


class OxiDryReactionBlock(ReactionBlockBase):

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
            # Smoothed O2 concentration
            if hasattr(k, "C_O2_smooth_eqn"):
                calculate_variable_from_constraint(
                    k.C_O2_smooth, k.C_O2_smooth_eqn
                )

            # Step I: k_chr 
            if hasattr(k, "k_chr_eqn"):
                calculate_variable_from_constraint(k.k_chr, k.k_chr_eqn)

            # Step II: D_eff 
            if hasattr(k, "D_eff_eqn"):
                calculate_variable_from_constraint(k.D_eff, k.D_eff_eqn)

            # Transition point X_chr 
            if hasattr(k, "X_chr_eqn"):
                calculate_variable_from_constraint(k.X_chr, k.X_chr_eqn)

            # Rate components dXdt_I, dXdt_II, sigmoid_w
            if hasattr(k, "dXdt_I_eqn"):
                calculate_variable_from_constraint(k.dXdt_I, k.dXdt_I_eqn)
            if hasattr(k, "dXdt_II_eqn"):
                calculate_variable_from_constraint(k.dXdt_II, k.dXdt_II_eqn)
            if hasattr(k, "sigmoid_w_eqn"):
                calculate_variable_from_constraint(k.sigmoid_w, k.sigmoid_w_eqn)

            # OC_conv algebraic from mass fractions
            if hasattr(k, "OC_conv_eqn"):
                calculate_variable_from_constraint(k.OC_conv, k.OC_conv_eqn)

            # Reeaction_rate
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


@declare_process_block_class("ReactionBlock", block_class=OxiDryReactionBlock)
class ReactionBlockData(ReactionBlockDataBase):

    """

    Overall lumped oxidation reaction package for iron

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

    
    # Fraction of Fe converted to Fe2O3
    
    def _OC_conv(self):
        self.OC_conv = Var(
            domain=Reals, initialize=0.0, bounds=(0, 1),
            doc="Fraction of Fe converted to Fe2O3 [-]",
            units=pyunits.dimensionless,
        )

        def OC_conv_eqn(b):
            return b.OC_conv * (
                b.solid_state_ref.mass_frac_comp["Fe2O3"]
                * 2 / b.solid_state_ref.params.mw_comp["Fe2O3"]
                + b.solid_state_ref.mass_frac_comp["Fe"]
                / b.solid_state_ref.params.mw_comp["Fe"]
            ) == (
                b.solid_state_ref.mass_frac_comp["Fe2O3"]
                * 2 / b.solid_state_ref.params.mw_comp["Fe2O3"]
            )

        try:
            self.OC_conv_eqn = Constraint(rule=OC_conv_eqn)
        except AttributeError:
            self.del_component(self.OC_conv)
            self.del_component(self.OC_conv_eqn)
            raise

    
    
    # k_chr = k_chr_0 * (rp_ref/rp)^(n_k_rp) * exp(-E_chr/(Rg*T))
    
    def _k_chr(self):
        self.k_chr = Var(
            domain=Reals, initialize=0.002,
            doc="Rate constant [m3/mol/s]" ,
            units=pyunits.m**3 / pyunits.mol / pyunits.s,
        )

        def k_chr_eqn(b):
            Rg = pyunits.convert(
                Constants.gas_constant,
                to_units=pyunits.J / pyunits.mol / pyunits.K,
            )
            rp = b.solid_state_ref.params.particle_dia / 2
            return b.k_chr == (
                b.params.k_chr_0
                * (b.params.rp_ref / rp) ** b.params.n_k_rp
                * exp(-b.params.E_chr / (Rg * b.solid_state_ref.temperature))
            )

        try:
            self.k_chr_eqn = Constraint(rule=k_chr_eqn)
        except AttributeError:
            self.del_component(self.k_chr)
            self.del_component(self.k_chr_eqn)
            raise

    
    
    # D_eff = (D_g + D_s) * (rp_ref/rp)^n_D_rp
    
    def _D_eff(self):
        self.D_eff = Var(
            domain=Reals, initialize=1e-5,
            doc="Effective diffusion coefficient  [m3/mol/s]",
            units=pyunits.m**3 / pyunits.mol / pyunits.s,
        )

        def D_eff_eqn(b):
            Rg = pyunits.convert(
                Constants.gas_constant,
                to_units=pyunits.J / pyunits.mol / pyunits.K,
            )
            T = b.solid_state_ref.temperature
            rp = b.solid_state_ref.params.particle_dia / 2
            D_g = b.params.D_g_0 * exp(-b.params.E_g / (Rg * T))
            D_s = b.params.D_s_0 * exp(-b.params.E_s / (Rg * T))
            return b.D_eff == (
                (D_g + D_s)
                * (b.params.rp_ref / rp) ** b.params.n_D_rp
            )

        try:
            self.D_eff_eqn = Constraint(rule=D_eff_eqn)
        except AttributeError:
            self.del_component(self.D_eff)
            self.del_component(self.D_eff_eqn)
            raise

    
    
    # X_chr = (Xchr_O2 + a_O2*exp(b_O2*C_O2)) * (rp_ref/rp)^n_X_rp
    
    def _X_chr(self):
        self.X_chr = Var(
            domain=Reals, initialize=0.15,
            doc="Transition conversion  [-] ",
            units=pyunits.dimensionless,
        )

        def X_chr_eqn(b):
            T = b.solid_state_ref.temperature
            rp = b.solid_state_ref.params.particle_dia / 2
            Xchr_O2 = b.params.Xchr_O2_a * exp(b.params.Xchr_O2_b / T)
            a_O2 = (
                b.params.a_O2_0
                + b.params.a_O2_a * exp(b.params.a_O2_b / T)
            )
            b_O2 = b.params.b_O2_a * exp(b.params.b_O2_b / T)
            return b.X_chr == (
                (Xchr_O2 + a_O2 * exp(b_O2 * b.C_O2_smooth))
                * (b.params.rp_ref / rp) ** b.params.n_X_rp
            )

        try:
            self.X_chr_eqn = Constraint(rule=X_chr_eqn)
        except AttributeError:
            self.del_component(self.X_chr)
            self.del_component(self.X_chr_eqn)
            raise



    # Rate components C_O2_smooth, dXdt_I, dXdt_II, sigmoid_w
    
    def _rate_components(self):
        self.C_O2_smooth = Var(
            domain=Reals, initialize=0.5,
            doc="Smoothed O2 concentration [mol/m3]",
            units=pyunits.mol / pyunits.m**3,
        )
        self.dXdt_I = Var(
            domain=Reals, initialize=0.01,
            doc="Rate dX/dt Step I [1/s] ",
            units=pyunits.s**(-1),
        )
        self.dXdt_II = Var(
            domain=Reals, initialize=0.001,
            doc="Rate dX/dt Step II [1/s] ",
            units=pyunits.s**(-1),
        )
        self.sigmoid_w = Var(
            domain=Reals, initialize=0.0,
            doc="Sigmoid blending weight [-]",
            units=pyunits.dimensionless,
        )

        def C_O2_smooth_eqn(b):
            return b.C_O2_smooth == (
                (b.gas_state_ref.dens_mol_comp["O2"] ** 2
                 + b.params.eps ** 2) ** 0.5
            )
        
        # dX/dt = (3/tau_chr) * (1-X)^(2/3)

        def dXdt_I_eqn(b):
            tau_chr = 1.0 / (b.k_chr * b.C_O2_smooth ** b.params.n_chr)
            eps_x = b.params.eps_x
            return b.dXdt_I == (
                (3.0 / tau_chr)
                * (1 - b.OC_conv + eps_x) ** (2.0 / 3.0)
            )

        # ZLT 3D diffusion rate
        # Reformulated to avoid (1-X_dif)^(-1/3) divergence at X_dif=0

        def dXdt_II_eqn(b):
            tau_dif = 1.0 / (b.D_eff * b.C_O2_smooth ** b.params.n_dif)
            eps_x = b.params.eps_x
            X = b.OC_conv
            X_chr = b.X_chr

            # Transformed variable with smooth clipping 
            X_dif_raw = (X - X_chr) / (1 - X_chr + eps_x)
            X_dif = (X_dif_raw**2 + (eps_x * 0.1) ** 2) ** 0.5
            omXdif = 1 - X_dif + eps_x

            dXdif_dt = (
                (3.0 / (2.0 * tau_dif))
                * omXdif ** (5.0 / 3.0)
                / (1 - omXdif ** (1.0 / 3.0) + eps_x)
            )
            return b.dXdt_II == (1 - X_chr) * dXdif_dt
        
        # Sigmoid for a smoorh transition
        def sigmoid_w_eqn(b):
            return b.sigmoid_w == 0.5 * (
                1 + tanh((b.OC_conv - b.X_chr) / b.params.delta_smooth)
            )

        for name, rule in [
            ("C_O2_smooth_eqn", C_O2_smooth_eqn),
            ("dXdt_I_eqn", dXdt_I_eqn),
            ("dXdt_II_eqn", dXdt_II_eqn),
            ("sigmoid_w_eqn", sigmoid_w_eqn),
        ]:
            try:
                setattr(self, name, Constraint(rule=rule))
            except AttributeError:
                for attr in [name, name.replace("_eqn", "")]:
                    if hasattr(self, attr):
                        self.del_component(getattr(self, attr))
                raise

    
    # reaction_rate [mol/m3/s]
    # rate = (dX/dt * n_Fe_vol) / |nu_Fe|
    # n_Fe_vol = (1-porosity)*rho_skel*w_Fe/mw [mol/m3]
    
    def _reaction_rate(self):
        self.reaction_rate = Var(
            self.params.rate_reaction_idx,
            domain=Reals, initialize=0,
            doc="Rate of reaction [mol/m3/s] ",
            units=pyunits.mol / pyunits.m**3 / pyunits.s,
        )

        def rate_rule(b, r):
            if r != "R1":
                return Constraint.Skip

            dXdt = (1 - b.sigmoid_w) * b.dXdt_I + b.sigmoid_w * b.dXdt_II

            n_Fe_vol = (
                (1 - b.solid_state_ref.particle_porosity)
                * b.solid_state_ref.dens_mass_skeletal
                * b.solid_state_ref.mass_frac_comp["Fe"]
                / b.solid_state_ref.params.mw_comp["Fe"]
            )

            return b.reaction_rate[r] == b.params._scale_factor_rxn * (
                dXdt * n_Fe_vol / 2.0
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
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, s)

        _set_default(self.OC_conv, 1e2)
        _set_default(self.reaction_rate, 1e-8)
        _set_default(self.k_chr, 1e2)
        _set_default(self.D_eff, 1e2)
        _set_default(self.X_chr, 1e1)
        _set_default(self.dXdt_I, 1e3)
        _set_default(self.dXdt_II, 1e3)
        _set_default(self.sigmoid_w, 1e1)
        _set_default(self.C_O2_smooth, 1e1)

        if self.is_property_constructed("OC_conv_eqn"):
            iscale.constraint_scaling_transform(
                self.OC_conv_eqn,
                iscale.get_scaling_factor(self.OC_conv),
                overwrite=False,
            )
        if self.is_property_constructed("k_chr_eqn"):
            iscale.constraint_scaling_transform(
                self.k_chr_eqn,
                iscale.get_scaling_factor(self.k_chr),
                overwrite=False,
            )
        if self.is_property_constructed("D_eff_eqn"):
            iscale.constraint_scaling_transform(
                self.D_eff_eqn,
                iscale.get_scaling_factor(self.D_eff),
                overwrite=False,
            )
        if self.is_property_constructed("X_chr_eqn"):
            iscale.constraint_scaling_transform(
                self.X_chr_eqn,
                iscale.get_scaling_factor(self.X_chr),
                overwrite=False,
            )
        if self.is_property_constructed("C_O2_smooth_eqn"):
            iscale.constraint_scaling_transform(
                self.C_O2_smooth_eqn,
                iscale.get_scaling_factor(self.C_O2_smooth),
                overwrite=False,
            )
        if self.is_property_constructed("dXdt_I_eqn"):
            iscale.constraint_scaling_transform(
                self.dXdt_I_eqn,
                iscale.get_scaling_factor(self.dXdt_I),
                overwrite=False,
            )
        if self.is_property_constructed("dXdt_II_eqn"):
            iscale.constraint_scaling_transform(
                self.dXdt_II_eqn,
                iscale.get_scaling_factor(self.dXdt_II),
                overwrite=False,
            )
        if self.is_property_constructed("sigmoid_w_eqn"):
            iscale.constraint_scaling_transform(
                self.sigmoid_w_eqn,
                iscale.get_scaling_factor(self.sigmoid_w),
                overwrite=False,
            )
        if self.is_property_constructed("gen_rate_expression"):
            for i, c in self.gen_rate_expression.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.reaction_rate[i]),
                    overwrite=False,
                )
