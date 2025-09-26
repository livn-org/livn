import equinox as eqx
from livn.models.rcsd.diffrax.pinsky_rinzel import PinskyRinzel


import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, RecursiveCheckpointAdjoint


class PinskyRinzel:
    def __init__(
        self,
        g_c=2.1,  # Coupling conductance [mS cm**-2]
        C_m=3.0,  # Membrane capacitance [uF cm**-2]
        p=0.5,  # Proportion of membrane area for soma
        g_L=0.1,  # Leak conductance [mS cm**-2]
        g_Na=30.0,  # Sodium conductance [mS cm**-2]
        g_DR=15.0,  # Delayed rectifier K+ conductance [mS cm**-2]
        g_Ca=10.0,  # Calcium conductance [mS cm**-2]
        g_AHP=0.8,  # After-hyperpolarization K+ conductance [mS cm**-2]
        g_C=15.0,  # Ca-dependent K+ conductance [mS cm**-2]
        E_L=-68.0,  # Leak reversal potential [mV]
        E_Na=60.0,  # Sodium reversal potential [mV]
        E_K=-75.0,  # Potassium reversal potential [mV]
        E_Ca=80.0,  # Calcium reversal potential [mV]
        initial_state=None,  # Optional initial state
    ):
        self.g_c = g_c
        self.C_m = C_m
        self.p = p

        # Conductances
        self.g_L = g_L
        self.g_Na = g_Na
        self.g_DR = g_DR
        self.g_Ca = g_Ca
        self.g_AHP = g_AHP
        self.g_C = g_C
        # Reversal potentials
        self.E_L = E_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_Ca = E_Ca

        if initial_state is None:
            self.initial_state = jnp.array(
                [-68.0, -68.0, 0.001, 0.999, 0.009, 0.007, 0.01, 0.2]
            )

        else:
            self.initial_state = jnp.array(initial_state)

        # Rate Functions

    def alpha_m(self, Vs):
        """Na+ channel activation rate"""
        V1 = Vs + 46.9
        alpha = -0.32 * V1 / (jnp.exp(-V1 / 4.0) - 1.0)
        return alpha

    def beta_m(self, Vs):
        """Na+ channel deactivation rate"""
        V2 = Vs + 19.9
        beta = 0.28 * V2 / (jnp.exp(V2 / 5.0) - 1.0)
        return beta

    def alpha_h(self, Vs):
        """Na+ channel inactivation rate"""
        alpha = 0.128 * jnp.exp((-43.0 - Vs) / 18.0)
        return alpha

    def beta_h(self, Vs):
        """Na+ channel deinactivation rate"""
        V5 = Vs + 20.0
        beta = 4.0 / (1 + jnp.exp(-V5 / 5.0))
        return beta

    def alpha_n(self, Vs):
        """K+ delayed rectifier activation rate"""
        V3 = Vs + 24.9
        alpha = -0.016 * V3 / (jnp.exp(-V3 / 5.0) - 1)
        return alpha

    def beta_n(self, Vs):
        """K+ delayed rectifier deactivation rate"""
        V4 = Vs + 40.0
        beta = 0.25 * jnp.exp(-V4 / 40.0)
        return beta

    def alpha_s(self, Vd):
        """Ca2+ channel activation rate"""
        alpha = 1.6 / (1 + jnp.exp(-0.072 * (Vd - 5.0)))
        return alpha

    def beta_s(self, Vd):
        """Ca2+ channel deactivation rate"""
        V6 = Vd + 8.9
        beta = 0.02 * V6 / (jnp.exp(V6 / 5.0) - 1.0)
        return beta

    def alpha_c(self, Vd):
        """Ca-dependent K+ channel activation rate"""
        V7 = Vd + 53.5
        V8 = Vd + 50.0
        return jnp.where(
            Vd <= -10, 0.0527 * jnp.exp(V8 / 11.0 - V7 / 27.0), 2 * jnp.exp(-V7 / 27.0)
        )

    def beta_c(self, Vd):
        """Ca-dependent K+ channel deactivation rate"""
        V7 = Vd + 53.5
        alpha_c_val = self.alpha_c(Vd)
        return jnp.where(Vd <= -10, 2.0 * jnp.exp(-V7 / 27.0) - alpha_c_val, 0.0)

    def alpha_q(self, Ca):
        """AHP K+ channel activation rate"""
        return jnp.minimum(0.00002 * Ca, 0.01)

    def beta_q(self, Ca):
        """AHP K+ channel deactivation rate"""
        return 0.001

    def chi(self, Ca):
        """Ca-dependent activation function"""
        return jnp.minimum(Ca / 250.0, 1.0)

    def m_inf(self, Vs):
        """Na+ channel steady-state activation"""
        return self.alpha_m(Vs) / (self.alpha_m(Vs) + self.beta_m(Vs))

    def vector_field(self, t, y, args):
        """Vector field for the Pinsky-Rinzel model."""
        Vs, Vd, n, h, s, c, q, Ca = y

        # Get stimulus parameters from args
        I_stim, stim_start, stim_end = args

        # Calculate ionic currents in somatic compartment
        I_leak_s = self.g_L * (Vs - self.E_L)
        I_Na = self.g_Na * self.m_inf(Vs) ** 2 * h * (Vs - self.E_Na)
        I_DR = self.g_DR * n * (Vs - self.E_K)
        I_ds = self.g_c * (Vd - Vs)  # Coupling current from dendrite to soma

        # Calculate ionic currents in dendritic compartment
        I_leak_d = self.g_L * (Vd - self.E_L)
        I_Ca = self.g_Ca * s**2 * (Vd - self.E_Ca)
        I_AHP = self.g_AHP * q * (Vd - self.E_K)
        I_C = self.g_C * c * self.chi(Ca) * (Vd - self.E_K)
        I_sd = -I_ds  # Coupling current from soma to dendrite

        # Apply stimulus using jax's where for conditional logic
        stimulus = jnp.where((t > stim_start) & (t < stim_end), I_stim / self.p, 0.0)

        # Differential equations
        dVsdt = (1.0 / self.C_m) * (-I_leak_s - I_Na - I_DR + I_ds / self.p + stimulus)
        dVddt = (1.0 / self.C_m) * (
            -I_leak_d - I_Ca - I_AHP - I_C + I_sd / (1 - self.p)
        )
        dhdt = self.alpha_h(Vs) * (1 - h) - self.beta_h(Vs) * h
        dndt = self.alpha_n(Vs) * (1 - n) - self.beta_n(Vs) * n
        dsdt = self.alpha_s(Vd) * (1 - s) - self.beta_s(Vd) * s
        dcdt = self.alpha_c(Vd) * (1 - c) - self.beta_c(Vd) * c
        dqdt = self.alpha_q(Ca) * (1 - q) - self.beta_q(Ca) * q
        dCadt = -0.13 * I_Ca - 0.075 * Ca

        return jnp.array([dVsdt, dVddt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt])

    def solve(
        self,
        t_dur,
        I_stim=0.0,
        stim_start=0.0,
        stim_end=0.0,
        dt=0.05,
        saveat=None,
        adjoint=RecursiveCheckpointAdjoint(checkpoints=10),
        max_steps=100000,
    ):
        term = ODETerm(self.vector_field)
        solver = Dopri5()
        if saveat is None:
            saveat = SaveAt(ts=jnp.linspace(0, t_dur, int(t_dur / dt) + 1))

        safe_dt = jnp.maximum(dt, 1e-6)
        # estimated_steps = int(t_dur / safe_dt * 2) + 100 # Example: double the estimated steps + buffer
        # # Use the provided max_steps argument, ensuring it's reasonable
        # final_max_steps = max(max_steps, estimated_steps)

        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_dur,
            y0=self.initial_state,
            args=(I_stim, stim_start, stim_end),
            dt0=dt,
            saveat=saveat,
            adjoint=adjoint,
            max_steps=max_steps,
        )
        return sol

    def get_spike_times(
        self,
        sol,
        threshold=-20.0,
    ):
        Vs = sol.ys[:, 0]
        times = sol.ts
        spikes = jnp.logical_and(Vs[:-1] < threshold, Vs[1:] >= threshold)
        spike_times = times[:-1][spikes]
        return spike_times



class MotoneuronModel(PinskyRinzel):
    """
    Motoneuron model based on the Pinsky-Rinzel model with adaptations for parameters
    specified in motoneuron.yaml.
    """

    def __init__(self, yaml_file=None, param_set="20230210_84", **kwargs):
        """
        Initialize the motoneuron model with parameters from a YAML file.

        Args:
            yaml_file: Path to the YAML file with motoneuron parameters
            param_set: Which parameter set to use from the YAML file ('20230210_84',
                      '20230210_8', 'test0', 'p0')
            **kwargs: Additional parameters to override those from the YAML file
        """
        # Default parameters (from Pinsky-Rinzel model)
        self.params = {
            "g_c": 2.1,  # Coupling conductance [mS cm**-2]
            "C_m": 3.0,  # Membrane capacitance [uF cm**-2]
            "p": 0.5,  # Proportion of membrane area for soma
            "g_L_soma": 0.1,  # Leak conductance in soma [mS cm**-2]
            "g_L_dend": 0.1,  # Leak conductance in dendrite [mS cm**-2]
            "g_Na": 30.0,  # Sodium conductance [mS cm**-2]
            "g_DR": 15.0,  # Delayed rectifier K+ conductance [mS cm**-2]
            "g_Ca": 10.0,  # Calcium conductance [mS cm**-2]
            "g_AHP": 0.8,  # After-hyperpolarization K+ conductance [mS cm**-2]
            "g_C": 15.0,  # Ca-dependent K+ conductance [mS cm**-2]
            "E_L": -68.0,  # Leak reversal potential [mV]
            "E_Na": 60.0,  # Sodium reversal potential [mV]
            "E_K": -75.0,  # Potassium reversal potential [mV]
            "E_Ca": 80.0,  # Calcium reversal potential [mV]
            "f_Caconc": 0.004,  # Calcium dynamics parameter
            "alpha_Caconc": 1.0,  # Calcium dynamics parameter
            "kCa_Caconc": 8.0,  # Calcium dynamics parameter
        }

        # Load parameters from YAML file if provided
        if yaml_file is not None:
            self.load_yaml_params(yaml_file, param_set)

        # Override parameters with any provided in kwargs
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # Initialize the parent class with our parameters
        super().__init__(
            g_c=self.params["g_c"],
            C_m=self.params["C_m"],
            p=self.params["p"],
            g_L=self.params["g_L_soma"],  # Use soma leak conductance for now
            g_Na=self.params["g_Na"],
            g_DR=self.params["g_DR"],
            g_Ca=self.params["g_Ca"],
            g_AHP=self.params["g_AHP"],
            g_C=self.params["g_C"],
            E_L=self.params["E_L"],
            E_Na=self.params["E_Na"],
            E_K=self.params["E_K"],
            E_Ca=self.params["E_Ca"],
        )

        # Store any additional parameters specific to motoneuron model
        self.f_Caconc = self.params["f_Caconc"]
        self.alpha_Caconc = self.params["alpha_Caconc"]
        self.kCa_Caconc = self.params["kCa_Caconc"]

        # Store different leak conductances for soma and dendrite
        self.g_L_soma = self.params["g_L_soma"]
        self.g_L_dend = self.params["g_L_dend"]

    def load_yaml_params(self, yaml_file, param_set):
        """Load parameters from a YAML file."""
        if yaml is None:
            print(
                f"Cannot load parameters from {yaml_file} because PyYAML is not installed."
            )
            return

        try:
            with open(yaml_file, "r") as f:
                yaml_data = yaml.safe_load(f)

            # Load default parameters from yaml file
            if "Parameters" in yaml_data:
                for key, value in yaml_data["Parameters"].items():
                    if key in self.params_map:
                        self.params[self.params_map[key]] = value

            # Load specific parameter set
            if "best" in yaml_data and param_set in yaml_data["best"]:
                best_params = yaml_data["best"][param_set]
                for key, value in best_params.items():
                    if key in self.params_map:
                        self.params[self.params_map[key]] = value
        except Exception as e:
            print(f"Error loading parameters from {yaml_file}: {e}")
            print("Continuing with default parameters...")

    @property
    def params_map(self):
        """Map from YAML parameter names to model parameter names."""
        return {
            "global_cm": "C_m",
            "pp": "p",
            "e_pas": "E_L",
            "gc": "g_c",
            "soma_gmax_Na": "g_Na",
            "soma_gmax_K": "g_DR",
            "soma_gmax_KCa": "g_C",
            "soma_gmax_CaN": "g_Ca",
            "soma_g_pas": "g_L_soma",
            "dend_g_pas": "g_L_dend",
            "dend_gmax_KCa": "g_AHP",
            "soma_f_Caconc": "f_Caconc",
            "soma_alpha_Caconc": "alpha_Caconc",
            "soma_kCa_Caconc": "kCa_Caconc",
        }

    def vector_field(self, t, y, args):
        """
        Modified vector field for the motoneuron model.
        Uses separate leak conductances for soma and dendrite.
        """
        Vs, Vd, n, h, s, c, q, Ca = y

        # Get stimulus parameters from args
        I_stim, stim_start, stim_end = args

        # Calculate ionic currents in somatic compartment
        I_leak_s = self.g_L_soma * (Vs - self.E_L)
        I_Na = self.g_Na * self.m_inf(Vs) ** 2 * h * (Vs - self.E_Na)
        I_DR = self.g_DR * n * (Vs - self.E_K)
        I_ds = self.g_c * (Vd - Vs)  # Coupling current from dendrite to soma

        # Calculate ionic currents in dendritic compartment
        I_leak_d = self.g_L_dend * (Vd - self.E_L)  # Use dendrite-specific leak
        I_Ca = self.g_Ca * s**2 * (Vd - self.E_Ca)
        I_AHP = self.g_AHP * q * (Vd - self.E_K)
        I_C = self.g_C * c * self.chi(Ca) * (Vd - self.E_K)
        I_sd = -I_ds  # Coupling current from soma to dendrite

        # Apply stimulus using jax's where for conditional logic
        stimulus = jnp.where((t > stim_start) & (t < stim_end), I_stim / self.p, 0.0)

        # Differential equations
        dVsdt = (1.0 / self.C_m) * (-I_leak_s - I_Na - I_DR + I_ds / self.p + stimulus)
        dVddt = (1.0 / self.C_m) * (
            -I_leak_d - I_Ca - I_AHP - I_C + I_sd / (1 - self.p)
        )
        dhdt = self.alpha_h(Vs) * (1 - h) - self.beta_h(Vs) * h
        dndt = self.alpha_n(Vs) * (1 - n) - self.beta_n(Vs) * n
        dsdt = self.alpha_s(Vd) * (1 - s) - self.beta_s(Vd) * s
        dcdt = self.alpha_c(Vd) * (1 - c) - self.beta_c(Vd) * c
        dqdt = self.alpha_q(Ca) * (1 - q) - self.beta_q(Ca) * q

        # Custom calcium dynamics using motoneuron parameters
        dCadt = -self.f_Caconc * I_Ca - self.alpha_Caconc * Ca / self.kCa_Caconc

        return jnp.array([dVsdt, dVddt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt])
