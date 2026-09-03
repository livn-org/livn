/* librcsd: the ReducedCalciumSomaDendrite cell models and their network, in C.
 *
 * The numerics follow NEURON's fixed-step, staggered Crank-Nicolson scheme
 * (`secondorder = 2`) with cnexp state updates, so a run under this library
 * reproduces the NEURON backend step for step. Geometry is NEURON's: every
 * section is `nseg` centre nodes plus a zero-area node at its 1 end, and a
 * root section carries an extra zero-area node at its 0 end.
 *
 * The Python layer (`livn.backend.native`) owns the graph reading, parameter
 * resolution and stimulus rendering; this library owns the timestep.
 */
#ifndef RCSD_H
#define RCSD_H

#ifdef _WIN32
#define RCSD_API __declspec(dllexport)
#else
#define RCSD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RCSD_VERSION "0.1.0"

typedef struct RCSDSim RCSDSim;

/* --- section kinds (informational; the mechanism mask decides the physics) */
enum { RCSD_SEC_SOMA = 0, RCSD_SEC_DEND = 1, RCSD_SEC_AXON = 2 };

/* --- density mechanisms, as a bitmask per section ---------------------- */
enum {
    RCSD_M_PAS = 1 << 0,
    RCSD_M_CONSTANT = 1 << 1,
    RCSD_M_NA_CONC = 1 << 2,
    RCSD_M_K_CONC = 1 << 3,
    RCSD_M_CA_CONC = 1 << 4,
    RCSD_M_NAS = 1 << 5,
    RCSD_M_KDR = 1 << 6,
    RCSD_M_CAN = 1 << 7,
    RCSD_M_CAL = 1 << 8,
    RCSD_M_KCA = 1 << 9,
    RCSD_M_KA_V1IN = 1 << 10
};

/* --- per-node parameters (RANGE variables), addressed by id ------------ */
enum {
    RCSD_P_CM = 0,
    RCSD_P_G_PAS,
    RCSD_P_E_PAS,
    RCSD_P_IC,          /* constant.ic (mA/cm2) */
    RCSD_P_GMAX_NAS,
    RCSD_P_VHALF_NAS,
    RCSD_P_SLOPE_NAS,
    RCSD_P_GMAX_KDR,
    RCSD_P_GMAX_CAN,
    RCSD_P_GMAX_CAL,
    RCSD_P_GMAX_KCA,
    RCSD_P_KD_KCA,
    RCSD_P_GMAX_KA,
    RCSD_P_F_CA,
    RCSD_P_ALPHA_CA,
    RCSD_P_KCA_CA,
    RCSD_P_CAI0,
    RCSD_P_D_NA,
    RCSD_P_BETA_NA,
    RCSD_P_NAI0,
    RCSD_P_NAO0,
    RCSD_P_D_K,
    RCSD_P_BETA_K,
    RCSD_P_KI0,
    RCSD_P_KO0,
    RCSD_P_CAO,
    RCSD_NPARAM
};

/* --- per-node states, addressed by id (for reading back) --------------- */
enum {
    RCSD_S_V = 0,
    RCSD_S_H,
    RCSD_S_N,
    RCSD_S_MN,
    RCSD_S_HN,
    RCSD_S_ML,
    RCSD_S_A,
    RCSD_S_B,
    RCSD_S_CAI,
    RCSD_S_NAI,
    RCSD_S_NAO,
    RCSD_S_KI,
    RCSD_S_KO,
    RCSD_S_ENA,
    RCSD_S_EK,
    RCSD_S_INA,
    RCSD_S_IK,
    RCSD_S_ICA,
    RCSD_S_IPAS,
    RCSD_S_IREST,
    RCSD_S_IMEM,
    RCSD_NSTATE
};

/* --- synapse mechanisms -------------------------------------------------- */
enum {
    RCSD_SYN_LINEXP2 = 0,   /* LinExp2Syn */
    RCSD_SYN_NMDA,          /* LinExp2SynNMDA */
    RCSD_SYN_DEP,           /* DepLinExp2Syn */
    RCSD_SYN_STDP,          /* StdpLinExp2Syn */
    RCSD_SYN_STDP_NMDA,     /* StdpLinExp2SynNMDA */
    RCSD_SYN_STDP_INH       /* StdpLinExp2SynInh */
};

/* per-site parameters: one row of RCSD_SP_N doubles per site */
enum {
    RCSD_SP_TAU_RISE = 0,
    RCSD_SP_TAU_DECAY,
    RCSD_SP_E,
    RCSD_SP_MG,
    RCSD_SP_KD,
    RCSD_SP_GAMMA,
    RCSD_SP_VSHIFT,
    RCSD_SP_U,
    RCSD_SP_TAU_REC,
    RCSD_SP_PLASTICITY_ON,
    RCSD_SP_W_INIT,
    RCSD_SP_A_LTP,
    RCSD_SP_A_LTD,
    RCSD_SP_THETA_LTP,
    RCSD_SP_THETA_LTD,
    RCSD_SP_LTP_SIGMOID_HALF,
    RCSD_SP_LTD_SIGMOID_HALF,
    RCSD_SP_LEARNING_SLOPE,
    RCSD_SP_LEARNING_TAU,
    RCSD_SP_W_MAX,
    RCSD_SP_W_MIN,
    RCSD_SP_N
};

/* per-site states: one row of RCSD_SS_N doubles per site */
enum {
    RCSD_SS_A = 0,
    RCSD_SS_B,
    RCSD_SS_LEARNING_W,
    RCSD_SS_LEARN_INT,
    RCSD_SS_LTD,
    RCSD_SS_LTP,
    RCSD_SS_W,
    RCSD_SS_FACTOR,
    RCSD_SS_G,
    RCSD_SS_I,
    RCSD_SS_N
};

/* NetCon weight slots per connection: weight, g_unit, w_plastic|R, last_int|tlast */
#define RCSD_NWEIGHT 4

/* --- stimulus modes ---------------------------------------------------- */
enum {
    RCSD_STIM_EXTRACELLULAR = 0,
    RCSD_STIM_CURRENT = 1,
    RCSD_STIM_CURRENT_DENSITY = 2,
    RCSD_STIM_PHOTON_FLUX = 3,
    RCSD_STIM_N
};

/* --- run() return codes ------------------------------------------------- */
enum {
    RCSD_OK = 0,
    RCSD_NEED_STIMULUS = 1,  /* stopped early: a stimulus window has to be refilled */
    RCSD_ERROR = -1
};

/* --- lifecycle ---------------------------------------------------------- */
RCSD_API const char* rcsd_version(void);
RCSD_API const char* rcsd_last_error(void);
RCSD_API RCSDSim* rcsd_create(double celsius, double v_init);
RCSD_API void rcsd_destroy(RCSDSim* sim);

/* --- building cells ------------------------------------------------------ */
RCSD_API int rcsd_add_cell(RCSDSim* sim, int gid, int population, double v_threshold,
                           double v_hold, double tref);
/* A section of `nseg` segments. `parent_section` is a section index of the
 * same cell, or -1 for the root; `parent_x` is where on the parent it
 * attaches (0 or 1 for the chain). Returns the section index. */
RCSD_API int rcsd_add_section(RCSDSim* sim, int cell, int kind, int nseg, double L,
                              double diam, double Ra, double cm, unsigned mechanisms,
                              int parent_section, double parent_x);
RCSD_API int rcsd_section_set(RCSDSim* sim, int section, int param, double value);
RCSD_API double rcsd_section_get(RCSDSim* sim, int section, int param);
RCSD_API int rcsd_section_geometry(RCSDSim* sim, int section, double L, double diam,
                                   double Ra);
RCSD_API int rcsd_section_info(RCSDSim* sim, int section, int* nseg, double* L,
                               double* diam, double* Ra, unsigned* mechanisms);
RCSD_API int rcsd_cell_count(RCSDSim* sim);
RCSD_API int rcsd_node_count(RCSDSim* sim);
RCSD_API int rcsd_cell_section_count(RCSDSim* sim, int cell);
RCSD_API int rcsd_cell_section(RCSDSim* sim, int cell, int index);
/* the centre node a section position maps to (NEURON's segment containing x) */
RCSD_API int rcsd_section_node(RCSDSim* sim, int section, double x);
RCSD_API int rcsd_cell_set(RCSDSim* sim, int cell, double v_threshold, double v_hold,
                           double tref);
RCSD_API double rcsd_node_state(RCSDSim* sim, int node, int state);
RCSD_API double rcsd_node_area(RCSDSim* sim, int node);

/* --- synapses -------------------------------------------------------------- */
RCSD_API int rcsd_add_synapse(RCSDSim* sim, int cell, int section, double x, int kind);
RCSD_API int rcsd_add_input(RCSDSim* sim, int gid);
RCSD_API int rcsd_set_input_spikes(RCSDSim* sim, int input, const double* times, int n);
/* `source` >= 0 is a cell index, < 0 is an input: -(input + 1). The
 * delay is the physical one; the 2*dt floor is applied when dt is known. */
RCSD_API int rcsd_add_connections(RCSDSim* sim, int n, const int* source, const int* site,
                                  const double* delay, const double* weights);
RCSD_API int rcsd_synapse_count(RCSDSim* sim);
RCSD_API int rcsd_connection_count(RCSDSim* sim);
/* Column-major: parameter p of site i is at [p * stride + i], with the
 * stride from rcsd_synapse_stride() (it changes while sites are added). */
RCSD_API double* rcsd_synapse_params(RCSDSim* sim);   /* [RCSD_SP_N][stride] */
RCSD_API double* rcsd_synapse_states(RCSDSim* sim);   /* [RCSD_SS_N][stride] */
RCSD_API int rcsd_synapse_stride(RCSDSim* sim);
RCSD_API double* rcsd_connection_weights(RCSDSim* sim); /* [n_conn][RCSD_NWEIGHT] */
RCSD_API int rcsd_synapse_node(RCSDSim* sim, int site);
/* recompute the normalisation `factor` of a site from its time constants
 * (NEURON does this in INITIAL, so call it after changing tau_rise/tau_decay) */
RCSD_API int rcsd_synapse_refresh(RCSDSim* sim, int site);

/* --- noise (Gfluct3) ------------------------------------------------------- */
RCSD_API int rcsd_set_noise(RCSDSim* sim, int cell, int section, double g_e0, double g_i0,
                            double std_e, double std_i, double tau_e, double tau_i,
                            double E_e, double E_i, double h, int on);
RCSD_API int rcsd_set_noise_stream(RCSDSim* sim, int cell, int section, unsigned id1,
                                   unsigned id2, unsigned id3);
RCSD_API int rcsd_noise_count(RCSDSim* sim);
/* draw from a Random123 stream, for testing the generator against NEURON */
RCSD_API double rcsd_random123_normal(unsigned id1, unsigned id2, unsigned id3,
                                      unsigned seq, int index);

/* --- opsin (RhO3c) --------------------------------------------------------- */
RCSD_API int rcsd_add_opsin(RCSDSim* sim, int cell, int section, double x);
RCSD_API int rcsd_opsin_set(RCSDSim* sim, int opsin, double g0, double E, double v0,
                            double k_a, double k_r, double p, double q, double Gd,
                            double Gr0, double phi_m);
RCSD_API int rcsd_opsin_count(RCSDSim* sim);
RCSD_API int rcsd_opsin_state(RCSDSim* sim, int opsin, double* C, double* O, double* phi);

/* --- stimulus --------------------------------------------------------------- */
/* Declare the rows a mode is delivered on. For EXTRACELLULAR a row is a
 * (cell, section) whose field is sampled; for the others `section` names
 * the section the current or flux lands on. Rows persist across windows;
 * calling again replaces them. */
RCSD_API int rcsd_set_stimulus_rows(RCSDSim* sim, int mode, int n_rows, const int* cell,
                                    const int* section, double stim_dt);
/* Hand over samples [first, first + n_samples) of the stimulus, laid out
 * [n_samples][n_rows]. `extent` is the total number of samples the stimulus
 * has; samples at or beyond it read as zero without a refill. */
RCSD_API int rcsd_set_stimulus_window(RCSDSim* sim, int mode, long first, int n_samples,
                                      const double* values, long extent);
RCSD_API int rcsd_clear_stimulus(RCSDSim* sim, int mode);
/* how many of the junctions the extracellular field drives were dropped
 * because one of their sections has no row */
RCSD_API int rcsd_extracellular_junctions(RCSDSim* sim, int* driven, int* dropped);
/* the sample index the next step would need for `mode`, or -1 */
RCSD_API long rcsd_stimulus_needed(RCSDSim* sim, int mode);

/* --- recording -------------------------------------------------------------- */
RCSD_API int rcsd_record_spikes(RCSDSim* sim, int cell);
RCSD_API int rcsd_record_voltage(RCSDSim* sim, int cell, int section, double dt);
RCSD_API int rcsd_record_current(RCSDSim* sim, int cell, int section, double dt);
RCSD_API int rcsd_clear_recordings(RCSDSim* sim);
RCSD_API int rcsd_spike_count(RCSDSim* sim);
RCSD_API const int* rcsd_spike_cells(RCSDSim* sim);
RCSD_API const double* rcsd_spike_times(RCSDSim* sim);
RCSD_API int rcsd_voltage_record_count(RCSDSim* sim);
RCSD_API int rcsd_voltage_record_length(RCSDSim* sim, int index);
RCSD_API const double* rcsd_voltage_record(RCSDSim* sim, int index);
RCSD_API int rcsd_current_record_count(RCSDSim* sim);
RCSD_API int rcsd_current_record_length(RCSDSim* sim, int index);
RCSD_API const double* rcsd_current_record(RCSDSim* sim, int index);

/* --- initialisation and running ---------------------------------------------- */
RCSD_API int rcsd_set_dt(RCSDSim* sim, double dt);
RCSD_API double rcsd_dt(RCSDSim* sim);
RCSD_API int rcsd_set_v_init(RCSDSim* sim, double v_init);
/* Build the coefficients, pin the resting currents at each cell's hold
 * potential, then initialise every state at v_init. Resets time to zero. */
RCSD_API int rcsd_init(RCSDSim* sim);
/* Re-pin the resting currents without disturbing a running simulation. */
RCSD_API int rcsd_pin_resting(RCSDSim* sim);
/* Advance up to n_steps. Returns RCSD_OK, RCSD_NEED_STIMULUS or RCSD_ERROR;
 * `*done` receives the number of steps taken. */
RCSD_API int rcsd_run(RCSDSim* sim, long n_steps, long* done);
RCSD_API long rcsd_step(RCSDSim* sim);
RCSD_API double rcsd_time(RCSDSim* sim);
RCSD_API int rcsd_initialized(RCSDSim* sim);

#ifdef __cplusplus
}
#endif

#endif
