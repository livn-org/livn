: Refractory spike-source filter for biophysical cell spike detection.
:
: An ARTIFICIAL_CELL that re-emits incoming events while enforcing an
: absolute refractory window ``tref`` (ms). Used to wrap the somatic
: voltage-threshold NetCon of biophysical cells so that bursting models
: do not register multiple "spikes" within a single biological
: action potential / plateau
:
: (see livn.backend.neuron.Env._wrap_spike_sources):
:
:   detector = h.NetCon(soma(0.5)._ref_v, filter, sec=soma)
:   detector.threshold = V_threshold
:   output   = h.NetCon(filter, None)
:   pc.cell(gid, output, 1)        # gid spike train = filtered output
:   pc.spike_record(gid, tvec, idvec)

NEURON {
    ARTIFICIAL_CELL SpikeFilter
    RANGE tref, t_last
}

PARAMETER {
    tref = 2 (ms)   : absolute refractory period
}

ASSIGNED {
    t_last (ms)
}

INITIAL {
    t_last = -1e9
}

NET_RECEIVE (w) {
    if (t - t_last >= tref) {
        t_last = t
        net_event(t)
    }
}
