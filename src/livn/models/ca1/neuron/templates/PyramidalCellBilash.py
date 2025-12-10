import sys
from neuron import h
from collections import defaultdict
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Dict, Any, Type, TypeVar, Optional
from typing_extensions import get_type_hints
import yaml

T = TypeVar("T")


def create_from_dict(cls: Type[T], data: Dict[str, Any], missing="error") -> T:
    """
    Create a dataclass instance from a dictionary, handling nested dataclasses.

    Args:
        cls: The dataclass type to create
        data: Dictionary containing parameter values
        missing: behavior when a data field is missing in the dictionary:
         - 'error': raise error
         - 'skip': skip field

    Returns:
        Instance of the dataclass populated with values from the dictionary

    Raises:
        ValueError: If required parameters are missing or values can't be converted
    """
    if not data:
        return cls()

    field_types = get_type_hints(cls)
    kwargs = {}

    for field_name, field_type in field_types.items():
        # Skip if field not in data and has default value
        if field_name not in data and hasattr(cls, field_name):
            if missing == "error":
                raise RuntimeError(f"field {field_name} not found in dictionary")
            else:
                continue

        value = data.get(field_name)

        # Handle nested dataclasses
        if hasattr(field_type, "__dataclass_fields__"):
            nested_data = value if value else {}
            kwargs[field_name] = create_from_dict(field_type, nested_data)
        else:
            if value is not None:
                try:
                    kwargs[field_name] = field_type(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Cannot convert value '{value}' to {field_type} for parameter '{field_name}': {str(e)}"
                    )

    return cls(**kwargs)


@dataclass
class SomaParameters:
    g_pas: float = 5e-8  # S/cm2
    e_pas: float = -70  # mV
    gbar_na: float = 3.5e-2  # S/cm2
    gkdr: float = 1.4e-3  # S/cm2
    gka_prox: float = 7.5e-3  # S/cm2
    gbar_km: float = 1.66e-2  # S/cm2
    gh: float = 1.5e-5  # S/cm2
    eh: float = -30  # mV
    gcal: float = 7e-3  # S/cm2
    gcar: float = 3e-3  # S/cm2
    gcat: float = 5e-5  # S/cm2
    gkca: float = 9.075e-2  # S/cm2
    gkahp: float = 5e-4  # S/cm2
    cm: float = 1.0  # µF/cm2
    Ra: float = 300  # Ohm cm
    ic: float = 0.0  # mA/cm2


@dataclass
class AxonParameters:
    g_pas: float = 5e-8
    e_pas: float = -70  # mV
    gbar_na: float = 1.5e0
    gkdr: float = 1e-1
    gbar_km: float = 3e-6
    cm: float = 1.0
    Ra: float = 300
    ic: float = 0.0  # mA/cm2


@dataclass
class AISParameters:
    g_pas: float = 5e-8
    e_pas: float = -70  # mV
    gbar_na: float = 1.5e0
    gkdr: float = 1e-1
    gbar_km: float = 3e-6
    cm: float = 1.0
    Ra: float = 300
    ic: float = 0.0  # mA/cm2


@dataclass
class HillockParameters:
    g_pas: float = 5e-8
    e_pas: float = -70  # mV
    gbar_na: float = 1.5e0
    gkdr: float = 1e-1
    gbar_km: float = 3e-6
    cm: float = 1.0
    Ra: float = 300
    ic: float = 0.0  # mA/cm2


@dataclass
class ApicalTrunkParameters:
    g_pas: float = 5.26e-8
    e_pas: float = -70  # mV
    gbar_na: float = 1.4e-2
    ar2_na: float = 0.9
    gkdr: float = 1.74e-3
    gka: float = 1.8e-3
    gbar_km: float = 6e-4
    gh: float = 1.8e-5
    eh: float = -30  # mV
    gcal: float = 3.16e-6
    gcar: float = 3e-4
    gcat: float = 4e-4
    gkca: float = 6.6e-3
    gkahp: float = 5e-4
    cm: float = 2.2
    Ra: float = 285
    ic: float = 0.0  # mA/cm2


@dataclass
class ApicalParameters:
    g_pas: float = 8.3e-7
    e_pas: float = -70  # mV
    gbar_na: float = 1.5e-3
    ar2_na: float = 0.95
    gkdr: float = 1.9e-4
    gka_dist: float = 4.86e-2
    gbar_km: float = 1.2e-3
    gh: float = 1.2e-4
    eh: float = -30  # mV
    gcal: float = 3.16e-6
    gkca: float = 4.125e-3
    gkahp: float = 5e-5
    cm: float = 3.0
    Ra: float = 150
    ic: float = 0.0  # mA/cm2


@dataclass
class BasalParameters:
    g_pas: float = 5.55e-7
    e_pas: float = -70  # mV
    gbar_na: float = 7e-3
    ar2_na: float = 1.0
    gkdr: float = 8.6e-4
    gka_prox: float = 7.5e-3
    gbar_km: float = 6e-4
    gh: float = 1.5e-5
    eh: float = -30  # mV
    gkca: float = 1.65e-2
    gkahp: float = 5e-4
    cm: float = 3.0
    Ra: float = 150
    ic: float = 0.0  # mA/cm2


@dataclass
class NeuronParameters:
    """Complete set of parameters for all compartments of the neuron model."""

    soma: SomaParameters = field(default_factory=SomaParameters)
    axon: AxonParameters = field(default_factory=AxonParameters)
    ais: AISParameters = field(default_factory=AISParameters)
    hillock: HillockParameters = field(default_factory=HillockParameters)
    radTprox: ApicalTrunkParameters = field(default_factory=ApicalTrunkParameters)
    radTmed: ApicalTrunkParameters = field(default_factory=ApicalTrunkParameters)
    radTdist: ApicalTrunkParameters = field(default_factory=ApicalTrunkParameters)
    apical_tuft: ApicalParameters = field(default_factory=ApicalParameters)
    apical_oblique: ApicalParameters = field(default_factory=ApicalParameters)
    basal: BasalParameters = field(default_factory=BasalParameters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuronParameters":
        """
        Create a NeuronParameters instance from a dictionary.

        Args:
            data: Dictionary containing parameter values organized by compartment

        Returns:
            NeuronParameters instance with values from the dictionary

        Example:
            params = NeuronParameters.from_dict({
                'soma': {'gkabar_kap': 0.008, 'gbar_kmb': 0.002},
                'axon': {'gbar_nax': 0.04}
            })
        """
        return create_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the NeuronParameters instance to a dictionary.

        Returns:
            Dictionary containing all parameter values organized by compartment
        """
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if hasattr(value, "__dataclass_fields__"):
                result[field.name] = {
                    nested_field.name: getattr(value, nested_field.name)
                    for nested_field in fields(value)
                }
            else:
                result[field.name] = value
        return result

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update parameters from a dictionary without creating a new instance.

        Args:
            data: Dictionary containing parameter values to update

        Example:
            params.update_from_dict({
                'soma': {'gkabar_kap': 0.008},
                'axon': {'gbar_nax': 0.04}
            })
        """
        new_params = self.from_dict(data)
        for field in fields(self):
            setattr(self, field.name, getattr(new_params, field.name))

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "NeuronParameters":
        """
        Create a NeuronParameters instance from a YAML file.

        Args:
            yaml_path: Path to the YAML file

        Returns:
            NeuronParameters instance with values from the YAML file

        Raises:
            yaml.YAMLError: If YAML parsing fails
            FileNotFoundError: If the file doesn't exist
        """
        yaml_path = Path(yaml_path)
        with yaml_path.open("r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "NeuronParameters":
        """
        Create a NeuronParameters instance from a YAML string.

        Args:
            yaml_string: String containing YAML data

        Returns:
            NeuronParameters instance with values from the YAML string

        Raises:
            yaml.YAMLError: If YAML parsing fails
        """
        data = yaml.safe_load(yaml_string)
        return cls.from_dict(data)

    def to_yaml(self, yaml_path: Path | str, **kwargs) -> None:
        """
        Save parameters to a YAML file.

        Args:
            yaml_path: Path where to save the YAML file
            **kwargs: Additional arguments passed to yaml.dump()

        Default kwargs include:
            default_flow_style=False (for better readability)
            sort_keys=False (to maintain parameter order)
        """
        yaml_path = Path(yaml_path)
        data = self.to_dict()

        # Default YAML formatting options
        kwargs.setdefault("default_flow_style", False)
        kwargs.setdefault("sort_keys", False)

        with yaml_path.open("w") as f:
            yaml.dump(data, f, **kwargs)

    def to_yaml_string(self, **kwargs) -> str:
        """
        Convert parameters to a YAML string.

        Args:
            **kwargs: Additional arguments passed to yaml.dump()

        Returns:
            String containing YAML representation of parameters
        """
        data = self.to_dict()
        kwargs.setdefault("default_flow_style", False)
        kwargs.setdefault("sort_keys", False)
        return yaml.dump(data, **kwargs)

    def apply_to_model(self, model):
        """
        Apply all parameters to the given model instance.

        Args:
            model: The neuron model instance to apply parameters to
        """
        for compartment_name, compartment in self.__dict__.items():
            for param_name, value in compartment.__dict__.items():
                full_param_name = f"{compartment_name}_{param_name}"
                setattr(model, full_param_name, value)


class PyramidalCell:
    def __init__(self, params=None):
        # Initialize parameters
        self.set_default_parameters()
        if params is not None:
            self.set_parameters(params)

        # Create sections
        self.soma = h.Section(name="soma", cell=self)
        self.radTprox = h.Section(name="radTprox", cell=self)
        self.radTmed = h.Section(name="radTmed", cell=self)
        self.radTdist1 = h.Section(name="radTdist1", cell=self)
        self.radTdist2 = h.Section(name="radTdist2", cell=self)
        self.radTdist3 = h.Section(name="radTdist3", cell=self)
        self.lm_thick1 = h.Section(name="lm_thick1", cell=self)
        self.lm_medium1 = h.Section(name="lm_medium1", cell=self)
        self.lm_thin1a = h.Section(name="lm_thin1a", cell=self)
        self.lm_thin1b = h.Section(name="lm_thin1b", cell=self)
        self.lm_thick2 = h.Section(name="lm_thick2", cell=self)
        self.lm_medium2 = h.Section(name="lm_medium2", cell=self)
        self.lm_thin2a = h.Section(name="lm_thin2a", cell=self)
        self.lm_thin2b = h.Section(name="lm_thin2b", cell=self)
        self.rad_thick1 = h.Section(name="rad_thick1", cell=self)
        self.rad_medium1 = h.Section(name="rad_medium1", cell=self)
        self.rad_thin1a = h.Section(name="rad_thin1a", cell=self)
        self.rad_thin1b = h.Section(name="rad_thin1b", cell=self)
        self.rad_thick2 = h.Section(name="rad_thick2", cell=self)
        self.rad_medium2 = h.Section(name="rad_medium2", cell=self)
        self.rad_thin2a = h.Section(name="rad_thin2a", cell=self)
        self.rad_thin2b = h.Section(name="rad_thin2b", cell=self)
        self.oriprox1 = h.Section(name="oriprox1", cell=self)
        self.oridist1a = h.Section(name="oridist1a", cell=self)
        self.oridist1b = h.Section(name="oridist1b", cell=self)
        self.oriprox2 = h.Section(name="oriprox2", cell=self)
        self.oridist2a = h.Section(name="oridist2a", cell=self)
        self.oridist2b = h.Section(name="oridist2b", cell=self)
        self.axon = h.Section(name="axon", cell=self)
        self.hillock = h.Section(name="hillock", cell=self)
        self.ais = h.Section(name="ais", cell=self)

        # Initialize all the section lists
        self.all = h.SectionList()
        self.soma_list = h.SectionList()
        self.apical_list = h.SectionList()
        self.basal_list = h.SectionList()
        self.axon_list = h.SectionList()
        self.hillock_list = h.SectionList()
        self.ais_list = h.SectionList()
        self.oblique_list = h.SectionList()
        self.tuft_list = h.SectionList()
        self.trunk_list = h.SectionList()

        # Layer-specific section lists
        self.soma_SP_list = h.SectionList()
        self.apical_SR_list = h.SectionList()
        self.apical_SLM_list = h.SectionList()
        self.basal_SO_list = h.SectionList()
        self.axon_SR_list = h.SectionList()
        self.ais_SP_list = h.SectionList()
        self.hillock_SP_list = h.SectionList()

        # Initialize the cell
        self.init()

    def set_default_parameters(self):
        self.params = NeuronParameters()

    def set_parameters(self, params):
        # Update existing instance
        self.params.update_from_dict(params)

        self.params.apply_to_model(self)

    def init(self):
        self.topol()
        self.subsets()
        self.geom()
        self.geom_nseg()
        self.biophys()

    def init_ic(self, v_init):
        h.finitialize(v_init)
        for sec in self.soma_list:
            for seg in sec:
                seg.constant.ic += -(
                    seg.ina + seg.ik + seg.ica + seg.h_mech.ih + seg.pas.i
                )
        for sec in self.ais_list:
            for seg in sec:
                seg.constant.ic += -(seg.ina + seg.ik + seg.pas.i)
        for sec in self.hillock_list:
            for seg in sec:
                seg.constant.ic += -(seg.ina + seg.ik + seg.pas.i)
        for sec in self.axon_list:
            for seg in sec:
                seg.constant.ic += -(seg.ina + seg.ik + seg.pas.i)
        for sec in self.apical_list:
            for seg in sec:
                seg.constant.ic += -(
                    seg.ina + seg.ik + seg.ica + seg.ica + seg.h_mech.ih + seg.pas.i
                )
        for sec in self.basal_list:
            for seg in sec:
                seg.constant.ic += -(
                    seg.ina + seg.ik + seg.ica + seg.h_mech.ih + seg.pas.i
                )

    def topol(self):
        # Connect sections
        self.radTprox.connect(self.soma(1))
        self.radTmed.connect(self.radTprox(1))
        self.radTdist1.connect(self.radTmed(1))
        self.radTdist2.connect(self.radTdist1(1))
        self.radTdist3.connect(self.radTdist2(1))
        # Apical oblique tree
        # Right
        self.rad_thick1.connect(self.radTmed(1))
        self.rad_medium1.connect(self.rad_thick1(1))
        self.rad_thin1a.connect(self.rad_medium1(1))
        self.rad_thin1b.connect(self.rad_medium1(1))
        # Left
        self.rad_thick2.connect(self.radTmed(1))
        self.rad_medium2.connect(self.rad_thick2(1))
        self.rad_thin2a.connect(self.rad_medium2(1))
        self.rad_thin2b.connect(self.rad_medium2(1))
        # Apical tuft tree
        # Right
        self.lm_thick1.connect(self.radTdist3(1))
        self.lm_medium1.connect(self.lm_thick1(1))
        self.lm_thin1a.connect(self.lm_medium1(1))
        self.lm_thin1b.connect(self.lm_medium1(1))
        # Left
        self.lm_thick2.connect(self.radTdist3(1))
        self.lm_medium2.connect(self.lm_thick2(1))
        self.lm_thin2a.connect(self.lm_medium2(1))
        self.lm_thin2b.connect(self.lm_medium2(1))
        # Basal tree
        # Right
        self.oriprox1.connect(self.soma(0))
        self.oridist1a.connect(self.oriprox1(1))
        self.oridist1b.connect(self.oriprox1(1))
        # Left
        self.oriprox2.connect(self.soma(0))
        self.oridist2a.connect(self.oriprox2(1))
        self.oridist2b.connect(self.oriprox2(1))
        # Axon
        self.hillock.connect(self.soma(0))
        self.ais.connect(self.hillock(1))
        self.axon.connect(self.ais(1))

    def subsets(self):
        # Add sections to all list
        for sec in self.soma.wholetree():
            self.all.append(sec=sec)

        # Initialize indices for different regions
        self.soma_SP_index = h.Vector()
        self.apical_SR_index = h.Vector()
        self.apical_SLM_index = h.Vector()
        self.basal_SO_index = h.Vector()
        self.axon_SR_index = h.Vector()
        self.hillock_SP_index = h.Vector()
        self.ais_SP_index = h.Vector()

        # Add sections to specific lists with indices
        section_index = 0

        # Soma
        self.soma_list.append(sec=self.soma)
        self.soma_SP_list.append(sec=self.soma)
        self.soma_SP_index.append(section_index)
        section_index += 1

        # Apical dendrites
        for sec in [
            self.radTprox,
            self.radTmed,
            self.radTdist1,
            self.radTdist2,
            self.radTdist3,
        ]:
            self.apical_list.append(sec=sec)
            self.trunk_list.append(sec=sec)
            section_index += 1

        # SR sections
        for sec in [
            self.rad_thick1,
            self.rad_medium1,
            self.rad_thin1a,
            self.rad_thin1b,
            self.rad_thick2,
            self.rad_medium2,
            self.rad_thin2a,
            self.rad_thin2b,
        ]:
            self.apical_list.append(sec=sec)
            self.apical_SR_list.append(sec=sec)
            self.apical_SR_index.append(section_index)
            section_index += 1

        # SLM sections
        for sec in [
            self.lm_thick1,
            self.lm_medium1,
            self.lm_thin1a,
            self.lm_thin1b,
            self.lm_thick2,
            self.lm_medium2,
            self.lm_thin2a,
            self.lm_thin2b,
        ]:
            self.apical_list.append(sec=sec)
            self.apical_SLM_list.append(sec=sec)
            self.apical_SLM_index.append(section_index)
            section_index += 1

        # Tuft list
        for sec in [
            self.lm_thick1,
            self.lm_medium1,
            self.lm_thin1a,
            self.lm_thin1b,
            self.lm_thick2,
            self.lm_medium2,
            self.lm_thin2a,
            self.lm_thin2b,
        ]:
            self.tuft_list.append(sec=sec)

        # Oblique list
        for sec in [
            self.rad_thick1,
            self.rad_medium1,
            self.rad_thin1a,
            self.rad_thin1b,
            self.rad_thick2,
            self.rad_medium2,
            self.rad_thin2a,
            self.rad_thin2b,
        ]:
            self.oblique_list.append(sec=sec)

        # Basal dendrites
        for sec in [
            self.oriprox1,
            self.oridist1a,
            self.oridist1b,
            self.oriprox2,
            self.oridist2a,
            self.oridist2b,
        ]:
            self.basal_list.append(sec=sec)
            self.basal_SO_list.append(sec=sec)
            self.basal_SO_index.append(section_index)
            section_index += 1

        # AIS
        self.ais_list.append(sec=self.ais)
        self.ais_SP_list.append(sec=self.ais)
        self.ais_SP_index.append(section_index)
        section_index += 1

        # Hillock
        self.hillock_list.append(sec=self.hillock)
        self.hillock_SP_list.append(sec=self.hillock)
        self.hillock_SP_index.append(section_index)
        section_index += 1

        # Axon
        self.axon_list.append(sec=self.axon)
        self.axon_SR_list.append(sec=self.axon)
        self.axon_SR_index.append(section_index)
        section_index += 1

    def geom(self):
        # Set geometry parameters for all sections
        geom_params = {
            self.soma: (10, 20),
            self.radTprox: (80, (3, 2.5)),
            self.radTmed: (70, 2.5),
            self.radTdist1: (50, 2.0),
            self.radTdist2: (50, 1.75),
            self.radTdist3: (50, 1.5),
            self.lm_thick1: (50, 0.95),
            self.lm_medium1: (50, 0.9),
            self.lm_thin1a: (75, 0.6),
            self.lm_thin1b: (75, 0.6),
            self.lm_thick2: (100, 0.75),
            self.lm_medium2: (50, 0.9),
            self.lm_thin2a: (75, 0.6),
            self.lm_thin2b: (75, 0.6),
            self.rad_thick1: (50, 1.9),
            self.rad_medium1: (50, 1.5),
            self.rad_thin1a: (75, 0.95),
            self.rad_thin1b: (75, 0.95),
            self.rad_thick2: (50, 1.9),
            self.rad_medium2: (50, 1.5),
            self.rad_thin2a: (75, 0.95),
            self.rad_thin2b: (75, 0.95),
            self.oriprox1: (75, 1.6),
            self.oridist1a: (75, 1.1),
            self.oridist1b: (75, 1.1),
            self.oriprox2: (75, 1.6),
            self.oridist2a: (75, 1.1),
            self.oridist2b: (75, 1.1),
            self.hillock: (20, (3.5, 1.0)),
            self.ais: (10, (1.0, 1.0)),
            self.axon: (150, (0.5, 0.5)),
        }

        for sec, (L, diam) in geom_params.items():
            sec.L = L
            if isinstance(diam, tuple):
                nseg = int(sec.L // 5)
                if (nseg < 1) or (nseg % 2 == 0):
                    nseg = nseg + 1
                sec.nseg = nseg
                for seg in sec:
                    seg.diam = np.interp(seg.x, [0, 1], [diam[0], diam[1]])
            else:
                sec.diam = diam
        h.define_shape()

    def geom_nseg(self, freq=100, d_lambda=0.1):
        for sec in self.all:
            nseg = (
                int((sec.L / (d_lambda * self.lambda_f(sec, freq)) + 0.9) / 2) * 2 + 1
            )
            sec.nseg = nseg

    def lambda_f(self, section, freq):
        if section.n3d() < 2:
            return 1e5 * np.sqrt(
                section.diam / (4 * np.pi * freq * section.Ra * section.cm)
            )

        x1 = section.arc3d(0)
        d1 = section.diam3d(0)
        lam = 0

        for i in range(1, section.n3d()):
            x2 = section.arc3d(i)
            d2 = section.diam3d(i)
            lam += (x2 - x1) / np.sqrt(d1 + d2)
            x1, d1 = x2, d2

        lam *= np.sqrt(2) * 1e-5 * np.sqrt(4 * np.pi * freq * section.Ra * section.cm)
        return section.L / lam

    def distribute_distance(self, section_list, mechanism, expression):
        """
        Distribute mechanism values based on distance from soma.
        Args:
        section_list: NEURON SectionList object
        mechanism: String name of the mechanism
        expression: Expression for calculating values based on distance
        """
        h.distance(0, 0.5, sec=self.soma)  # Set soma as the origin

        for sec in section_list:
            for seg in sec:
                dist = h.distance(seg.x, sec=sec)
                mech_val = eval(
                    expression % dist
                )  # Evaluate the expression with the distance
                setattr(seg, mechanism, mech_val)

    def biophys(self):
        """Set biophysical properties of the cell."""
        # Set global parameters
        h.celsius = 35

        for sec in self.all:
            sec.insert("pas")
            sec.insert("cad")
            sec.insert("K_conc")
            sec.insert("Na_conc")
            sec.insert("constant")

        for sec in self.soma_list:
            # sec.insert("extracellular")

            sec.insert("hha2")
            sec.insert("h_mech")
            sec.insert("kap")
            sec.insert("km")
            sec.insert("mAHP")
            sec.insert("kca")
            sec.insert("cal")
            sec.insert("cat")
            sec.insert("carsoma")

        for sec in self.axon_list:
            sec.insert("hha2")
            sec.insert("km")

        for sec in self.ais_list:
            sec.insert("hha2")
            sec.insert("km")

        for sec in self.hillock_list:
            sec.insert("hha2")
            sec.insert("km")

        # Apical trunk -- all compartments
        for sec in self.trunk_list:
            sec.insert("hhadend")
            sec.insert("h_mech")
            if sec.name().__contains__("radTprox"):
                sec.insert("kap")
            else:
                sec.insert("kad")
            sec.insert("km")
            sec.insert("kca")
            sec.insert("mAHP")
            sec.insert("caldend")
            sec.insert("cat")
            sec.insert("car")

        # Apical tuft (SLM)
        for sec in self.tuft_list:
            sec.insert("hhadend")
            sec.insert("h_mech")
            sec.insert("kad")
            sec.insert("km")
            sec.insert("kca")
            sec.insert("mAHP")
            sec.insert("caldend")

        # Apical oblique (SR)
        for sec in self.oblique_list:
            sec.insert("hhadend")
            sec.insert("h_mech")
            sec.insert("kad")
            sec.insert("km")
            sec.insert("mAHP")
            sec.insert("kca")
            sec.insert("caldend")

        # Basal dendrites
        for sec in self.basal_list:
            sec.insert("hhadend")
            sec.insert("h_mech")
            sec.insert("kap")
            sec.insert("km")
            sec.insert("kca")
            sec.insert("mAHP")

        # Set parameters for all sections
        for sec in self.all:
            sec.nai0_Na_conc = 10

        # Parameters for soma
        for sec in self.soma_list:
            sec.Ra = self.params.soma.Ra
            sec.cm = self.params.soma.cm
            for seg in sec:
                seg.hha2.gnabar = self.params.soma.gbar_na
                seg.hha2.gkbar = self.params.soma.gkdr
                seg.kap.gkabar = self.params.soma.gka_prox
                seg.km.gbar = self.params.soma.gbar_km
                seg.mAHP.gkbar = self.params.soma.gkahp
                seg.kca.gbar = self.params.soma.gkca
                seg.h_mech.ghbar = self.params.soma.gh
                seg.h_mech.vhalf = -90
                seg.h_mech.eh = self.params.soma.eh
                seg.cat.gcatbar = self.params.soma.gcat
                seg.cal.gcalbar = self.params.soma.gcal
                seg.carsoma.gcabar = self.params.soma.gcar
                seg.g_pas = self.params.soma.g_pas
                seg.e_pas = self.params.soma.e_pas
                seg.constant.ic = self.params.soma.ic

        # Parameters for hillock
        for sec in self.hillock_list:
            sec.Ra = self.params.hillock.Ra
            sec.cm = self.params.hillock.cm
            for seg in sec:
                seg.hha2.gnabar = self.params.hillock.gbar_na
                seg.hha2.gkbar = self.params.hillock.gkdr
                seg.km.gbar = self.params.hillock.gbar_km
                seg.g_pas = self.params.hillock.g_pas
                seg.e_pas = self.params.hillock.e_pas
                seg.constant.ic = self.params.hillock.ic

        # Parameters for AIS
        for sec in self.ais_list:
            sec.Ra = self.params.ais.Ra
            sec.cm = self.params.ais.cm
            for seg in sec:
                seg.hha2.gnabar = self.params.ais.gbar_na
                seg.hha2.gkbar = self.params.ais.gkdr
                seg.km.gbar = self.params.ais.gbar_km
                seg.g_pas = self.params.ais.g_pas
                seg.e_pas = self.params.ais.e_pas
                seg.constant.ic = self.params.ais.ic

        # Parameters for axon
        for sec in self.axon_list:
            sec.Ra = self.params.axon.Ra
            sec.cm = self.params.axon.cm
            for seg in sec:
                seg.hha2.gnabar = self.params.axon.gbar_na
                seg.hha2.gkbar = self.params.axon.gkdr
                seg.km.gbar = self.params.axon.gbar_km
                seg.g_pas = self.params.axon.g_pas
                seg.e_pas = self.params.axon.e_pas
                seg.constant.ic = self.params.axon.ic

        # Parameters for apical dendrites
        # Apical trunk
        for sec in self.trunk_list:
            if sec.name().__contains__("radTprox"):
                params = self.params.radTprox
            elif sec.name().__contains__("radTmed"):
                params = self.params.radTmed
            elif sec.name().__contains__("radTdist"):
                params = self.params.radTdist
            else:
                raise RuntimeException(f"Unknown apical trunk section {sec}")

            sec.Ra = params.Ra
            sec.cm = params.cm  # Membrane capacitance in uF/cm2
            for seg in sec:
                seg.g_pas = params.g_pas
                seg.e_pas = params.e_pas
                seg.hhadend.gnabar = params.gbar_na
                seg.hhadend.gkbar = params.gkdr
                seg.hhadend.ar2 = params.ar2_na
                seg.h_mech.ghbar = params.gh
                seg.h_mech.vhalf = -90
                seg.h_mech.eh = params.eh
                if sec.name().__contains__("radTprox"):
                    seg.kap.gkabar = params.gka
                else:
                    seg.kad.gkabar = params.gka
                seg.km.gbar = params.gbar_km
                seg.kca.gbar = params.gkca
                seg.mAHP.gkbar = params.gkahp

                # L-type Ca2+ channels
                seg.caldend.gcalbar = params.gcal
                # T-type Ca2+ channel
                seg.cat.gcatbar = params.gcat
                # R-type Ca2+ channel
                seg.car.gcabar = params.gcar
                seg.constant.ic = params.ic

        # Apical tuft
        for sec in self.tuft_list:
            sec.Ra = self.params.apical_tuft.Ra
            sec.cm = self.params.apical_tuft.cm  # Membrane capacitance in uF/cm2
            for seg in sec:
                seg.g_pas = self.params.apical_tuft.g_pas
                seg.e_pas = self.params.apical_tuft.e_pas
                seg.hhadend.gnabar = self.params.apical_tuft.gbar_na
                seg.hhadend.gkbar = self.params.apical_tuft.gkdr
                seg.hhadend.ar2 = self.params.apical_tuft.ar2_na
                seg.h_mech.ghbar = self.params.apical_tuft.gh
                seg.h_mech.vhalf = -90
                seg.h_mech.eh = self.params.apical_tuft.eh
                seg.kad.gkabar = self.params.apical_tuft.gka_dist
                seg.km.gbar = self.params.apical_tuft.gbar_km

                # medium Ca2+-dependent K+ channel (mAHP)
                seg.mAHP.gkbar = self.params.apical_tuft.gkahp

                # slow Ca2+-dependent K+ channel (sAHP)
                seg.kca.gbar = self.params.apical_tuft.gkca

                # L-type Ca2+ channels
                seg.caldend.gcalbar = self.params.apical_tuft.gcal
                seg.constant.ic = self.params.apical_tuft.ic

        # Apical oblique dendrites
        for sec in self.oblique_list:
            sec.Ra = self.params.apical_oblique.Ra
            sec.cm = self.params.apical_oblique.cm  # Membrane capacitance in uF/cm2
            for seg in sec:
                seg.g_pas = self.params.apical_oblique.g_pas
                seg.e_pas = self.params.apical_oblique.e_pas
                seg.hhadend.gnabar = self.params.apical_oblique.gbar_na
                seg.hhadend.gkbar = self.params.apical_oblique.gkdr
                seg.hhadend.ar2 = self.params.apical_oblique.ar2_na
                seg.h_mech.ghbar = self.params.apical_oblique.gh
                seg.h_mech.vhalf = -90
                seg.h_mech.eh = self.params.apical_oblique.eh
                seg.kad.gkabar = self.params.apical_oblique.gka_dist
                seg.km.gbar = self.params.apical_oblique.gbar_km

                # medium Ca2+-dependent K+ channel (mAHP)
                seg.mAHP.gkbar = self.params.apical_oblique.gkahp

                # slow Ca2+-dependent K+ channel (sAHP)
                seg.kca.gbar = self.params.apical_oblique.gkca

                # L-type Ca2+ channels
                seg.caldend.gcalbar = self.params.apical_oblique.gcal
                seg.constant.ic = self.params.apical_oblique.ic

        # Basal dendrites
        for sec in self.basal_list:
            sec.Ra = self.params.basal.Ra
            sec.cm = self.params.basal.cm  # Membrane capacitance in uF/cm2
            for seg in sec:
                seg.g_pas = self.params.basal.g_pas
                seg.e_pas = self.params.basal.e_pas
                seg.hhadend.gnabar = self.params.basal.gbar_na
                seg.hhadend.gkbar = self.params.basal.gkdr
                seg.hhadend.ar2 = self.params.basal.ar2_na
                seg.h_mech.ghbar = self.params.basal.gh
                seg.h_mech.vhalf = -90
                seg.h_mech.eh = self.params.basal.eh
                seg.kap.gkabar = self.params.basal.gka_prox
                seg.km.gbar = self.params.basal.gbar_km

                # medium Ca2+-dependent K+ channel (mAHP)
                seg.mAHP.gkbar = self.params.basal.gkahp

                # slow Ca2+-dependent K+ channel (sAHP)
                seg.kca.gbar = self.params.basal.gkca
                seg.constant.ic = self.params.basal.ic

    def export_swc(
        self,
        sections=[
            ("soma", 1),
            ("apical", 4),
            ("basal", 3),
            ("axon", 2),
            ("ais", 7),
            ("hillock", 8),
        ],
    ):
        swc_point_idx = 0
        swc_points = []
        swc_point_sec_dict = defaultdict(list)
        sec_dict = {}
        seen = set([])
        for section, sectype in sections:
            if hasattr(self, f"{section}_list"):
                seclist = list(getattr(self, f"{section}_list"))
                for secidx, sec in enumerate(seclist):
                    if hasattr(sec, "sec"):
                        sec = sec.sec
                    if sec in seen:
                        continue

                    seen.add(sec)
                    n3d = sec.n3d()
                    if n3d == 2:
                        x1 = sec.x3d(0)
                        y1 = sec.y3d(0)
                        z1 = sec.z3d(0)
                        d1 = sec.diam3d(0)
                        x2 = sec.x3d(1)
                        y2 = sec.y3d(1)
                        z2 = sec.z3d(1)
                        d2 = sec.diam3d(1)
                        mx = (x2 + x1) / 2.0
                        my = (y2 + y1) / 2.0
                        mz = (z2 + z1) / 2.0
                        dd = d1 - (d1 - d2) / 2.0
                        sec.pt3dinsert(1, mx, my, mz, dd)
                        n3d = sec.n3d()
                    L = sec.L
                    for i in range(n3d):
                        x = sec.x3d(i)
                        y = sec.y3d(i)
                        z = sec.z3d(i)
                        d = sec.diam3d(i)
                        ll = sec.arc3d(i)
                        rad = d / 2.0
                        loc = ll / L
                        first = True if i == 0 else False
                        swc_point = (
                            swc_point_idx,
                            section,
                            sectype,
                            x,
                            y,
                            z,
                            rad,
                            loc,
                            sec,
                            first,
                        )
                        swc_points.append(swc_point)
                        swc_point_sec_dict[sec.name()].append(swc_point)
                        swc_point_idx += 1
        soma_sec = list(self.soma_list)[0]
        for swc_point in swc_points:
            (swc_point_idx, section, sectype, x, y, z, rad, loc, sec, first) = swc_point
            parent_idx = -1
            distance_to_soma = h.distance(soma_sec(0.5), sec(loc))
            if not first:
                parent_idx = swc_point_idx - 1
            else:
                parent_seg = sec.parentseg()
                if parent_seg is not None:
                    parent_x = parent_seg.x
                    parent_sec = parent_seg.sec
                    parent_points = swc_point_sec_dict[parent_sec.name()]
                    for parent_point in parent_points:
                        (parent_point_idx, _, _, _, _, _, _, parent_point_loc, _, _) = (
                            parent_point
                        )
                        if parent_point_loc >= parent_x:
                            parent_idx = parent_point_idx
                            break
            # layer = get_layer(distance_to_soma, sectype)
            print(
                "%d %i %.04f %.04f %.04f %.04f %d"
                % (swc_point_idx, sectype, x, y, z, rad, parent_idx)
            )

    def position(self, x, y, z):
        xx = yy = zz = 0
        for sec in [self.soma]:  # , self.dend]:
            for i in range(sec.n3d()):
                pt3d = h.pt3dchange(
                    i,
                    x - xx + sec.x3d(i),
                    y - yy + sec.y3d(i),
                    z - zz + sec.z3d(i),
                    sec.diam3d(i),
                )
        xx, yy, zz = x, y, z

    def is_art(self):
        return False

    def is_reduced(self):
        return True

    def ic_constant_0(self):
        result = []
        for sec in self.soma_list:
            result.append(sec.ic_constant)
        for sec in self.ais_list:
            result.append(sec.ic_constant)
        for sec in self.hillock_list:
            result.append(sec.ic_constant)
        for sec in self.axon_list:
            result.append(sec.ic_constant)
        for sec in self.apical_list:
            result.append(sec.ic_constant)
        for sec in self.basal_list:
            result.append(sec.ic_constant)
        return np.asarray(result)


def ic_constant_f(
    x,
    cell,
    ic_constant,
    v_hold=-70,
    tstop=150.0,
    dt=0.01,
    record_dt=0.01,
    celsius=35.0,
    section_types=["soma"],
    use_cvode=True,
    use_coreneuron=True,
):
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
    h.dt = dt

    if record_dt < dt:
        record_dt = dt

    # Enable variable time step solver
    if use_cvode:
        h.cvode.active(1)

    if use_coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True

    h.celsius = celsius

    # Create the recording vectors for time and voltage
    vec_t = h.Vector()
    vec_t.record(h._ref_t, record_dt)  # Time
    vec_v_dict = {}
    for section_type in section_types:
        for sec in getattr(cell, f"{section_type}_list"):
            vec_v = h.Vector()
            vec_v.record(cell.soma(0.5)._ref_v, record_dt)  # Voltage
            vec_v_dict[sec] = vec_v

    # Run the simulation
    h.tstop = tstop
    h.v_init = v_hold
    h.init()
    for i, section_type in enumerate(section_types):
        for sec in getattr(cell, f"{section_type}_list"):
            for seg in sec:
                if isinstance(x, float):
                    seg.constant.ic = ic_constant[i] + round(x, 6)
                else:
                    seg.constant.ic = ic_constant[i] + round(x[i], 6)

    h.finitialize(h.v_init)
    h.finitialize(h.v_init)
    try:
        h.run()
    except:
        pass

    t = vec_t.as_numpy()

    mean_vs = []
    for sec, vec_v in vec_v_dict.items():
        v = vec_v.as_numpy()
        mean_v = np.mean(v) if np.max(v) < 0.0 else 0.0
        mean_vs.append(mean_v)

    mean_v = np.mean(np.asarray(mean_vs))

    return mean_v - v_hold


if __name__ == "__main__":
    cell = PyramidalCell()
    cell.export_swc()
