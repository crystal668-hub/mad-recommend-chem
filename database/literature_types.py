from __future__ import annotations

from typing import Dict


LITERATURE_TYPE_CONFIGS: Dict[str, Dict[str, str]] = {
    "CO2RR": {"path": "CO2RR", "metadata_csv": "./metadata/CO2RR.csv"},
    "EOR": {"path": "EOR", "metadata_csv": "./metadata/EOR.csv"},
    "HER": {"path": "HER", "metadata_csv": "./metadata/HER.csv"},
    "HOR": {"path": "HOR", "metadata_csv": "./metadata/HOR.csv"},
    "HZOR": {"path": "HZOR", "metadata_csv": "./metadata/HZOR.csv"},
    "O5H": {"path": "O5H", "metadata_csv": "./metadata/O5H.csv"},
    "OER": {"path": "OER", "metadata_csv": "./metadata/OER.csv"},
    "ORR": {"path": "ORR", "metadata_csv": "./metadata/ORR.csv"},
    "UOR": {"path": "UOR", "metadata_csv": "./metadata/UOR.csv"},
    "Antibacterial": {
        "path": "Antibacterial",
        "metadata_csv": "./metadata/Antibacterial.csv",
    },
    "Thermoelectric": {
        "path": "Thermoelectric",
        "metadata_csv": "./metadata/Thermoelectric.csv",
    },
    "antiferromagnetism": {
        "path": "antiferromagnetism",
        "metadata_csv": "./metadata/Antiferromagnetism.csv",
    },
    "conductivity": {
        "path": "conductivity",
        "metadata_csv": "./metadata/Conductivity.csv",
    },
    "ferrimagnetism": {
        "path": "ferrimagnetism",
        "metadata_csv": "./metadata/Ferrimagnetism.csv",
    },
    "ferromagnetism": {
        "path": "ferromagnetism",
        "metadata_csv": "./metadata/Ferromagnetism.csv",
    },
    "photothermal conversion efficiency": {
        "path": "photothermal conversion efficiency",
        "metadata_csv": "./metadata/Photothermal conversion efficiency.csv",
    },
    "photocatalytic H2O2 production": {
        "path": "photocatalytic H2O2 production",
        "metadata_csv": "./metadata/Photocatalytic H2O2 production.csv",
    },
    "hydrogenation of furfural": {
        "path": "hydrogenation of furfural",
        "metadata_csv": "./metadata/Hydrogenation of furfural.csv",
    },
    "thermal conductivity": {
        "path": "thermal conductivity",
        "metadata_csv": "./metadata/Thermal Conductivity.csv",
    },
}
