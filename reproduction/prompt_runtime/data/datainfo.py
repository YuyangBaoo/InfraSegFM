# -*- coding: utf-8 -*-
"""Infrastructure taxonomy definitions for InfraSegFM.

Hierarchy
---------
- L0: acquisition platform (sensor / rig)
- L1: asset category (road / bridge / building / ...)
- L2: subsystem / scene group (surface / structural / ...)
- L3: material / surface type (asphalt / concrete / steel / ...)
- L4: task / defect type (leaf)

All indices are contiguous (0..N-1).
"""

from __future__ import annotations

# -------------------------
# L0: Acquisition platform
# -------------------------
platform_dict = {
    "Handheld": [
        "ConcreteCrack",
        "AsphaltCrack",
        "ConcretePavementPothole",
        "ConstructionJoint",
        "Efflorescence",
        "Rust",
        "HollowArea",
        "Spalling",
        "Weathering",
        "MasonryCrack",
        "SteelCorrosion",
        "ExposedRebar",
        "DamagedBuilding",
        "Debris",
        "UndamagedRoad",
        "UndamagedBuilding",
    ],
    "IndustrialLineScan": ["RailCrack", "RailPothole"],
    "VehicleProfiler": ["AsphaltCrack", "ConcreteCrack", "ConcretePavementPothole", "ConstructionJoint"],
    "Aerial": ["RoadCrack", "Pothole"],
    "Satellite": ["RoadCrack", "Pothole"],
    "IoT": ["RoadCrack", "Pothole"],
}

platform_list = sorted(platform_dict.keys())
platform_map = {k: i for i, k in enumerate(platform_list)}


# -------------------------
# L4: Task / defect leaf
# -------------------------
# Keep stable ordering for checkpoints.

task_list = [
    "ConcreteCrack",
    "AsphaltCrack",
    "RailCrack",
    "RoadCrack",
    "ConcretePavementPothole",
    "AsphaltPothole",
    "RailPothole",
    "ConstructionJoint",
    "Efflorescence",
    "Rust",
    "HollowArea",
    "Spalling",
    "Weathering",
    "MasonryCrack",
    "SteelCorrosion",
    "ExposedRebar",
    "DamagedBuilding",
    "Debris",
    "UndamagedRoad",
    "UndamagedBuilding",
]

task_id = {k: i for i, k in enumerate(task_list)}


# -------------------------
# L1: Asset category
# -------------------------
asset_level_1_dict = {
    "Road": [
        "ConcreteCrack",
        "AsphaltCrack",
        "RoadCrack",
        "ConcretePavementPothole",
        "AsphaltPothole",
        "UndamagedRoad",
        "ConstructionJoint",
    ],
    "Railway": ["RailCrack", "RailPothole"],
    "Bridge": ["ConcreteCrack", "Rust", "Efflorescence", "HollowArea", "Spalling", "Weathering", "SteelCorrosion"],
    "Building": ["MasonryCrack", "ExposedRebar", "DamagedBuilding", "UndamagedBuilding"],
    "Infrastructure": ["DamagedBuilding", "Debris", "UndamagedRoad", "UndamagedBuilding"],
}
asset_level_1_map = {k: i for i, k in enumerate(asset_level_1_dict.keys())}


# -------------------------
# L2: Subsystem / scene group
# -------------------------
asset_level_2_dict = {
    "Surface": [
        "ConcreteCrack",
        "AsphaltCrack",
        "RoadCrack",
        "ConcretePavementPothole",
        "AsphaltPothole",
        "ConcretePavementPothole",
        "ConstructionJoint",
        "RailCrack",
        "RailPothole",
    ],
    "Structural": [
        "ConcreteCrack",
        "Rust",
        "Efflorescence",
        "HollowArea",
        "Spalling",
        "Weathering",
        "MasonryCrack",
        "SteelCorrosion",
        "ExposedRebar",
        "DamagedBuilding",
        "Debris",
        "UndamagedRoad",
        "UndamagedBuilding",
    ],
}
asset_level_2_map = {k: i for i, k in enumerate(asset_level_2_dict.keys())}


# -------------------------
# L3: Material / surface type
# -------------------------
asset_level_3_dict = {
    "Concrete": [
        "ConcreteCrack",
        "ConcretePavementPothole",
        "ConstructionJoint",
        "Efflorescence",
        "HollowArea",
        "Spalling",
        "Weathering",
        "ExposedRebar",
    ],
    "Asphalt": ["AsphaltCrack", "AsphaltPothole", "RoadCrack"],
    "Steel": ["SteelCorrosion"],
    "Metal": ["Rust"],
    "Masonry": ["MasonryCrack"],
    "Mixed": ["DamagedBuilding", "Debris", "UndamagedRoad", "UndamagedBuilding"],
}
asset_level_3_map = {k: i for i, k in enumerate(asset_level_3_dict.keys())}


# -------------------------
# TaskFolder -> (L0,L1,L2,L3,L4)
# -------------------------
TASK_FOLDER_META = {
    "Handheld_ConcreteCrack": ("Handheld", "Bridge", "Structural", "Concrete", "ConcreteCrack"),
    "Handheld_AsphaltCrack": ("Handheld", "Road", "Surface", "Asphalt", "AsphaltCrack"),
    "Handheld_ConcretePavementPothole": ("Handheld", "Road", "Surface", "Concrete", "ConcretePavementPothole"),
    "Handheld_ConstructionJoint": ("Handheld", "Road", "Surface", "Concrete", "ConstructionJoint"),
    "Handheld_Efflorescence": ("Handheld", "Bridge", "Structural", "Concrete", "Efflorescence"),
    "Handheld_Rust": ("Handheld", "Bridge", "Structural", "Metal", "Rust"),
    "Handheld_HollowArea": ("Handheld", "Bridge", "Structural", "Concrete", "HollowArea"),
    "Handheld_Spalling": ("Handheld", "Bridge", "Structural", "Concrete", "Spalling"),
    "Handheld_Weathering": ("Handheld", "Bridge", "Structural", "Concrete", "Weathering"),
    "Handheld_MasonryCrack": ("Handheld", "Building", "Structural", "Masonry", "MasonryCrack"),
    "Handheld_SteelCorrosion": ("Handheld", "Bridge", "Structural", "Steel", "SteelCorrosion"),
    "Handheld_ExposedRebar": ("Handheld", "Building", "Structural", "Concrete", "ExposedRebar"),
    "Handheld_DamagedBuilding": ("Handheld", "Infrastructure", "Structural", "Mixed", "DamagedBuilding"),
    "Handheld_Debris": ("Handheld", "Infrastructure", "Structural", "Mixed", "Debris"),
    "Handheld_UndamagedRoad": ("Handheld", "Infrastructure", "Structural", "Mixed", "UndamagedRoad"),
    "Handheld_UndamagedBuilding": ("Handheld", "Infrastructure", "Structural", "Mixed", "UndamagedBuilding"),

    "IndustrialLineScan_RailCrack": ("IndustrialLineScan", "Railway", "Surface", "Steel", "RailCrack"),
    "IndustrialLineScan_RailPothole": ("IndustrialLineScan", "Railway", "Surface", "Steel", "RailPothole"),

    "VehicleProfiler_AsphaltCrack": ("VehicleProfiler", "Road", "Surface", "Asphalt", "AsphaltCrack"),
    "VehicleProfiler_ConcreteCrack": ("VehicleProfiler", "Road", "Surface", "Concrete", "ConcreteCrack"),
    "VehicleProfiler_ConcretePavementPothole": ("VehicleProfiler", "Road", "Surface", "Concrete", "ConcretePavementPothole"),
    "VehicleProfiler_ConstructionJoint": ("VehicleProfiler", "Road", "Surface", "Concrete", "ConstructionJoint"),

    "Aerial_RoadCrack": ("Aerial", "Road", "Surface", "Asphalt", "RoadCrack"),
    "Aerial_Pothole": ("Aerial", "Road", "Surface", "Asphalt", "AsphaltPothole"),

    "Satellite_RoadCrack": ("Satellite", "Road", "Surface", "Asphalt", "RoadCrack"),
    "Satellite_Pothole": ("Satellite", "Road", "Surface", "Asphalt", "AsphaltPothole"),

    "IoT_RoadCrack": ("IoT", "Road", "Surface", "Asphalt", "RoadCrack"),
    "IoT_Pothole": ("IoT", "Road", "Surface", "Asphalt", "AsphaltPothole"),
}
