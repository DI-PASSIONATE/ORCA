import os
from dataclasses import dataclass

from typing import Any, Dict

@dataclass
class OrcaFolderStructure:
    """
    This class defines the folder structure for the ORCA project.
    It provides methods to get the paths for different stages of the pipeline.
    """

    @staticmethod
    def get_base_dir(context: Dict[str, Any]) -> str:
        return context.get("base_dir", os.getcwd())
    
    @staticmethod
    def get_geometry_dir(context: Dict[str, Any]) -> str:
        base_dir = OrcaFolderStructure.get_base_dir(context)
        return os.path.join(base_dir, "geometries")
    
    @staticmethod
    def get_gds_csv(context: Dict[str, Any]) -> str:
        geometry_dir = OrcaFolderStructure.get_geometry_dir(context)
        geometry_name = context["geometry"].name
        return os.path.join(geometry_dir, f"{geometry_name}.csv")
    
    @staticmethod
    def get_palace_sim_dir(context: Dict[str, Any]) -> str:
        base_dir = OrcaFolderStructure.get_base_dir(context)
        return os.path.join(base_dir, "palace_sims")
    
    @staticmethod
    def get_palace_csv(context: Dict[str, Any]) -> str:
        palace_sim_dir = OrcaFolderStructure.get_palace_sim_dir(context)
        geometry_name = context["geometry"].name
        return os.path.join(palace_sim_dir, f"{geometry_name}.csv")
    
    @staticmethod
    def get_result_dir(context: Dict[str, Any]) -> str:
        base_dir = OrcaFolderStructure.get_base_dir(context)
        return context.get("result_dir", os.path.join(base_dir, "results"))
    
    @staticmethod
    def get_result_csv(context: Dict[str, Any]) -> str:
        result_dir = OrcaFolderStructure.get_result_dir(context)
        geometry_name = context["geometry"].name
        return context.get("result_csv", os.path.join(result_dir, f"{geometry_name}.csv"))