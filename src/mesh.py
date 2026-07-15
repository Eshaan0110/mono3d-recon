"""Surface mesh reconstruction from point clouds."""

import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import Optional, Tuple

from .config import MeshConfig


class MeshReconstructor:
    """Reconstruct triangle meshes from point clouds.

    Args:
        config: Mesh reconstruction configuration.
    """

    def __init__(self, config: Optional[MeshConfig] = None):
        self.config = config or MeshConfig()

    def reconstruct(
        self, pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.TriangleMesh:
        """Reconstruct a mesh from a point cloud.

        Args:
            pcd: Input point cloud with normals.

        Returns:
            Triangle mesh.
        """
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.05, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        if self.config.method == "poisson":
            mesh = self._poisson_reconstruction(pcd)
        elif self.config.method == "bpa":
            mesh = self._bpa_reconstruction(pcd)
        else:
            raise ValueError(f"Unknown mesh method: {self.config.method}")

        # Post-process
        mesh = self._postprocess(mesh, pcd)

        return mesh

    def _poisson_reconstruction(
        self, pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.TriangleMesh:
        """Poisson surface reconstruction."""
        print(f"Running Poisson reconstruction (depth={self.config.poisson_depth})...")

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.config.poisson_depth,
            scale=self.config.poisson_scale,
            linear_fit=False,
        )

        # Remove low-density vertices (artifacts at boundaries)
        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, 5)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh

    def _bpa_reconstruction(
        self, pcd: o3d.geometry.PointCloud
    ) -> o3d.geometry.TriangleMesh:
        """Ball Pivoting Algorithm reconstruction."""
        print("Running Ball Pivoting reconstruction...")

        # Estimate ball radii from point cloud
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 1.0, avg_dist * 2.0, avg_dist * 4.0]

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii),
        )

        return mesh

    def _postprocess(
        self,
        mesh: o3d.geometry.TriangleMesh,
        pcd: o3d.geometry.PointCloud,
    ) -> o3d.geometry.TriangleMesh:
        """Clean and simplify the mesh."""
        original_triangles = len(mesh.triangles)

        # Remove degenerate triangles
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Smooth
        if self.config.smooth_iterations > 0:
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=self.config.smooth_iterations
            )

        # Simplify if over target face count
        if len(mesh.triangles) > self.config.target_faces:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=self.config.target_faces
            )

        # Transfer vertex colors from point cloud
        if pcd.has_colors() and not mesh.has_vertex_colors():
            self._transfer_colors(mesh, pcd)

        # Compute normals for rendering
        mesh.compute_vertex_normals()

        final_triangles = len(mesh.triangles)
        print(f"Mesh: {original_triangles} -> {final_triangles} triangles")

        return mesh

    @staticmethod
    def _transfer_colors(
        mesh: o3d.geometry.TriangleMesh,
        pcd: o3d.geometry.PointCloud,
    ):
        """Transfer colors from point cloud to mesh vertices via nearest neighbor."""
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_vertices = np.asarray(mesh.vertices)
        pcd_colors = np.asarray(pcd.colors)
        vertex_colors = np.zeros_like(mesh_vertices)

        for i, vertex in enumerate(mesh_vertices):
            _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
            vertex_colors[i] = pcd_colors[idx[0]]

        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    @staticmethod
    def get_mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict:
        """Compute mesh quality statistics."""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min

        stats = {
            "num_vertices": len(vertices),
            "num_triangles": len(triangles),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "bbox_size": bbox_size.tolist(),
            "surface_area": mesh.get_surface_area(),
            "is_watertight": mesh.is_watertight(),
            "has_vertex_colors": mesh.has_vertex_colors(),
        }
        return stats


def save_mesh_ply(mesh: o3d.geometry.TriangleMesh, path: str):
    """Save mesh in PLY format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)
    print(f"Saved mesh (PLY): {path}")


def save_mesh_glb(mesh: o3d.geometry.TriangleMesh, path: str):
    """Save mesh in GLB (binary glTF) format for web viewing.

    Uses trimesh for GLB export since Open3D doesn't support it natively.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Build trimesh object
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Transfer vertex colors if available
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        # Convert to uint8 RGBA
        rgba = np.ones((len(colors), 4), dtype=np.uint8) * 255
        rgba[:, :3] = (colors * 255).astype(np.uint8)
        tri_mesh.visual.vertex_colors = rgba

    # Transfer vertex normals
    if mesh.has_vertex_normals():
        tri_mesh.vertex_normals = np.asarray(mesh.vertex_normals)

    tri_mesh.export(str(path), file_type="glb")
    print(f"Saved mesh (GLB): {path}")


def save_mesh_obj(mesh: o3d.geometry.TriangleMesh, path: str):
    """Save mesh in OBJ format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)
    print(f"Saved mesh (OBJ): {path}")
