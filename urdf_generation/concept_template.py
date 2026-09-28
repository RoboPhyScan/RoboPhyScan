import numpy as np
from base_template import ConceptTemplate
from geometry_template import *
from utils import apply_transformation
from knowledge_utils import *
import trimesh

class Multilevel_Body(ConceptTemplate):
    def __init__(self, num_levels, level_1_size, level_2_size, level_3_size, level_4_size, position = [0, 0, 0], rotation = [0, 0, 0]):

        # Process rotation param
        rotation = [x / 180 * np.pi for x in rotation]
        super().__init__(position, rotation)

        # Record Parameters
        self.num_levels = num_levels
        self.level_1_size = level_1_size
        self.level_2_size = level_2_size
        self.level_3_size = level_3_size
        self.level_4_size = level_4_size

        # Instantiate component geometries
        vertices_list = []
        faces_list = []
        total_num_vertices = 0

        self.bottom_mesh = Cylinder(level_1_size[1], level_1_size[0], level_1_size[2])
        vertices_list.append(self.bottom_mesh.vertices)
        faces_list.append(self.bottom_mesh.faces + total_num_vertices)
        total_num_vertices += len(self.bottom_mesh.vertices)

        delta_height = level_1_size[1] / 2
        for i in range(num_levels[0] - 1):
            delta_height += locals()['level_'+ str(i+2) +'_size'][1] / 2
            mesh_position = [0, delta_height, 0]
            delta_height += locals()['level_'+ str(i+2) +'_size'][1] / 2
            self.mesh = Cylinder(locals()['level_'+ str(i+2) +'_size'][1], locals()['level_'+ str(i+2) +'_size'][0], locals()['level_'+ str(i+1) +'_size'][0], 
                                 position = mesh_position)
            vertices_list.append(self.mesh.vertices)
            faces_list.append(self.mesh.faces + total_num_vertices)
            total_num_vertices += len(self.mesh.vertices)

        self.vertices = np.concatenate(vertices_list)
        self.faces = np.concatenate(faces_list)

        # Global Transformation
        self.vertices = apply_transformation(self.vertices, position, rotation)


        self.overall_obj_mesh = trimesh.Trimesh(self.vertices, self.faces)
        self.overall_obj_pts = np.array(self.overall_obj_mesh.sample(SAMPLENUM))

        self.semantic = 'Body'


class Cylindrical_Lid(ConceptTemplate):
    def __init__(self, outer_size, inner_size, position = [0, 0, 0], rotation = [0, 0, 0]):

        # Process rotation param
        rotation = [x / 180 * np.pi for x in rotation]
        super().__init__(position, rotation)

        # Record Parameters
        self.outer_size = outer_size
        self.inner_size = inner_size

        # Instantiate component geometries
        vertices_list = []
        faces_list = []
        total_num_vertices = 0

        middle_radius = outer_size[1] * (1 - inner_size[2] / outer_size[2]) + outer_size[0] * inner_size[2] / outer_size[2]
        top_height = outer_size[2] - inner_size[2]

        bottom_mesh_position = [0, -(outer_size[2] - inner_size[2]) / 2, 0]
        self.bottom_mesh = Ring(inner_size[2], middle_radius, inner_size[0], 
                                outer_bottom_radius = outer_size[1],
                                inner_bottom_radius = inner_size[1],
                                position=bottom_mesh_position)
        vertices_list.append(self.bottom_mesh.vertices)
        faces_list.append(self.bottom_mesh.faces + total_num_vertices)
        total_num_vertices += len(self.bottom_mesh.vertices)

        if top_height > 1e-8:
            top_mesh_position = [0, inner_size[2] / 2, 0]
            self.top_mesh = Cylinder(top_height, outer_size[0], middle_radius,
                                     position=top_mesh_position)
            vertices_list.append(self.top_mesh.vertices)
            faces_list.append(self.top_mesh.faces + total_num_vertices)
            total_num_vertices += len(self.top_mesh.vertices)

        self.vertices = np.concatenate(vertices_list)
        self.faces = np.concatenate(faces_list)

        # Global Transformation
        self.vertices = apply_transformation(self.vertices, position, rotation)

        self.overall_obj_mesh = trimesh.Trimesh(self.vertices, self.faces)
        self.overall_obj_pts = np.array(self.overall_obj_mesh.sample(SAMPLENUM))

        self.semantic = 'Lid'


class Cylindrical_Button(ConceptTemplate):
    def __init__(
        self,
        radius,
        thickness,
        position=[0, 0, 0],
        rotation=[0, 0, 0],
    ):
        rotation = [x / 180 * np.pi for x in rotation]
        super().__init__(position, rotation)

        vertices_list = []
        faces_list = []
        total_num_vertices = 0

        r = radius[0]
        t = thickness[0]

        cylinder = Cylinder(t, r)
        cylinder.vertices = apply_transformation(
            cylinder.vertices, position=[0, 0, t / 2], rotation=[np.pi / 2, 0, 0]
        )

        vertices_list.append(cylinder.vertices)
        faces_list.append(cylinder.faces + total_num_vertices)
        total_num_vertices += len(cylinder.vertices)

        self.radius = radius
        self.thickness = thickness
        self.vertices = np.concatenate(vertices_list)
        self.faces = np.concatenate(faces_list)

        # Global Transformation
        self.vertices = apply_transformation(
            self.vertices, position, rotation, offset_first=True
        )

        self.overall_obj_mesh = trimesh.Trimesh(self.vertices, self.faces)
        self.overall_obj_pts = np.array(self.overall_obj_mesh.sample(SAMPLENUM))

        self.semantic = 'Button'


class Round_U_Handle(ConceptTemplate):
    def __init__(self, inner_radius, vertical_separation, vertical_length, position = [0, 0, 0], rotation = [0, 0, 0]):

        # Process rotation param
        rotation = [x / 180 * np.pi for x in rotation]
        super().__init__(position, rotation)

        # Record Parameters
        self.inner_radius = inner_radius
        self.vertical_separation = vertical_separation
        self.vertical_length = vertical_length

        # Instantiate component geometries
        vertices_list = []
        faces_list = []
        total_num_vertices = 0

        left_mesh_position = [
            vertical_separation[0] / 2,
            vertical_length[0] / 2,
            0
            ]
        self.left_mesh = Cylinder(vertical_length[0], inner_radius[0],
                                  position=left_mesh_position)
        vertices_list.append(self.left_mesh.vertices)
        faces_list.append(self.left_mesh.faces + total_num_vertices)
        total_num_vertices += len(self.left_mesh.vertices)

        right_mesh_position = [
            -vertical_separation[0] / 2,
            vertical_length[0] / 2,
            0
            ]
        self.right_mesh = Cylinder(vertical_length[0], inner_radius[0],
                                  position=right_mesh_position)
        vertices_list.append(self.right_mesh.vertices)
        faces_list.append(self.right_mesh.faces + total_num_vertices)
        total_num_vertices += len(self.right_mesh.vertices)

        curve_mesh_position = [
            0,
            vertical_length[0],
            0
            ]
        curve_mesh_rotation = [-np.pi / 2, 0, 0]
        self.curve_mesh = Torus(vertical_separation[0] / 2, inner_radius[0], np.pi,
                                position=curve_mesh_position,
                                rotation=curve_mesh_rotation)
        vertices_list.append(self.curve_mesh.vertices)
        faces_list.append(self.curve_mesh.faces + total_num_vertices)
        total_num_vertices += len(self.curve_mesh.vertices)

        self.vertices = np.concatenate(vertices_list)
        self.faces = np.concatenate(faces_list)

        # Global Transformation
        self.vertices = apply_transformation(self.vertices, position, rotation)

        self.overall_obj_mesh = trimesh.Trimesh(self.vertices, self.faces)
        self.overall_obj_pts = np.array(self.overall_obj_mesh.sample(SAMPLENUM))

        self.semantic = 'Handle'

class Cylindrical_Connector(ConceptTemplate):
    def __init__(
        self,
        radius,  # 按钮半径
        thickness,  # 按钮厚度
        position=[0, 0, 0],
        rotation=[0, 0, 0],
    ):
        # 角度值转弧度制
        rotation = [x / 180 * np.pi for x in rotation]
        super().__init__(position, rotation)

        # 点和面，总点数用来计算面数
        vertices_list = []
        faces_list = []
        total_num_vertices = 0

        # 将list转换为数
        r = radius[0]
        t = thickness[0]

        # 构建
        cylinder = Cylinder(t, r)
        cylinder.vertices = apply_transformation(
            cylinder.vertices, position=[0, 0, t / 2], rotation=[np.pi / 2, 0, 0]
        )

        vertices_list.append(cylinder.vertices)
        faces_list.append(cylinder.faces + total_num_vertices)
        total_num_vertices += len(cylinder.vertices)

        self.vertices = np.concatenate(vertices_list)
        self.faces = np.concatenate(faces_list)

        # 全局变换
        self.vertices = apply_transformation(
            self.vertices, position, rotation, offset_first=True
        )


