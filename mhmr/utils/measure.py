# import importlib

# libraries = [
#     'smplx', 'torch', 'os', 'plotly', 'trimesh', 'argparse', 'json', 'sys',
#     'numpy', 'plotly.graph_objects', 'plotly.express',
#     'typing', 'scipy.spatial', 'plotly.subplots'
# ]

# for lib in libraries:
#     try:
#         importlib.import_module(lib)
#     except ImportError:
#         print(f"{lib} 라이브러리가 설치되어 있지 않습니다.")

import smplx, torch, os, plotly, trimesh, argparse, json, sys

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from typing import List, Dict
from scipy.spatial import ConvexHull
from plotly.subplots import make_subplots
from pprint import pprint

from .measurement_definitions import MEASUREMENT_TYPES, SMPL_IND2JOINT, SMPL_JOINT2IND, SMPL_LANDMARK_INDICES, SMPL_NUM_JOINTS, SMPLX_IND2JOINT, SMPLX_JOINT2IND, SMPLX_LANDMARK_INDICES, SMPLX_NUM_JOINTS, STANDARD_LABELS, MeasurementType, SMPLMeasurementDefinitions, SMPLXMeasurementDefinitions

def evaluate_mae(gt_measurements,estim_measurements):
    MAE = {}

    for m_name, m_value in gt_measurements.items():
        if m_name in estim_measurements.keys():
            error = abs(m_value - estim_measurements[m_name])
            MAE[m_name] = error

    if MAE == {}:
        print("Measurement dicts do not have any matching measurements!")
        print("Returning empty dict!")

    return MAE

def get_joint_regressor(body_model_type, body_model_root, gender="MALE", num_thetas=24):
    '''
    Extract joint regressor from SMPL body model
    :param body_model_type: str of body model type (smpl or smplx, etc.)
    :param body_model_root: str of location of folders where smpl/smplx 
                            inside which .pkl models 
    
    Return:
    :param model.J_regressor: torch.tensor (23,N) used to 
                              multiply with body model to get 
                              joint locations
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smplx.create(model_path=body_model_root, 
                        model_type=body_model_type,
                        gender=gender, 
                        use_face_contour=False,
                        num_betas=10,
                        body_pose=torch.zeros((1, num_thetas-1 * 3)),
                        ext='pkl')
    return model.J_regressor.to(device)

def load_face_segmentation(path: str):
        '''
        Load face segmentation which defines for each body model part
        the faces that belong to it.
        :param path: str - path to json file with defined face segmentation
        '''

        try:
            with open(path, 'r') as f:
                face_segmentation = json.load(f)
        except FileNotFoundError:
            sys.exit(f"No such file - {path}")

        return face_segmentation


def convex_hull_from_3D_points(slice_segments: np.ndarray):
        '''
        Cretes convex hull from 3D points
        :param slice_segments: np.ndarray, dim N x 2 x 3 representing N 3D segments

        Returns:
        :param slice_segments_hull: np.ndarray, dim N x 2 x 3 representing N 3D segments
                                    that form the convex hull
        '''

        # stack all points in N x 3 array
        merged_segment_points = np.concatenate(slice_segments)
        unique_segment_points = np.unique(merged_segment_points,
                                            axis=0)

        # points lie in plane -- find which ax of x,y,z is redundant
        redundant_plane_coord = np.argmin(np.max(unique_segment_points,axis=0) - 
                                            np.min(unique_segment_points,axis=0) )
        non_redundant_coords = [x for x in range(3) if x!=redundant_plane_coord]

        # create convex hull
        hull = ConvexHull(unique_segment_points[:,non_redundant_coords])
        segment_point_hull_inds = hull.simplices.reshape(-1)

        slice_segments_hull = unique_segment_points[segment_point_hull_inds]
        slice_segments_hull = slice_segments_hull.reshape(-1,2,3)

        return slice_segments_hull


def filter_body_part_slices(slice_segments:np.ndarray, 
                             sliced_faces:np.ndarray,
                             measurement_name: str,
                             circumf_2_bodypart: dict,
                             face_segmentation: dict
                            ):
        '''
        Remove segments that are not in the appropriate body part 
        for the given measurement.
        :param slice_segments: np.ndarray - (N,2,3) for N segments 
                                            represented as two 3D points
        :param sliced_faces: np.ndarray - (N,) representing the indices of the
                                            faces
        :param measurement_name: str - name of the measurement
        :param circumf_2_bodypart: dict - dict mapping measurement to body part
        :param face_segmentation: dict - dict mapping body part to all faces belonging
                                        to it

        Return:
        :param slice_segments: np.ndarray (K,2,3) where K < N, for K segments 
                                represented as two 3D points that are in the 
                                appropriate body part
        '''

        if measurement_name in circumf_2_bodypart.keys():

            body_parts = circumf_2_bodypart[measurement_name]

            if isinstance(body_parts,list):
                body_part_faces = [face_index for body_part in body_parts 
                                    for face_index in face_segmentation[body_part]]
            else:
                body_part_faces = face_segmentation[body_parts]

            N_sliced_faces = sliced_faces.shape[0]

            keep_segments = []
            for i in range(N_sliced_faces):
                if sliced_faces[i] in body_part_faces:
                    keep_segments.append(i)

            return slice_segments[keep_segments]

        else:
            return slice_segments


def point_segmentation_to_face_segmentation(
                point_segmentation: dict,
                faces: np.ndarray,
                save_as: str = None):
    """
    :param point_segmentation: dict - dict mapping body part to 
                                      all points belonging to it
    :param faces: np.ndarray - (N,3) representing the indices of the faces
    :param save_as: str - optional path to save face segmentation as json
    """

    import json
    from tqdm import tqdm
    from collections import Counter

    # create body parts to index mapping
    mapping_bp2ind = dict(zip(point_segmentation.keys(),
                              range(len(point_segmentation.keys()))))
    mapping_ind2bp = {v:k for k,v in mapping_bp2ind.items()}


    # assign each face to body part index
    faces_segmentation = np.zeros_like(faces)
    for i,face in tqdm(enumerate(faces)):
        for bp_name, bp_indices in point_segmentation.items():
            bp_label = mapping_bp2ind[bp_name]
            
            for k in range(3):
                if face[k] in bp_indices:
                    faces_segmentation[i,k] = bp_label
    

    # for each face, assign the most common body part
    face_segmentation_final = np.zeros(faces_segmentation.shape[0])
    for i,f in enumerate(faces_segmentation):
        c = Counter(list(f))
        face_segmentation_final[i] = c.most_common()[0][0]
     

    # create dict with body part as key and faces as values
    face_segmentation_dict = {k:[] for k in mapping_bp2ind.keys()} 
    for i,fff in enumerate(face_segmentation_final):
        face_segmentation_dict[mapping_ind2bp[int(fff)]].append(i)


    # save face segmentation
    if save_as:
        with open(save_as, 'w') as f:
            json.dump(face_segmentation_dict, f)

    return face_segmentation_dict

class Visualizer():
    '''
    Visualize the body model with measurements, landmarks and joints.
    All the measurements are expressed in cm.
    '''

    def __init__(self,
                 verts: np.ndarray,
                 faces: np.ndarray,
                 joints: np.ndarray,
                 landmarks: dict,
                 measurements: dict,
                 measurement_types: dict,
                 length_definitions: dict,
                 circumf_definitions: dict,
                 joint2ind: dict,
                 circumf_2_bodypart: dict,
                 face_segmentation: dict,
                 visualize_body: bool = True,
                 visualize_landmarks: bool = True,
                 visualize_joints: bool = True,
                 visualize_measurements: bool=True,
                 title: str = "Measurement visualization"
                ):
        

        self.verts = verts
        self.faces = faces
        self.joints = joints
        self.landmarks = landmarks
        self.measurements = measurements
        self.measurement_types = measurement_types
        self.length_definitions = length_definitions
        self.circumf_definitions = circumf_definitions
        self.joint2ind = joint2ind
        self.circumf_2_bodypart = circumf_2_bodypart
        self.face_segmentation = face_segmentation

        self.visualize_body = visualize_body
        self.visualize_landmarks = visualize_landmarks
        self.visualize_joints = visualize_joints
        self.visualize_measurements = visualize_measurements

        self.title = title
        
      

    @staticmethod
    def create_mesh_plot(verts: np.ndarray, faces: np.ndarray):
        '''
        Visualize smpl body mesh.
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices

        Return:
        plotly Mesh3d object for plotting
        '''
        mesh_plot = go.Mesh3d(
                            x=verts[:,0],
                            y=verts[:,1],
                            z=verts[:,2],
                            color="gray",
                            hovertemplate ='<i>Index</i>: %{text}',
                            text = [i for i in range(verts.shape[0])],
                            # i, j and k give the vertices of triangles
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2],
                            opacity=0.6,
                            name='body',
                            )
        return mesh_plot
        
    @staticmethod
    def create_joint_plot(joints: np.ndarray):

        return go.Scatter3d(x = joints[:,0],
                            y = joints[:,1], 
                            z = joints[:,2], 
                            mode='markers',
                            marker=dict(size=8,
                                        color="black",
                                        opacity=1,
                                        symbol="cross"
                                        ),
                            name="joints"
                                )
    
    @staticmethod
    def create_wireframe_plot(verts: np.ndarray,faces: np.ndarray):
        '''
        Given vertices and faces, creates a wireframe of plotly segments.
        Used for visualizing the wireframe.
        
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the verts
        '''
        i=faces[:,0]
        j=faces[:,1]
        k=faces[:,2]

        triangles = np.vstack((i,j,k)).T

        x=verts[:,0]
        y=verts[:,1]
        z=verts[:,2]

        vertices = np.vstack((x,y,z)).T
        tri_points = vertices[triangles]

        #extract the lists of x, y, z coordinates of the triangle 
        # vertices and connect them by a "line" by adding None
        # this is a plotly convention for plotting segments
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])

        # return Xe, Ye, Ze 
        wireframe = go.Scatter3d(
                        x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        name='wireframe',
                        line=dict(color= 'rgb(70,70,70)', width=1)
                        )
        return wireframe

    def create_landmarks_plot(self,
                              landmark_names: List[str], 
                              verts: np.ndarray
                              ) -> List[plotly.graph_objs.Scatter3d]:
        '''
        Visualize landmarks from landmark_names list
        :param landmark_names: List[str] of landmark names to visualize

        Return
        :param plots: list of plotly objects to plot
        '''

        plots = []

        landmark_colors = dict(zip(self.landmarks.keys(),
                                px.colors.qualitative.Alphabet))

        for lm_name in landmark_names:
            if lm_name not in self.landmarks.keys():
                print(f"Landmark {lm_name} is not defined.")
                pass

            lm_index = self.landmarks[lm_name]
            if isinstance(lm_index,tuple):
                lm = (verts[lm_index[0]] + verts[lm_index[1]]) / 2
            else:
                lm = verts[lm_index] 

            plot = go.Scatter3d(x = [lm[0]],
                                y = [lm[1]], 
                                z = [lm[2]], 
                                mode='markers',
                                marker=dict(size=8,
                                            color=landmark_colors[lm_name],
                                            opacity=1,
                                            ),
                               name=lm_name
                                )

            plots.append(plot)

        return plots

    def create_measurement_length_plot(self, 
                                       measurement_name: str,
                                       verts: np.ndarray,
                                       color: str
                                       ):
        '''
        Create length measurement plot.
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        '''
        
        measurement_landmarks_inds = self.length_definitions[measurement_name]

        segments = {"x":[],"y":[],"z":[]}
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i],tuple):
                lm_tnp = (verts[measurement_landmarks_inds[i][0]] + 
                          verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm_tnp = verts[measurement_landmarks_inds[i]]
            segments["x"].append(lm_tnp[0])
            segments["y"].append(lm_tnp[1])
            segments["z"].append(lm_tnp[2])
        for ax in ["x","y","z"]:
            segments[ax].append(None)

        if measurement_name in self.measurements:
            m_viz_name = f"{measurement_name}: {self.measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        return go.Scatter3d(x=segments["x"], 
                                    y=segments["y"], 
                                    z=segments["z"],
                                    marker=dict(
                                            size=4,
                                            color="rgba(0,0,0,0)",
                                        ),
                                        line=dict(
                                            color=color,
                                            width=10),
                                        name=m_viz_name
                                        )
        
    def create_measurement_circumference_plot(self,
                                              measurement_name: str,
                                              verts: np.ndarray,
                                              faces: np.ndarray,
                                              color: str):
        '''
        Create circumference measurement plot
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        '''

        circumf_landmarks = self.circumf_definitions[measurement_name]["LANDMARKS"]
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = self.joint2ind[circumf_n1], self.joint2ind[circumf_n2]
        
        plane_origin = np.mean(verts[circumf_landmark_indices,:],axis=0)
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh, 
                        plane_normal=plane_normal, 
                        plane_origin=plane_origin, 
                        return_faces=True) # (N, 2, 3), (N,)

        slice_segments = filter_body_part_slices(slice_segments,
                                                 sliced_faces,
                                                 measurement_name,
                                                 self.circumf_2_bodypart,
                                                 self.face_segmentation)
        
        slice_segments_hull = convex_hull_from_3D_points(slice_segments)
        
        
        draw_segments = {"x":[],"y":[],"z":[]}
        map_ax = {0:"x",1:"y",2:"z"}

        for i in range(slice_segments_hull.shape[0]):
            for j in range(3):
                draw_segments[map_ax[j]].append(slice_segments_hull[i,0,j])
                draw_segments[map_ax[j]].append(slice_segments_hull[i,1,j])
                draw_segments[map_ax[j]].append(None)

        if measurement_name in self.measurements:
            m_viz_name = f"{measurement_name}: {self.measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        return go.Scatter3d(
                            x=draw_segments["x"],
                            y=draw_segments["y"],
                            z=draw_segments["z"],
                            mode="lines",
                            line=dict(
                                color=color,
                                width=10),
                            name=m_viz_name
                                )

    def visualize(self, 
                  measurement_names: List[str] = [], 
                  landmark_names: List[str] = [],
                  title="Measurement visualization"
                  ):
        '''
        Visualize the body model with measurements, landmarks and joints.

        :param measurement_names: List[str], list of strings with measurement names
        :param landmark_names: List[str], list of strings with landmark names
        :param title: str, title of plot
        '''


        fig = go.Figure()

        if self.visualize_body:
            # visualize model mesh
            mesh_plot = self.create_mesh_plot(self.verts, self.faces)
            fig.add_trace(mesh_plot)
            # visualize wireframe
            wireframe_plot = self.create_wireframe_plot(self.verts, self.faces)
            fig.add_trace(wireframe_plot)

        # visualize joints
        if self.visualize_joints:
            joint_plot = self.create_joint_plot(self.joints)
            fig.add_trace(joint_plot)


        # visualize landmarks
        if self.visualize_landmarks:
            landmarks_plot = self.create_landmarks_plot(landmark_names, self.verts)
            fig.add_traces(landmarks_plot)
        

        # visualize measurements
        measurement_colors = dict(zip(self.measurement_types.keys(),
                                  px.colors.qualitative.Alphabet))

        if self.visualize_measurements:
            for m_name in measurement_names:
                if m_name not in self.measurement_types.keys():
                    print(f"Measurement {m_name} not defined.")
                    pass

                if self.measurement_types[m_name] == MeasurementType().LENGTH:
                    measurement_plot = self.create_measurement_length_plot(measurement_name=m_name,
                                                                        verts=self.verts,
                                                                        color=measurement_colors[m_name])     
                elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:
                    measurement_plot = self.create_measurement_circumference_plot(measurement_name=m_name,
                                                                                    verts=self.verts,
                                                                                    faces=self.faces,
                                                                                    color=measurement_colors[m_name])
                
                fig.add_trace(measurement_plot)
                

        fig.update_layout(scene_aspectmode='data',
                            width=1000, height=700,
                            title=title,
                            )
            
        fig.show()


def viz_smplx_joints(visualize_body=True,fig=None,show=True,title="SMPLX joints"):
    """
    Visualize smpl joints on the same plot.
    :param visualize_body: bool, whether to visualize the body or not.
    :param fig: plotly Figure object, if None, create new figure.
    """

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smplx_model =  smplx.create(model_path="data",
                                model_type="smplx",
                                gender="NEUTRAL", 
                                use_face_contour=False,
                                num_betas=10,
                                #body_pose=torch.zeros((1, (55-1) * 3)),
                                ext='pkl')
    
    smplx_model = smplx_model(betas=betas, return_verts=True)
    smplx_joints = smplx_model.joints.detach().numpy()[0]
    smplx_joint_pelvis = smplx_joints[0,:]
    smplx_joints = smplx_joints - smplx_joint_pelvis
    smplx_vertices = smplx_model.vertices.detach().numpy()[0]
    smplx_vertices = smplx_vertices - smplx_joint_pelvis
    smplx_faces = smplx.SMPLX("mhmr/data/smplx",ext="pkl").faces

    joint_colors = px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   ["#000000"]
    
    if isinstance(fig,type(None)):
        fig = go.Figure()

    for i in range(smplx_joints.shape[0]):

        if i in SMPLX_IND2JOINT.keys():
            joint_name = SMPLX_IND2JOINT[i]
        else:
            joint_name = f"noname-{i}"

        joint_plot = go.Scatter3d(x = [smplx_joints[i,0]],
                                    y = [smplx_joints[i,1]], 
                                    z = [smplx_joints[i,2]], 
                                    mode='markers',
                                    marker=dict(size=10,
                                                color=joint_colors[i],
                                                opacity=1,
                                                symbol="circle"
                                                ),
                                    name="smplx-"+joint_name
                                        )


        fig.add_trace(joint_plot)


    if visualize_body:
        plot_body = go.Mesh3d(
                            x=smplx_vertices[:,0],
                            y=smplx_vertices[:,1],
                            z=smplx_vertices[:,2],
                            color = "red",
                            i=smplx_faces[:,0],
                            j=smplx_faces[:,1],
                            k=smplx_faces[:,2],
                            name='smplx mesh',
                            showscale=True,
                            opacity=0.5
                        )
        fig.add_trace(plot_body)

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title,
                        )
    

    if show:
        fig.show()
    else:
        return fig


def viz_smpl_joints(visualize_body=True,fig=None,show=True,title="SMPL joints"):
    """
    Visualize smpl joints on the same plot.
    :param visualize_body: bool, whether to visualize the body or not.
    :param fig: plotly Figure object, if None, create new figure.
    """

    betas = torch.zeros((1, 10), dtype=torch.float32)
    
    smpl_model =  smplx.create(model_path="data",
                                model_type="smpl",
                                gender="NEUTRAL", 
                                use_face_contour=False,
                                num_betas=10,
                                ext='pkl')
    
    smpl_model = smpl_model(betas=betas, return_verts=True)
    smpl_joints = smpl_model.joints.detach().numpy()[0]
    smpl_joints_pelvis = smpl_joints[0,:]
    smpl_joints = smpl_joints - smpl_joints_pelvis
    smpl_vertices = smpl_model.vertices.detach().numpy()[0]
    smpl_vertices = smpl_vertices - smpl_joints_pelvis
    smpl_faces = smplx.SMPL("mhmr/data/smpl",ext="pkl").faces


    joint_colors = px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   ["#000000"]
    
    if isinstance(fig,type(None)):
        fig = go.Figure()

    for i in range(smpl_joints.shape[0]):

        if i in SMPL_IND2JOINT.keys():
            joint_name = SMPL_IND2JOINT[i]
        else:
            joint_name = f"noname-{i}"

        joint_plot = go.Scatter3d(x = [smpl_joints[i,0]],
                                    y = [smpl_joints[i,1]], 
                                    z = [smpl_joints[i,2]], 
                                    mode='markers',
                                    marker=dict(size=10,
                                                color=joint_colors[i],
                                                opacity=1,
                                                symbol="cross"
                                                ),
                                    name="smpl-"+joint_name
                                        )


        fig.add_trace(joint_plot)

    if visualize_body:
        plot_body = go.Mesh3d(
                            x=smpl_vertices[:,0],
                            y=smpl_vertices[:,1],
                            z=smpl_vertices[:,2],
                            #facecolor=face_colors,
                            color = "blue",
                            i=smpl_faces[:,0],
                            j=smpl_faces[:,1],
                            k=smpl_faces[:,2],
                            name='smpl mesh',
                            showscale=True,
                            opacity=0.5
                        )
        fig.add_trace(plot_body)

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title,
                        )
    if show:
        fig.show()
    else:
        return fig
                           

def viz_face_segmentation(verts,faces,face_colors,
                          title="Segmented body",name="mesh",show=True):
    """
    Visualize face segmentation defined in face_colors.
    :param verts: np.ndarray - (N,3) representing the vertices
    :param faces: np.ndarray - (F,3) representing the indices of the faces
    :param face_colors: np.ndarray - (F,3) representing the colors of the faces
    """

    fig = go.Figure()
    mesh_plot = go.Mesh3d(
            x=verts[:,0],
            y=verts[:,1],
            z=verts[:,2],
            facecolor=face_colors,
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            name=name,
            showscale=True,
            opacity=1
        )
    fig.add_trace(mesh_plot)
    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title)
    
    if show:
        fig.show()
    else:
        return fig


def viz_smpl_face_segmentation(fig=None, show=True, title="SMPL face segmentation"):
    body = smplx.SMPL("mhmr/data/smpl",ext="pkl")

    with open("mhmr/data/smpl/smpl_body_parts_2_faces.json","r") as f:
        face_segmentation = json.load(f) 

    faces = body.faces
    verts = body.v_template

    # create colors for each face
    colors = px.colors.qualitative.Alphabet + \
            px.colors.qualitative.Dark24
    mapping_bp2ind = dict(zip(face_segmentation.keys(),
                            range(len(face_segmentation.keys()))
                            ))
    face_colors = [0]*faces.shape[0]
    for bp_name,bp_indices in face_segmentation.items():
        bp_label = mapping_bp2ind[bp_name]
        for i in bp_indices:
            face_colors[i] = colors[bp_label]

    if isinstance(fig,type(None)):
        fig = go.Figure()

    fig = viz_face_segmentation(verts,faces,face_colors,title=title,name="smpl",show=False)

    if show:
        fig.show()
    else:
        return fig


def viz_smplx_face_segmentation(fig=None,show=True,title="SMPLX face segmentation"):
    """
    Visualize face segmentations for smplx.
    """
    
    body = smplx.SMPLX("mhmr/data/smplx",ext="pkl")
    
    with open("mhmr/data/smplx/smplx_body_parts_2_faces.json","r") as f:
        face_segmentation = json.load(f) 


    faces = body.faces
    verts = body.v_template

    # create colors for each face
    colors = px.colors.qualitative.Alphabet + \
            px.colors.qualitative.Dark24
    mapping_bp2ind = dict(zip(face_segmentation.keys(),
                            range(len(face_segmentation.keys()))
                            ))
    face_colors = [0]*faces.shape[0]
    for bp_name,bp_indices in face_segmentation.items():
        bp_label = mapping_bp2ind[bp_name]
        for i in bp_indices:
            face_colors[i] = colors[bp_label]

    if isinstance(fig,type(None)):
        fig = go.Figure()

    fig = viz_face_segmentation(verts,faces,face_colors,title=title,name="smpl",show=False)

    if show:
        fig.show()
    else:
        return fig


def viz_point_segmentation(verts,point_segm,title="Segmented body",fig=None,show=True):
    """
    Visualze points and their segmentation defined in dict point_segm.
    :param verts: np.ndarray - (N,3) representing the vertices
    :param point_segm: dict - dict mapping body part to all points belonging
                                to it
    """
    colors = px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24
    
    if isinstance(fig,type(None)):
        fig = go.Figure()

    for i, (body_part, body_indices) in enumerate(point_segm.items()):
        plot = go.Scatter3d(x = verts[body_indices,0],
                            y = verts[body_indices,1], 
                            z = verts[body_indices,2], 
                            mode='markers',
                            marker=dict(size=5,
                                        color=colors[i],
                                        opacity=1,
                                        #symbol="cross"
                                        ),
                            name=body_part
                                )
        fig.add_trace(plot)
    fig.update_layout(scene_aspectmode='data',
                    width=1000, height=700,
                    title=title)
    if show:
        fig.show()
    return fig


def viz_smplx_point_segmentation(fig=None,show=True,title="SMPLX point segmentation"):
    """
    Visualize point segmentations for smplx.
    """

    model_path = "mhmr/data/smplx"
    smpl_verts = smplx.SMPLX(model_path,ext="pkl").v_template
    with open("mhmr/data/smplx/point_segmentation_meshcapade.json","r") as f:
        point_segm = json.load(f)
    fig = viz_point_segmentation(smpl_verts,point_segm,title=title,fig=fig,show=show)

    if show:
        fig.show()
    else:
        return fig


def viz_smpl_point_segmentation(fig=None,show=True,title="SMPL point segmentation"):
    """
    Visualize point segmentations for smpl.
    """

    model_path = "mhmr/data/smpl"
    smpl_verts = smplx.SMPL(model_path,ext="pkl").v_template
    with open("mhmr/data/smpl/point_segmentation_meshcapade.json","r") as f:
        point_segm = json.load(f)
    fig = viz_point_segmentation(smpl_verts,point_segm,title=title,fig=fig,show=show)

    if show:
        fig.show()
    else:
        return fig


def viz_landmarks(verts,landmark_dict,title="Visualize landmarks",fig=None,show=True,name="points"):
    
    if isinstance(fig,type(None)):
        fig = go.Figure()

    plot = go.Scatter3d(x = verts[:,0],
                        y = verts[:,1], 
                        z = verts[:,2], 
                        mode='markers',
                        hovertemplate ='<i>Index</i>: %{text}',
                        text = [i for i in range(verts.shape[0])],
                        marker=dict(size=5,
                                    color="black",
                                    opacity=0.2,
                                    # line=dict(color='black',width=1)
                                    ),
                        name=name
                            )
    
    fig.add_trace(plot)

    colors = px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24  + \
             px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24
    
    for i, (lm_name, lm_ind) in enumerate(landmark_dict.items()):
        plot = go.Scatter3d(x = [verts[lm_ind,0]],
                            y = [verts[lm_ind,1]], 
                            z = [verts[lm_ind,2]], 
                            mode='markers',
                            marker=dict(size=10,
                                        color=colors[i],
                                        opacity=1,
                                        symbol="cross"
                                        ),
                            name=name+"-"+lm_name
                                )
        fig.add_trace(plot)

    fig.update_layout(scene_aspectmode='data',
                    width=1000, height=700,
                    title=title)

    if show:
        fig.show()
    else:
        return fig
    

def viz_smpl_landmarks(fig=None,show=True,title="SMPL landmarks"):
    """
    Visualize smpl landmarks.
    """

    verts = smplx.SMPL("mhmr/data/smpl",ext="pkl").v_template
    landmark_dict = SMPL_LANDMARK_INDICES

    if isinstance(fig,type(None)):
        fig=go.Figure()

    fig = viz_landmarks(verts,
                        landmark_dict,
                        title="Visualize landmarks",
                        fig=fig,
                        show=show,
                        name="smpl")

    if show:
        fig.show()
    else:
        return fig


def viz_smplx_landmarks(fig=None,show=True,title="SMPLX landmarks"):
    """
    Visualize smplx landmarks.
    """

    verts = smplx.SMPLX("mhmr/data/smplx",ext="pkl").v_template
    landmark_dict = SMPLX_LANDMARK_INDICES

    if isinstance(fig,type(None)):
        fig=go.Figure()

    fig = viz_landmarks(verts,
                        landmark_dict,
                        title="Visualize landmarks",
                        fig=fig,
                        show=show,
                        name="smplx")

    if show:
        fig.show()
    else:
        return fig

def set_shape(model, shape_coefs):
    '''
    Set shape of body model.
    :param model: smplx body model
    :param shape_coefs: torch.tensor dim (10,)

    Return
    shaped smplx body model
    '''
    device = next(model.parameters()).device
    shape_coefs = shape_coefs.to(device)
    return model(betas=shape_coefs, return_verts=True)

def create_model(model_type, model_root, gender, num_betas=10, num_thetas=24):
    '''
    Create SMPL/SMPLX/etc. body model
    :param model_type: str of model type: smpl, smplx, etc.
    :param model_root: str of location where there are smpl/smplx/etc. folders with .pkl models
                        (clumsy definition in smplx package)
    :param gender: str of gender: MALE or FEMALE or NEUTRAL
    :param num_betas: int of number of shape coefficients
                      requires the model with num_coefs in model_root
    :param num_thetas: int of number of pose coefficients
    
    Return:
    :param smplx body model (SMPL, SMPLX, etc.)
    '''
    
    #body_pose = torch.zeros((1, (num_thetas-1) * 3))
    return smplx.create(model_path=model_root,
                        model_type=model_type,
                        gender=gender, 
                        use_face_contour=False,
                        num_betas=num_betas,
                        #body_pose=body_pose,
                        ext='pkl')



class Measurer():
    '''
    Measure a parametric body model defined either.
    Parent class for Measure{SMPL,SMPLX,..}.

    All the measurements are expressed in cm.
    '''

    def __init__(self):
        self.verts = None
        self.faces = None
        self.joints = None
        self.gender = None

        self.measurements = {}
        self.height_normalized_measurements = {}
        self.labeled_measurements = {}
        self.height_normalized_labeled_measurements = {}
        self.labels2names = {}

    def from_verts(self):
        pass

    def from_body_model(self):
        pass

    def measure(self, 
                measurement_names: List[str]
                ):
        '''
        Measure the given measurement names from measurement_names list
        :param measurement_names - list of strings of defined measurements
                                    to measure from MeasurementDefinitions class
        '''

        for m_name in measurement_names:
            if m_name not in self.all_possible_measurements:
                print(f"Measurement {m_name} not defined.")
                pass

            if m_name in self.measurements:
                pass

            if self.measurement_types[m_name] == MeasurementType().LENGTH:

                value = self.measure_length(m_name)
                self.measurements[m_name] = value

            elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:

                value = self.measure_circumference(m_name)
                self.measurements[m_name] = value
    
            else:
                print(f"Measurement {m_name} not defined")

    def measure_length(self, measurement_name: str):
        '''
        Measure distance between 2 landmarks
        :param measurement_name: str - defined in MeasurementDefinitions

        Returns
        :float of measurement in cm
        '''

        measurement_landmarks_inds = self.length_definitions[measurement_name]

        landmark_points = []
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i],tuple):
                # if touple of indices for landmark, take their average
                lm = (self.verts[measurement_landmarks_inds[i][0]] + 
                          self.verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm = self.verts[measurement_landmarks_inds[i]]
            
            landmark_points.append(lm)

        landmark_points = np.vstack(landmark_points)[None,...]

        return self._get_dist(landmark_points)

    @staticmethod
    def _get_dist(verts: np.ndarray) -> float:
        '''
        The Euclidean distance between vertices.
        The distance is found as the sum of each pair i 
        of 3D vertices (i,0,:) and (i,1,:) 
        :param verts: np.ndarray (N,2,3) - vertices used 
                        to find distances
        
        Returns:
        :param dist: float, sumed distances between vertices
        '''

        verts_distances = np.linalg.norm(verts[:, 1] - verts[:, 0],axis=1)
        distance = np.sum(verts_distances)
        distance_cm = distance * 100 # convert to cm
        return distance_cm
    
    def measure_circumference(self, 
                              measurement_name: str, 
                              ):
        '''
        Measure circumferences. Circumferences are defined with 
        landmarks and joints - the measurement is found by cutting the 
        SMPL model with the  plane defined by a point (landmark point) and 
        normal (vector connecting the two joints).
        :param measurement_name: str - measurement name

        Return
        float of measurement value in cm
        '''

        measurement_definition = self.circumf_definitions[measurement_name]
        circumf_landmarks = measurement_definition["LANDMARKS"]
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = self.joint2ind[circumf_n1], self.joint2ind[circumf_n2]
        
        plane_origin = np.mean(self.verts[circumf_landmark_indices,:],axis=0)
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

        # new version            
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh, 
                                plane_normal=plane_normal, 
                                plane_origin=plane_origin, 
                                return_faces=True) # (N, 2, 3), (N,)
        
        slice_segments = filter_body_part_slices(slice_segments,
                                                 sliced_faces,
                                                 measurement_name,
                                                 self.circumf_2_bodypart,
                                                 self.face_segmentation)
        
        slice_segments_hull = convex_hull_from_3D_points(slice_segments)

        return self._get_dist(slice_segments_hull)

    def height_normalize_measurements(self, new_height: float):
        ''' 
        Scale all measurements so that the height measurement gets
        the value of new_height:
        new_measurement = (old_measurement / old_height) * new_height
        NOTE the measurements and body model remain unchanged, a new
        dictionary height_normalized_measurements is created.
        
        Input:
        :param new_height: float, the newly defined height.

        Return:
        self.height_normalized_measurements: dict of 
                {measurement:value} pairs with 
                height measurement = new_height, and other measurements
                scaled accordingly
        '''
        if self.measurements != {}:
            old_height = self.measurements["height"]
            for m_name, m_value in self.measurements.items():
                norm_value = (m_value / old_height) * new_height
                self.height_normalized_measurements[m_name] = norm_value

            if self.labeled_measurements != {}:
                for m_name, m_value in self.labeled_measurements.items():
                    norm_value = (m_value / old_height) * new_height
                    self.height_normalized_labeled_measurements[m_name] = norm_value

    def label_measurements(self,set_measurement_labels: Dict[str, str]):
        '''
        Create labeled_measurements dictionary with "label: x cm" structure
        for each given measurement.
        NOTE: This overwrites any prior labeling!
        
        :param set_measurement_labels: dict of labels and measurement names
                                        (example. {"A": "head_circumference"})
        '''

        if self.labeled_measurements != {}:
            print("Overwriting old labels")

        self.labeled_measurements = {}
        self.labels2names = {}

        for set_label, set_name in set_measurement_labels.items():
            
            if set_name not in self.all_possible_measurements:
                print(f"Measurement {set_name} not defined.")
                pass

            if set_name not in self.measurements.keys():
                self.measure([set_name])

            self.labeled_measurements[set_label] = self.measurements[set_name]
            self.labels2names[set_label] = set_name

    def visualize(self,
                 measurement_names: List[str] = [], 
                 landmark_names: List[str] = [],
                 title="Measurement visualization",
                 visualize_body: bool = True,
                 visualize_landmarks: bool = True,
                 visualize_joints: bool = True,
                 visualize_measurements: bool=True):

        # TODO: create default model if not defined
        # if self.verts is None:
        #     print("Model has not been defined. \
        #           Visualizing on default male model")
        #     model = create_model(self.smpl_path, "MALE", num_coefs=10)
        #     shape = torch.zeros((1, 10), dtype=torch.float32)
        #     model_output = set_shape(model, shape)
            
        #     verts = model_output.vertices.detach().cpu().numpy().squeeze()
        #     faces = model.faces.squeeze()
        # else:
        #     verts = self.verts
        #     faces = self.faces 

        if measurement_names == []:
            measurement_names = self.all_possible_measurements

        if landmark_names == []:
            landmark_names = list(self.landmarks.keys())

        vizz = Visualizer(verts=self.verts,
                        faces=self.faces,
                        joints=self.joints,
                        landmarks=self.landmarks,
                        measurements=self.measurements,
                        measurement_types=self.measurement_types,
                        length_definitions=self.length_definitions,
                        circumf_definitions=self.circumf_definitions,
                        joint2ind=self.joint2ind,
                        circumf_2_bodypart=self.circumf_2_bodypart,
                        face_segmentation=self.face_segmentation,
                        visualize_body=visualize_body,
                        visualize_landmarks=visualize_landmarks,
                        visualize_joints=visualize_joints,
                        visualize_measurements=visualize_measurements,
                        title=title
                        )
        
        vizz.visualize(measurement_names=measurement_names,
                       landmark_names=landmark_names,
                       title=title)


class MeasureSMPL(Measurer):
    '''
    Measure the SMPL model defined either by the shape parameters or
    by its 6890 vertices. 

    All the measurements are expressed in cm.
    '''

    def __init__(self):
        
        super().__init__()

        self.base_dir = "mhmr"
        self.model_type = "smpl"
        self.body_model_root = self.base_dir + "/models"
        self.body_model_path = os.path.join(self.body_model_root, 
                                            self.model_type)

        self.faces = smplx.SMPL(self.body_model_path, ext="pkl").faces
        face_segmentation_path = os.path.join(self.body_model_path,
                                              f"{self.model_type}_body_parts_2_faces.json")
        self.face_segmentation = load_face_segmentation(face_segmentation_path)

        self.landmarks = SMPL_LANDMARK_INDICES
        self.measurement_types = MEASUREMENT_TYPES
        self.length_definitions = SMPLMeasurementDefinitions().LENGTHS
        self.circumf_definitions = SMPLMeasurementDefinitions().CIRCUMFERENCES
        self.circumf_2_bodypart = SMPLMeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        self.all_possible_measurements = SMPLMeasurementDefinitions().possible_measurements

        self.joint2ind = SMPL_JOINT2IND
        self.num_joints = SMPL_NUM_JOINTS

        self.num_points = 6890

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (6890,3) of SMPL vertices
        '''        

        verts = verts.squeeze()
        error_msg = f"verts need to be of dimension ({self.num_points},3)"
        assert verts.shape == torch.Size([self.num_points,3]), error_msg

        joint_regressor = get_joint_regressor(self.model_type, 
                                              self.body_model_root,
                                              gender="NEUTRAL", 
                                              num_thetas=self.num_joints)

        joints = torch.matmul(joint_regressor, verts)
        self.joints = joints.cpu().numpy()
        self.verts = verts.cpu().numpy()

    def from_body_model(self,
                        gender: str,
                        shape: torch.tensor):
        '''
        Construct body model from given gender and shape params 
        of SMPl model.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
                                    for SMPL model
        '''  

        model = create_model(model_type=self.model_type, 
                             model_root=self.body_model_root, 
                             gender=gender,
                             num_betas=10,
                             num_thetas=self.num_joints)    
        model_output = set_shape(model, shape)
        
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender


class MeasureSMPLX(Measurer):
    '''
    Measure the SMPLX model defined either by the shape parameters or
    by its 10475 vertices. 

    All the measurements are expressed in cm.
    '''

    def __init__(self):
        
        super().__init__()

        self.base_dir = "mhmr"
        self.model_type = "smplx"
        self.body_model_root = self.base_dir + "/models"
        self.body_model_path = os.path.join(self.body_model_root, 
                                            self.model_type)

        self.faces = smplx.SMPLX(self.body_model_path, ext="pkl").faces
        face_segmentation_path = os.path.join(self.body_model_path,
                                              f"{self.model_type}_body_parts_2_faces.json")
        self.face_segmentation = load_face_segmentation(face_segmentation_path)

        self.landmarks = SMPLX_LANDMARK_INDICES
        self.measurement_types = MEASUREMENT_TYPES
        self.length_definitions = SMPLXMeasurementDefinitions().LENGTHS
        self.circumf_definitions = SMPLXMeasurementDefinitions().CIRCUMFERENCES
        self.circumf_2_bodypart = SMPLXMeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        self.all_possible_measurements = SMPLXMeasurementDefinitions().possible_measurements

        self.joint2ind = SMPLX_JOINT2IND
        self.num_joints = SMPLX_NUM_JOINTS

        self.num_points = 10475

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (10475,3) of SMPLX vertices
        '''        

        verts = verts.squeeze()
        error_msg = f"verts need to be of dimension ({self.num_points},3)"
        assert verts.shape == torch.Size([self.num_points,3]), error_msg

        joint_regressor = get_joint_regressor(self.model_type, 
                                              self.body_model_root,
                                              gender="NEUTRAL", 
                                              num_thetas=self.num_joints)

        joints = torch.matmul(joint_regressor, verts)
        self.joints = joints.cpu().numpy()
        self.verts = verts.cpu().numpy()

    def from_body_model(self,
                        gender: str,
                        shape: torch.tensor):
        '''
        Construct body model from given gender and shape params 
        of SMPl model.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
                                    for SMPL model
        '''  

        model = create_model(model_type=self.model_type, 
                             model_root=self.body_model_root, 
                             gender=gender,
                             num_betas=10,
                             num_thetas=self.num_joints)    
        model_output = set_shape(model, shape)
        
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender


class MeasureBody():
    def __new__(cls, model_type):
        model_type = model_type.lower()
        if model_type == 'smpl':
            return MeasureSMPL()
        elif model_type == 'smplx':
            return MeasureSMPLX()
        else:
            raise NotImplementedError("Model type not defined")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Measure body models.')
    parser.add_argument('--measure_neutral_smpl_with_mean_shape', action='store_true',
                        help="Measure a mean shape smpl model.")
    parser.add_argument('--measure_neutral_smplx_with_mean_shape', action='store_true',
                        help="Measure a mean shape smplx model.")
    args = parser.parse_args()

    model_types_to_measure = []
    if args.measure_neutral_smpl_with_mean_shape:
        model_types_to_measure.append("smpl")
    elif args.measure_neutral_smplx_with_mean_shape:
        model_types_to_measure.append("smplx")

    for model_type in model_types_to_measure:
        print(f"Measuring {model_type} body model")
        measurer = MeasureBody(model_type)

        betas = torch.zeros((1, 10), dtype=torch.float32)
        measurer.from_body_model(gender="NEUTRAL", shape=betas)

        measurement_names = measurer.all_possible_measurements
        measurer.measure(measurement_names)
        print("Measurements")
        pprint(measurer.measurements)

        measurer.label_measurements(STANDARD_LABELS)
        print("Labeled measurements")
        pprint(measurer.labeled_measurements)

        measurer.visualize()