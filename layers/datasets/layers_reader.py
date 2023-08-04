from collections import defaultdict
from itertools import chain
import os
import numpy as np
from layers.datasets.utils.io import readJSON, readPKL, writePKL
from layers.datasets.smpl.smpl_np import SMPLModel, NUM_SMPL_JOINTS


META_FN = 'meta.json'

GARMENT_TYPE = [
    'jacket', 
    'jacket_hood',
    'jumpsuit_sleeveless', 
    'tee', 
    'tee_sleeveless',
    'dress_sleeveless', 
    'wb_dress_sleeveless',
    'wb_pants_straight', 
    'pants_straight_sides',
    'skirt_2_panels', 
    'skirt_4_panels', 
    'skirt_8_panels',
]

class LayersReader:
    def __init__(self,
        data_root, smpl_dir, split_meta='train_val_test.json', phase='train'):
        self.data_dir = data_root
        self.seq_list = []
        split_meta = readJSON(os.path.join(data_root, 'data', split_meta))
        if isinstance(phase, list):
            phase_meta = list(chain(*[split_meta[i] for i in phase]))
        else:
            phase_meta = split_meta[phase]
        for seq_num in phase_meta:
            self.seq_list.append(seq_num)

        self.smpl = {
			'female': SMPLModel(os.path.join(smpl_dir, 'model_f.pkl')),
			'male': SMPLModel(os.path.join(smpl_dir, 'model_m.pkl'))
		}

    """ 
	Read sample info 
	Input:
	- sample: name of the sample e.g.:'01_01_s0'
	"""
    def read_info(self, sample):
        info_path = os.path.join(self.data_dir, 'data', sample, META_FN)
        infos = readJSON(info_path)
        return infos
        
    """ Human data """
    """
	Read SMPL parameters for the specified sample and frame
	Inputs:
	- sample: name of the sample
	- frame: frame number
	"""
    def read_smpl_params(self, sample, frame):
		# Read sample data
        info = self.read_info(sample)
		# SMPL parameters
        gender = info['human']['gender']
        # TODO: Change poses file location
        motion = readPKL(os.path.join(self.data_dir, info['human']['motion_path']))
        pose = motion['poses']
        trans = motion['trans']
        frame += info['human']['seq_start']
        if len(pose) == 1: frame = None
        pose = pose[frame].reshape(self.smpl[gender].pose_shape)
        shape = np.array(info['human']['betas'])
        # TODO: since when generating, the scale is not applied to the trans
        trans = trans[frame].reshape(self.smpl[gender].trans_shape) * info['human']['scale']
        return gender, pose, shape, trans
	
    """
	Computes human mesh for the specified sample and frame
	Inputs:
	- sample: name of the sample
	- frame: frame number
	Outputs:
	- V: human mesh vertices
	- F: mesh faces
	"""
    def read_human(self, sample, frame=None):
		# Read sample data
        info = self.read_info(sample)
        assert frame is not None
        gender, pose, shape, trans = self.read_smpl_params(sample, frame)
        # Compute SMPL
        _, V, root_offset = self.smpl[gender].set_params(pose=pose, beta=shape, trans=trans/info['human']['scale'], with_body=True)
        root_offset *= info['human']['scale']
        V *= info['human']['scale']
        F = self.smpl[gender].faces.copy()
        V += root_offset
        return V, F
    
    def read_human_joints(self, sample, frame, with_body=False):
        '''
            smpl output verts without root offset.
            When generating data using blender, we add root offset
        
        '''
        # Read sample data
        info = self.read_info(sample)
        # compute
        # SMPL parameters
        gender, pose, shape, trans = self.read_smpl_params(sample, frame)
        # Compute SMPL
        ## Only for clothenv to compute the smpl
        G, V, root_offset = self.smpl[gender].set_params(pose=pose, beta=shape, trans=trans/info['simulate']['human']['scale'], with_body=with_body)
        J = self.smpl[gender].J.copy()
        weights = self.smpl[gender].weights.copy()
        # TODO: cuz is scaled in clothenv
        V = V * info['simulate']['human']['scale']
        root_offset *= info['simulate']['human']['scale']
        return J, G, V, weights, root_offset

    def read_human_rest(self, sample, with_body=True, info=None):
        # Read sample data
        if info == None:
            info = self.read_info(sample)
        # compute
        # SMPL parameters
        gender, pose, shape, trans = self.read_smpl_params(sample, 0)
        # Compute SMPL
        ## Only for clothenv to compute the smpl
        smpl_model = self.smpl[gender]
        G, V, root_offset = smpl_model.set_params(pose=smpl_model.rest_pose, beta=shape, trans=trans, with_body=with_body)
        V -= trans
        V = V * info['human']['scale']
        return V
    
    """ Garment data """
    """
	Reads garment vertices location for the specified sample, garment and frame
    and the faces
	Inputs:
	- sample: name of the sample
	- garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
	- frame: frame number
	- absolute: True for absolute vertex locations, False for locations relative to SMPL root joint	
	Outputs:
	- V: 3D vertex locations for the specified sample, garment and frame
    - F: mesh faces	
	"""
    def read_garment_vertices_topology(self, sample, garment, frame):
		# Read garment vertices (relative to root joint)
        garment_path = os.path.join(self.data_dir, 'data', sample, garment + '.pkl')
        garment_seq = readPKL(garment_path)
        V = garment_seq['vertices'][frame]
        F = garment_seq['faces']
        # TODO: clean tpose data
        T = garment_seq['tpose']
        return V, F, T

    """	
	Reads garment UV map for the specified sample and garment
	Inputs:
	- sample: name of the sample
	- garment: type of garment (e.g.: 'Tshirt', 'Jumpsuit', ...)
	Outputs:
	- Vt: UV map vertices
	- Ft: UV map faces		
	"""
    def read_garment_UVMap(self, sample, garment):
		# Read OBJ file
        uv_path = os.path.join(self.data_dir, 'data', sample, f"uv_{garment}.pkl")
        uv_groups = readPKL(uv_path)
        return uv_groups
	
    def read_wind(self, sample, frame, info=None):
		# Read garment vertices (relative to root joint)
        if info is None:
            info = self.read_info(sample)
        wind_info = info['wind']
        seq_length = info['human']['seq_end'] - info['human']['seq_start']
        outer_forces_idx = [
                o['frame_start'] - info['human']['frame_start']
                for o in wind_info[0]['pivot_list']] \
                    + [seq_length]

        if frame < outer_forces_idx[0]:
            w_info = np.zeros(5)
        else:
            for o_idx in range(1, len(outer_forces_idx)):
                assert outer_forces_idx[o_idx-1] <= outer_forces_idx[o_idx]
                if frame < outer_forces_idx[o_idx] and frame >= outer_forces_idx[o_idx-1]:
                    # find it
                    w_meta = wind_info[0]['pivot_list'][o_idx-1]
                    rotations = w_meta['rotation_quaternion']
                    strengths = np.array(w_meta['strength']).reshape(1)
                    w_info = np.concatenate([rotations, strengths], axis=0)
                    break
        return w_info
        