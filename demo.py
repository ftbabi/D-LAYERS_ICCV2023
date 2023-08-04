import numpy as np
from layers.datasets import LayersReader
from layers.utils import MeshViewer, select_verts_faces


def main():
    # Variables
    data_root = 'data/demo'
    smpl_dir = 'data/smpl'
    sample = '00396'
    frame = 3

    m_viewer = MeshViewer()
    layers_reader = LayersReader(data_root, smpl_dir)
    h_V, h_F = layers_reader.read_human(sample, frame)
    m_viewer.add_mesh(h_V, h_F)
    m_viewer.show()

    g_Ts = []
    g_Fs = []
    infos = layers_reader.read_info(sample)
    for g_cfg in infos['garment']:
        g_V, g_F, T = layers_reader.read_garment_vertices_topology(sample, g_cfg['name'], frame)
        m_viewer.add_mesh(g_V, g_F)
        g_Ts.append(T)
        g_Fs.append(g_F)
    m_viewer.add_mesh(h_V, h_F)
    m_viewer.show()

    h_V_rest = layers_reader.read_human_rest(sample)
    m_viewer.add_mesh(h_V_rest, h_F)
    for T, F in zip(g_Ts, g_Fs):
        m_viewer.add_mesh(T, F)
    m_viewer.show()

    for g_idx, g_cfg in enumerate(infos['garment']):
        uv_groups = layers_reader.read_garment_UVMap(sample, g_cfg['name'])
        for vg_name, uv_map in uv_groups.items():
            Vt_dict = uv_map['vertices'] # dict: idx=uv.co
            Ft = uv_map['faces']
            filtered_V, filtered_F = select_verts_faces(Vt_dict, Ft)
            filtered_V = np.concatenate([np.zeros((filtered_V.shape[0], 1)), filtered_V], axis=-1)
            m_viewer.add_mesh(filtered_V, filtered_F, lightposition=dict(x=2, y=0, z=0))
        m_viewer.show()


if __name__ == '__main__':
    main()