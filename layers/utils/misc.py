import numpy as np


def select_verts_faces(vertices, faces):
    selected_vert_idx = set()
    for fc in faces:
        for v in fc:
            selected_vert_idx.add(v)
    selected_vert_idx = list(sorted(selected_vert_idx))
    map2new_idx = dict()
    for i in range(len(selected_vert_idx)):
        map2new_idx[selected_vert_idx[i]] = i
    
    rst_vertices = np.stack([vertices[i] for i in selected_vert_idx], axis=0)
    rst_faces = []
    for fc in faces:
        new_fc = np.array([map2new_idx[i] for i in fc])
        rst_faces.append(new_fc)
    rst_faces = np.stack(rst_faces, axis=0)
    return rst_vertices, rst_faces