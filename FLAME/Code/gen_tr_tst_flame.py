# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 00:58:39 2021

@author: shima
"""
import pickle
import numpy as np
import chumpy as ch
from os import listdir,makedirs
from os.path import isfile, join
from plyfile import PlyData, PlyElement
import scipy.io as sio


def load_binary_pickle( filepath ):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def load_embedding( file_path ):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = load_binary_pickle( file_path )
    lmk_face_idx = lmk_indexes_dict[ 'lmk_face_idx' ].astype( np.uint32 )
    lmk_b_coords = lmk_indexes_dict[ 'lmk_b_coords' ]
    return lmk_face_idx, lmk_b_coords

# -----------------------------------------------------------------------------

def mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords ):
    """ function: evaluation 3d points given mesh and landmark embedding
    """
    dif1 = ch.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1

if __name__ == "__main__":
    data_path = '..\\CoMA_raw_data\\'
    data_files = [f for f in listdir(data_path) ]#if isfile(join(data_path, f))]
    cnt = 1
    for i in range(len(data_files)):
        sub_data_files = [f for f in listdir(data_path+data_files[i]+'\\')]# if isfile(join(data_path+data_files[i]+'\\', f))]
        for j in range(len(sub_data_files)):
            ply_files = [f for f in listdir(data_path+data_files[i]+'\\'+sub_data_files[j]+'\\') if isfile(join(data_path+data_files[i]+'\\'+sub_data_files[j]+'\\', f))]
            for h in range(len(ply_files)):
                plydata = PlyData.read(data_path+data_files[i]+'\\'+sub_data_files[j]+'\\'+ply_files[h])
                
                mesh_verts = [plydata['vertex'].data['x'], plydata['vertex'].data['y'],plydata['vertex'].data['z']]
                mesh_faces= plydata['face'].data
                mesh_faces2 = []
                mesh_verts2 = []
                for g in range(len(mesh_faces)):
                    t = mesh_faces[g]
                    t=t[0]
                    # t2 = mesh_verts[g]
                    # t2=t2[0]
                    # t=t[0]
                    # print(t)
                    mesh_faces2.append( [t[0].astype(np.uint32),t[1].astype(np.uint32),t[2].astype(np.uint32)])
                    # mesh_verts2.append( [t2[0],t2[1],t2[2]])

                mesh_faces2 = np.array(mesh_faces2)
                mesh_verts = np.array(mesh_verts)
                mesh_verts = np.transpose(mesh_verts)
                # mesh_faces = mesh_faces.astype(np.uint32)
                lmk_face_idx, lmk_b_coords = load_embedding( 'TF_FLAME-master\\data\\flame_static_embedding.pkl' )
                # lmk_face_idx = lmk_face_idx.astype( np.uint32 )
                lmk_3d=mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces2, lmk_face_idx, lmk_b_coords )
                lmk_3d = np.array(lmk_3d)
                sio.savemat('ComA_lmks\\train\\x3d\\'+str(cnt)+'.mat',{'x3d': lmk_3d})
                print(cnt)
                cnt = cnt+1
                                            
    