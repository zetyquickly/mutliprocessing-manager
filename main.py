import os
from io import BytesIO
import itertools
import time
import random

import numpy as np
import trimesh
import py7zlib
import igl
from tqdm import tqdm

from multiprocessing import Pool, Manager

class MeshesArchive(object):
    def __init__(self, archive_path):
        fp = open(archive_path, 'rb')
        self.archive = py7zlib.Archive7z(fp)
        self.archive_path = archive_path
        self.names_list = self.archive.getnames()
        
    def __len__(self):
        return len(self.names_list)
    
    def get(self, name):
        bytes_io = BytesIO(self.archive.getmember(name).read())
        return bytes_io

    def __getitem__(self, idx):
        return self.get(self.names[idx])
    
    def __iter__(self):
        for name in self.names_list:
            yield self.get(name)


def load_mesh(obj_file):
    mesh = trimesh.load(obj_file, 'obj')
    return mesh


def load_mesh_unpack(args):
    obj_file, shared_list = args[0], args[1]
    mesh = load_mesh(obj_file)
    # shared_list.append(mesh)
    return mesh


def get_max_dist(meshes, ids, shared_list):
    base_mesh, point_cloud = meshes[0], np.array(meshes[1].vertices)
    base_id, pc_id = ids[0], ids[1]
    distance_sq, mesh_face_indexes, _ = igl.point_mesh_squared_distance(
        point_cloud,
        base_mesh.vertices,
        base_mesh.faces
    )
    shared_list.append((distance_sq.max(), base_id, pc_id, ))


def get_max_dist_unpack(args):
    meshes, ids, shared_list = args[0], args[1], args[2]
    get_max_dist(meshes, ids, shared_list)

def read_meshes_7z_pool_w_manager(archive_path, num_proc, num_iterations):
    # do the meshes loading within a pool with Manager
    elapsed_time = []
    pool = Pool(num_proc)
    manager = Manager()
    for _ in range(num_iterations):
        archive = MeshesArchive(archive_path)
        loaded_items = manager.list()
        start = time.time()
        result = pool.map(
            load_mesh_unpack, 
            zip(archive, itertools.repeat(loaded_items))
        )
        end = time.time()
        elapsed_time.append(end - start)
    pool.close()
    pool.join()
    print(f'[Loading meshes: pool + manager] Pool of {num_proc} processes elapsed time: {np.array(elapsed_time).mean()} sec')
    return result

def read_meshes_7z_loop(archive_path, num_proc, num_iterations):
    # do the meshes loading in a loop
    elapsed_time = []
    for _ in range(num_iterations):
        archive = MeshesArchive(archive_path)
        start = time.time()
        shared_list = list(map(load_mesh, archive))
        end = time.time()
        elapsed_time.append(end - start)
    print(f'[Loading meshes: loop] Loop elapsed time: {np.array(elapsed_time).mean()} sec')
    return shared_list

def read_meshes_7z_pool_no_manager(archive_path, num_proc, num_iterations):
    # do the meshes loading within a pool
    elapsed_time = []
    pool = Pool(num_proc)
    for _ in range(num_iterations):
        archive = MeshesArchive(archive_path)
        start = time.time()
        result = pool.map(
            load_mesh,
            archive,
        )
        end = time.time()
        elapsed_time.append(end - start)
    pool.close()
    pool.join()
    print(f'[Loading meshes: pool + no manager] Pool of {num_proc} processes elapsed time: {np.array(elapsed_time).mean()} sec')
    return result

if __name__ == "__main__":
    archive_path = "./data/meshes_big.7z"
    num_proc = 3
    num_iterations = 5

    read_meshes_7z_pool_w_manager(archive_path, num_proc, num_iterations)
    read_meshes_7z_loop(archive_path, num_proc, num_iterations)
    read_meshes_7z_pool_no_manager(archive_path, num_proc, num_iterations)

    import pdb; pdb.set_trace()
    
    # do computation of maximum distances between pointclouds and meshes in a pool
    start = time.time()
    pool = Pool(num_proc)
    pc_mesh_max_distances = manager.list()
    args_len = len(loaded_items) * (len(loaded_items) - 1) // 2
    result = list(tqdm(pool.imap(
        get_max_dist_unpack, 
        zip(
            itertools.combinations(loaded_items, 2),
            itertools.combinations(range(len(loaded_items)), 2), 
            itertools.repeat(pc_mesh_max_distances)
        )
    ), total=args_len))
    pool.close()
    pool.join()
    end = time.time()
    print(f'[Compute distances] Pool of {num_proc} processes elapsed time: {end - start} sec')

    # do computatin of of maximum distances between pointclouds and meshes in a loop
    start = time.time()
    shared_list = []
    for meshes, ids in tqdm(zip(itertools.combinations(loaded_items, 2),
                                itertools.combinations(range(len(loaded_items)), 2)), total=args_len):
        get_max_dist(meshes, ids, shared_list)

    end = time.time()
    print(f'[Compute distances] Loop elapsed time: {end - start} sec')
    del shared_list


