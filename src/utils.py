import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd


def evaluate_points(points_arr, camera_centers, shape, size_mm=-1, out_file=None):
    assert (len(points_arr) == len(camera_centers))
    df_points = pd.DataFrame()
    df_dists = pd.DataFrame()
    df_eval = pd.DataFrame()
    for index, points in enumerate(points_arr):
        points = np.array(points)
        df = evaluate_point_dists(points, size_mm)
        if index == 0:
            df_dists = df
        else:
            df_dists['distance'] = df['distance']
            df_dists['factor'] = df['factor']

        dists = df_dists['distance']

        df_dists = df_dists.rename(columns={'distance': 'distance (%i)' % (
            index+1), 'factor': 'factor (%i)' % (index+1)})

        area_simple = area_from_points_simple(points, shape)
        area_simple_i = area_from_points_simple(points, shape, True)
        area_poly = area_from_points_poly(points, shape)
        area_expected = (shape[0]-1) * (shape[1]-1) * size_mm ** 2

        df_points['X (%i)' % (index + 1)] = points[:, 0]
        df_points['Y (%i)' % (index + 1)] = points[:, 1]
        df_points['Z (%i)' % (index + 1)] = points[:, 2]
        depths_arr = []
        for c, camera_center in enumerate(camera_centers):
            depths = [distance(np.array(camera_center).ravel(), point)
                      for point in points[:, :3]]
            df_points['Tiefe C%i (%i)' % (c+1, index + 1)] = depths
            depths_arr.append(depths)

        df_eval = pd.concat([df_eval, pd.DataFrame({
            'mean distance': np.mean(dists),
            # 'median distance': np.median(dists),
            'std distance': np.std(dists),
            'max distance': np.max(dists),
            'min distance': np.min(dists),
            'first 7': distance(points[0], points[6]),
            'last 7': distance(points[-7], points[-1]),
            'area expected': area_expected,
            'area simple': area_simple,
            'area simple factor': round(area_simple / area_expected, 4),
            'area simple invers': area_simple_i,
            'area simple invers factor': round(area_simple_i / area_expected, 4),
            'area polygons': area_poly,
            'area polygons factor': round(area_poly / area_expected, 4),
            'C1 mean depth': np.mean(depths_arr[0]),
            'C2 mean depth': np.mean(depths_arr[1]),
            'C1 std depth': np.std(depths_arr[0]),
            'C2 std depth': np.std(depths_arr[1]),
            'C1 std/mean depth': np.std(depths_arr[0]) / np.mean(depths_arr[0]),
            'C2 std/mean depth': np.std(depths_arr[1]) / np.mean(depths_arr[1]),
            'C1 max depth': np.max(depths_arr[0]),
            'C2 max depth': np.max(depths_arr[1]),
            'C1 min depth': np.min(depths_arr[0]),
            'C2 min depth': np.min(depths_arr[1]),
        }, index=[index+1])])

    df_eval = df_eval.T

    print(df_points.to_string(index=True))
    print(df_dists.to_string(index=False))
    print(df_eval.to_string(index=True))
    if out_file is not None:
        df_points.to_markdown('%s-%s.md' % (out_file, 'points'), index=True)
        df_dists.to_markdown('%s-%s.md' % (out_file, 'dists'), index=False)
        df_eval.to_markdown('%s.md' % (out_file))
        pass


def camera_center(projection_matrix: np.ndarray) -> np.ndarray:
    K = projection_matrix

    x = np.linalg.det([K[:, 1], K[:, 2], K[:, 3]])
    y = -np.linalg.det([K[:, 0], K[:, 2], K[:, 3]])
    z = np.linalg.det([K[:, 0], K[:, 1], K[:, 3]])
    h = -np.linalg.det([K[:, 0], K[:, 1], K[:, 2]])

    C = np.array([x, y, z, h])
    return C


def camera_center_inhomogen(projection_matrix: np.ndarray) -> np.ndarray:
    C = camera_center(projection_matrix)
    h = C[3]
    C = np.array([C[0]/h, C[1]/h, C[2]/h])
    return C


def evaluate_point_dists(points, size_mm) -> pd.DataFrame:
    factor = 0
    df_dists = pd.DataFrame(columns=['from', 'to', 'distance', 'factor'])

    for i in range(0, len(points)):
        j = (i+1) % len(points)
        current = points[i]
        next = points[j]
        d = distance(current, next)
        if size_mm > 0:
            factor = d/size_mm
        df_dists = pd.concat([df_dists, pd.DataFrame(
            {'from': i, 'to': j, 'distance': d, 'factor': factor}, index=[i])])
        # print("{i:2} => {j:2}: {d:8.4f} ({factor:.2f})".format(**locals()))

    return df_dists


def plot_3d(points, marker='o', out_file=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = []
    ys = []
    zs = []
    for i, point in enumerate(points):
        x = point[0]
        y = point[1]
        z = point[2]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ax.scatter(x, y, z, marker=marker)
    ax.plot(xs, ys, zs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if out_file is not None:
        plt.savefig(out_file)
    plt.show()


def distance(X1, X2):
    return scipy.spatial.distance.euclidean(X1, X2)
    # return np.linalg.norm(X2[:3]-X1[:3]) # same result


def area_from_points_simple(points, shape: tuple, inverse_border=False) -> float:
    # Creates two triangles, calculates the area for both and adds them
    assert (len(points) == shape[0] * shape[1])
    points = np.array(points)
    points = points[:, :3]
    if inverse_border:
        triangle1 = np.array(
            [points[0], points[shape[0]-1], points[-shape[0]]])
        triangle2 = np.array(
            [points[-1], points[shape[0]-1], points[-shape[0]]])
    else:
        triangle1 = np.array([points[0], points[-1], points[shape[0]-1]])
        triangle2 = np.array([points[0], points[-1], points[-shape[0]]])
    area = area_from_triangle(triangle1)
    area += area_from_triangle(triangle2)
    return area


def area_from_points_poly(points, shape: tuple) -> float:
    # Creates triangles for each neighbored point, calculates the area for all and adds them
    assert (len(points) == shape[0] * shape[1])
    points = np.array(points)
    points = points[:, :3]
    area = 0
    # skip top line, iterate through each line and create 2 triangles per sqaure
    for i in range(0, shape[1]-1):
        for j in range(0, shape[0]):
            current_index = i*shape[1] + j
            top_index = current_index + shape[1]
            triangle1 = np.array(
                [points[current_index], points[current_index + 1], points[top_index]])
            triangle2 = np.array(
                [points[top_index], points[top_index + 1], points[current_index + 1]])
            area += area_from_triangle(triangle1)
            area += area_from_triangle(triangle2)
    return area

# code modified from https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on


def normal(points: np.ndarray) -> np.ndarray:
    # print(points)
    # The cross product of two sides is a normal vector
    return np.cross(points[1] - points[0],
                    points[2] - points[0])


def area_from_triangle(points: np.ndarray) -> float:
    # The norm of the cross product of two sides is twice the area
    res = np.linalg.norm(normal(points)) / 2
    # print(res)
    return res
# end code
