# Visualize OoD datasets
# Usage: python3 visualize_ood_dataset.py --dataset /path/to/ood_dataset --sequence XX [--predictions /path/to/predictions]

import os
import yaml
import argparse
import numpy as np
import vispy
import vispy.scene as scene
import matplotlib.pyplot as plt


############# PARMETERS #############
BGCOLOR = 'black' # 'white' or 'black'
POINT_SIZE = 1
SHARED_CAMERA = True # shared camera across all views to move all views at the same time
FOV = 45        # camera field of view
DISTANCE = 100  # camera distance
#####################################

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize OoD dataset.')
    parser.add_argument('--dataset', type=str, help='Path to the OoD dataset')
    parser.add_argument('--config', type=str, default='config/semantic-kitti.yaml', help='Path to the config file')
    parser.add_argument('--sequence', type=str, help='Sequence number to visualize')
    parser.add_argument('--predictions', type=str, default=None, help='Path to the predictions folder')
    return parser.parse_args()

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

def main(dataset_path, config_path, sequence, predictions=None):
    if 'poss' in dataset_path.lower():
        dataset_path = os.path.join(dataset_path, 'dataset')
    
    # data paths
    velodyne_path = os.path.join(dataset_path, 'sequences', sequence, 'velodyne')
    labels_path = os.path.join(dataset_path, 'sequences', sequence, 'labels')
    preds_path = os.path.join(predictions, 'sequences', sequence, 'predictions') if predictions else None
    scores_path = os.path.join(predictions, 'sequences', sequence, 'scores') if predictions else None

    assert len(os.listdir(velodyne_path)) == len(os.listdir(labels_path)), f'Velodyne and Labels folders have different number of files {len(os.listdir(velodyne_path))} != {len(os.listdir(labels_path))}'
    if preds_path and scores_path:
        assert len(os.listdir(velodyne_path)) == len(os.listdir(preds_path)), f'Velodyne and Predictions folders have different number of files {len(os.listdir(velodyne_path))} != {len(os.listdir(preds_path))}'
        assert len(os.listdir(velodyne_path)) == len(os.listdir(scores_path)), f'Velodyne and Scores folders have different number of files {len(os.listdir(velodyne_path))} != {len(os.listdir(scores_path))}'

    # gather all files names
    files = sorted(os.listdir(velodyne_path))
    print('Number of OoD files:', len(files), end='\n\n')

    # anomaly label
    anomaly_label = 2
    if 'nuscenes' in dataset_path.lower():
        anomaly_label = 100

    # data config
    CONFIG = yaml.safe_load(open(config_path, 'r'))
    learning_map = CONFIG['learning_map']
    learning_map_inv = CONFIG['learning_map_inv']
    color_map = CONFIG['color_map']
    color_map.update({anomaly_label: (255, 255, 255) if BGCOLOR == 'black' else (0, 0, 0)}) # white color for anomaly class

    # setup visualization
    # Create a canvas and add a view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor=BGCOLOR)
    grid = canvas.central_widget.add_grid()

    # Create scatter plot
    view1 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
    grid.add_widget(view1, 0, 0)

    view2 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
    grid.add_widget(view2, 0, 1)

    if predictions:
        view3 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
        grid.add_widget(view3, 0, 2)

    markers1 = scene.visuals.Markers()
    markers2 = scene.visuals.Markers()
    markers3 = scene.visuals.Markers()

    # shared camera across all views
    shared_camera = scene.cameras.TurntableCamera(fov=FOV, azimuth=30, distance=DISTANCE)

    # Set view properties
    view1.camera = shared_camera
    view1.add(markers1)

    view2.camera = shared_camera
    view2.add(markers2)
    
    if predictions:
        view3.camera = shared_camera
        view3.add(markers3)

    # Initialize the current file index
    current_file_index = 0

    # Function to update the point cloud visualization
    def update_pointcloud(increment=0):
        nonlocal current_file_index
        current_file_index = (current_file_index + increment) % len(files)
        # Get the current file
        current_file = sorted(os.listdir(velodyne_path))[current_file_index]

        view1.update()

        # Get the scan and label paths
        scan_path = os.path.join(velodyne_path, current_file)
        label_path = os.path.join(labels_path, '{:06d}.label'.format(int(current_file.split('.')[0])))
        if predictions:
            pred_path = os.path.join(preds_path, '{:06d}.label'.format(int(current_file.split('.')[0])))
            score_path = os.path.join(scores_path, '{:06d}.txt'.format(int(current_file.split('.')[0])))

        # Load scan and label
        pc = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))[:, :3]  # only xyz
        labels_inst = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
        labels = labels_inst & 0xFFFF  # delete high 16 digits binary
        # labels_train = np.vectorize(learning_map.__getitem__)(labels).astype(np.int32)
        if predictions:
            preds_inst = np.fromfile(pred_path, dtype=np.uint32).reshape((-1, 1))
            preds = preds_inst & 0xFFFF  # delete high 16 digits binary
            scores = np.loadtxt(score_path).astype(np.float32)

        # crop point cloud to 50m around the sensor
        dist = np.linalg.norm(pc, axis=1)
        mask = (dist < 50)
        pc = pc[mask]
        labels = labels[mask]
        if predictions:
            preds = preds[mask]
            scores = scores[mask]

            # print(f'Max score: {scores.max():.4f} | Min score: {scores.min():.4f} | Mean score: {scores.mean():.4f}')

        assert pc.shape[0] == labels.shape[0], f'Scan and labels have different number of points {pc.shape[0]} != {labels.shape[0]}'
        if predictions:
            assert pc.shape[0] == preds.shape[0], f'Scan and predictions have different number of points {pc.shape[0]} != {preds.shape[0]}'
            assert pc.shape[0] == scores.shape[0], f'Scan and scores have different number of points {pc.shape[0]} != {scores.shape[0]}'

        color_dict = {k: v[::-1] for k, v in color_map.items()}
        colors_labels = np.array([color_dict[pred[0]] for pred in preds]) if predictions else np.array([color_dict[label[0]] for label in labels])
        colors_labels = colors_labels / 255.0

        where = (labels == anomaly_label).reshape(-1)

        depth = np.linalg.norm(pc[:, :3], axis=1)
        viridis_map = get_mpl_colormap('viridis')
        colors = viridis_map[((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)]
        colors = colors[..., ::-1]
        colors[where] = np.array([1, 0, 0])       # Anomaly - red

        if predictions:
            plasma_map = get_mpl_colormap('plasma')
            score_colors = plasma_map[(scores*255.0).astype(np.uint8)]
            score_colors = score_colors[..., ::-1]

        # print(f'Frame {current_file} | Anomalies = {(labels == anomaly_label).sum()}/{len(pc)}')

        markers1.set_data(pos=pc[:, :3], size=POINT_SIZE, face_color=colors, edge_color=colors)
        markers2.set_data(pos=pc[:, :3], size=POINT_SIZE, face_color=colors_labels, edge_color=colors_labels)
        if predictions:
            markers3.set_data(pos=pc[:, :3], size=POINT_SIZE, face_color=score_colors, edge_color=score_colors)


    # Function to handle key press events
    @canvas.events.key_press.connect
    def on_key_press(event):
        # Check if the 'n' key is pressed
        if event.key == 'n':
            # Update the point cloud visualization
            update_pointcloud(increment=1)
        elif event.key == 'b':
            # Update the point cloud visualization
            update_pointcloud(increment=-1)
        elif event.key == 'q':
            # Close the canvas
            exit()

    # Update the initial point cloud visualization
    update_pointcloud()

    # Run the event loop
    vispy.app.run()

if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset
    config_path = args.config
    predictions = args.predictions
    sequence = args.sequence

    # Add your evaluation code here
    print('--------------- INFO ---------------')
    print('Dataset path: ', dataset_path)
    print('Config path: ', config_path)
    print('Sequence: ', sequence)
    print('Predictions path: ', predictions)
    print('------------------------------------')

    main(dataset_path, config_path, sequence, predictions)