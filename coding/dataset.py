from common import *
from kaggle_utils import *
from config import *

DATA_DIR = config.data_dir

classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]


class LyftDataset:
    """Database class for Lyft Dataset to help query and retrieve information from the database."""

    def __init__(self, data_path: str, json_path: str, verbose: bool = True, map_resolution: float = 0.1, mode = 'train'):
        """Loads database and creates reverse indexes and shortcuts.

        Args:
            data_path: Path to the tables and data.
            json_path: Path to the folder with json files
            verbose: Whether to print status messages during load.
            map_resolution: Resolution of maps (meters).
        """
        self.mode = mode
        self.data_path = Path(data_path).expanduser().absolute()
        self.json_path = Path(json_path)

        self.table_names = [
            "category",
            "attribute",
            "visibility",
            "instance",
            "sensor",
            "calibrated_sensor",
            "ego_pose",
            "log",
            "scene",
            "sample",
            "sample_data",
            "sample_annotation",
            "map",
        ]

        start_time = time.time()

        # Explicitly assign tables to help the IDE determine valid class members.
        self.category = self.__load_table__("category", verbose)
        self.attribute = self.__load_table__("attribute", verbose)
        self.visibility = self.__load_table__("visibility", verbose)
        self.instance = self.__load_table__("instance", verbose, missing_ok=True)
        self.sensor = self.__load_table__("sensor", verbose)
        self.calibrated_sensor = self.__load_table__("calibrated_sensor", verbose)
        self.ego_pose = self.__load_table__("ego_pose", verbose)
        self.log = self.__load_table__("log", verbose)
        self.scene = self.__load_table__("scene", verbose)
        self.sample = self.__load_table__("sample", verbose)
        self.sample_data = self.__load_table__("sample_data", verbose)
        self.sample_annotation = self.__load_table__("sample_annotation", verbose, missing_ok=True)
        self.map = self.__load_table__("map", verbose)

        # Initialize map mask for each map record.
        for map_record in self.map:
            map_record["mask"] = MapMask(self.data_path / '{0}_{1}'.format(self.mode, map_record["filename"]), resolution=map_resolution)

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.1f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize LyftDatasetExplorer class
        self.explorer = LyftDatasetExplorer(self)

    def __load_table__(self, table_name, verbose=False, missing_ok=False) -> dict:
        """Loads a table."""
        filepath = str(self.json_path.joinpath("{}.json".format(table_name)))

        if not os.path.isfile(filepath) and missing_ok:
            if verbose:
                print("JSON file {}.json missing, using empty list".format(table_name))
            return []

        with open(filepath) as f:
            table = json.load(f)
        return table

    def __make_reverse_index__(self, verbose: bool) -> None:
        """De-normalizes database to create reverse indices for common cases.

        Args:
            verbose: Whether to print outputs.

        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member["token"]] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get("instance", record["instance_token"])
            record["category_name"] = self.get("category", inst["category_token"])["name"]

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor_record = self.get("sensor", cs_record["sensor_token"])
            record["sensor_modality"] = sensor_record["modality"]
            record["channel"] = sensor_record["channel"]

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record["data"] = {}
            record["anns"] = []

        for record in self.sample_data:
            if record["is_key_frame"]:
                sample_record = self.get("sample", record["sample_token"])
                sample_record["data"][record["channel"]] = record["token"]

        for ann_record in self.sample_annotation:
            sample_record = self.get("sample", ann_record["sample_token"])
            sample_record["anns"].append(ann_record["token"])

        # Add reverse indices from log records to map records.
        if "log_tokens" not in self.map[0].keys():
            raise Exception("Error: log_tokens not in map table. This code is not compatible with the teaser dataset.")
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record["log_tokens"]:
                log_to_map[log_token] = map_record["token"]
        for log_record in self.log:
            log_record["map_token"] = log_to_map[log_record["token"]]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    def get(self, table_name: str, token: str) -> dict:
        """Returns a record from table in constant runtime.

        Args:
            table_name: Table name.
            token: Token of the record.

        Returns: Table record.

        """

        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """Returns the index of the record in a table in constant runtime.

        Args:
            table_name: Table name.
            token: The index of the record in table, table is an array.

        Returns:

        """
        return self._token2ind[table_name][token]

    def field2token(self, table_name: str, field: str, query) -> List[str]:
        """Query all records for a certain field value, and returns the tokens for the matching records.

        Runs in linear time.

        Args:
            table_name: Table name.
            field: Field name.
            query: Query to match against. Needs to type match the content of the query field.

        Returns: List of tokens for the matching records.

        """
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member["token"])
        return matches

    def get_sample_data_path(self, sample_data_token: str) -> Path:
        """Returns the path to a sample_data.

        Args:
            sample_data_token:

        Returns:

        """

        sd_record = self.get("sample_data", sample_data_token)
        return self.data_path / sd_record["filename"]

    def get_sample_data(
            self,
            sample_data_token: str,
            box_vis_level: BoxVisibility = BoxVisibility.ANY,
            selected_anntokens: List[str] = None,
            flat_vehicle_coordinates: bool = False,
    ) -> Tuple[Path, List[Box], np.array]:
        """Returns the data path as well as all annotations related to that sample_data.
        The boxes are transformed into the current sensor's coordinate frame.

        Args:
            sample_data_token: Sample_data token.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            selected_anntokens: If provided only return the selected annotation.
            flat_vehicle_coordinates: Instead of current sensor's coordinate frame, use vehicle frame which is
        aligned to z-plane in world

        Returns: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)

        """

        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        cs_record = self.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        sensor_record = self.get("sensor", cs_record["sensor_token"])
        pose_record = self.get("ego_pose", sd_record["ego_pose_token"])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record["modality"] == "camera":
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            imsize = (sd_record["width"], sd_record["height"])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane
                ypr = Quaternion(pose_record["rotation"]).yaw_pitch_roll
                yaw = ypr[0]

                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)

            else:
                # Move box to ego vehicle coord system
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)

                #  Move box to sensor coord system
                box.translate(-np.array(cs_record["translation"]))
                box.rotate(Quaternion(cs_record["rotation"]).inverse)

            if sensor_record["modality"] == "camera" and not box_in_image(
                    box, cam_intrinsic, imsize, vis_level=box_vis_level
            ):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_box(self, sample_annotation_token: str) -> Box:
        """Instantiates a Box class from a sample annotation record.

        Args:
            sample_annotation_token: Unique sample_annotation identifier.

        Returns:

        """
        record = self.get("sample_annotation", sample_annotation_token)
        return Box(
            record["translation"],
            record["size"],
            Quaternion(record["rotation"]),
            name=record["category_name"],
            token=record["token"],
        )

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.



        Args:
            sample_data_token: Unique sample_data identifier.

        Returns:

        """

        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        curr_sample_record = self.get("sample", sd_record["sample_token"])

        if curr_sample_record["prev"] == "" or sd_record["is_key_frame"]:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record["anns"]))

        else:
            prev_sample_record = self.get("sample", curr_sample_record["prev"])

            curr_ann_recs = [self.get("sample_annotation", token) for token in curr_sample_record["anns"]]
            prev_ann_recs = [self.get("sample_annotation", token) for token in prev_sample_record["anns"]]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry["instance_token"]: entry for entry in prev_ann_recs}

            t0 = prev_sample_record["timestamp"]
            t1 = curr_sample_record["timestamp"]
            t = sd_record["timestamp"]

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec["instance_token"] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec["instance_token"]]

                    # Interpolate center.
                    center = [
                        np.interp(t, [t0, t1], [c0, c1])
                        for c0, c1 in zip(prev_ann_rec["translation"], curr_ann_rec["translation"])
                    ]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(
                        q0=Quaternion(prev_ann_rec["rotation"]),
                        q1=Quaternion(curr_ann_rec["rotation"]),
                        amount=(t - t0) / (t1 - t0),
                    )

                    box = Box(
                        center,
                        curr_ann_rec["size"],
                        rotation,
                        name=curr_ann_rec["category_name"],
                        token=curr_ann_rec["token"],
                    )
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec["token"])

                boxes.append(box)
        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """Estimate the velocity for an annotation.

        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.

        Args:
            sample_annotation_token: Unique sample_annotation identifier.
            max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.


        Returns: <np.float: 3>. Velocity in x/y/z direction in m/s.

        """

        current = self.get("sample_annotation", sample_annotation_token)
        has_prev = current["prev"] != ""
        has_next = current["next"] != ""

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get("sample_annotation", current["prev"])
        else:
            first = current

        if has_next:
            last = self.get("sample_annotation", current["next"])
        else:
            last = current

        pos_last = np.array(last["translation"])
        pos_first = np.array(first["translation"])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get("sample", last["sample_token"])["timestamp"]
        time_first = 1e-6 * self.get("sample", first["sample_token"])["timestamp"]
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    def list_categories(self) -> None:
        self.explorer.list_categories()

    def list_attributes(self) -> None:
        self.explorer.list_attributes()

    def list_scenes(self) -> None:
        self.explorer.list_scenes()

    def list_sample(self, sample_token: str) -> None:
        self.explorer.list_sample(sample_token)

    def render_pointcloud_in_image(
            self,
            sample_token: str,
            dot_size: int = 5,
            pointsensor_channel: str = "LIDAR_TOP",
            camera_channel: str = "CAM_FRONT",
            out_path: str = None,
    ) -> None:
        self.explorer.render_pointcloud_in_image(
            sample_token,
            dot_size,
            pointsensor_channel=pointsensor_channel,
            camera_channel=camera_channel,
            out_path=out_path,
        )

    def render_sample(
            self,
            sample_token: str,
            box_vis_level: BoxVisibility = BoxVisibility.ANY,
            nsweeps: int = 1,
            out_path: str = None,
    ) -> None:
        self.explorer.render_sample(sample_token, box_vis_level, nsweeps=nsweeps, out_path=out_path)

    def render_sample_data(
            self,
            sample_data_token: str,
            with_anns: bool = True,
            box_vis_level: BoxVisibility = BoxVisibility.ANY,
            axes_limit: float = 40,
            ax: Axes = None,
            nsweeps: int = 1,
            out_path: str = None,
            underlay_map: bool = False,
    ) -> None:
        return self.explorer.render_sample_data(
            sample_data_token,
            with_anns,
            box_vis_level,
            axes_limit,
            ax,
            num_sweeps=nsweeps,
            out_path=out_path,
            underlay_map=underlay_map,
        )

    def render_annotation(
            self,
            sample_annotation_token: str,
            margin: float = 10,
            view: np.ndarray = np.eye(4),
            box_vis_level: BoxVisibility = BoxVisibility.ANY,
            out_path: str = None,
    ) -> None:
        self.explorer.render_annotation(sample_annotation_token, margin, view, box_vis_level, out_path)

    def render_instance(self, instance_token: str, out_path: str = None) -> None:
        self.explorer.render_instance(instance_token, out_path=out_path)

    def render_scene(self, scene_token: str, freq: float = 10, imwidth: int = 640, out_path: str = None) -> None:
        self.explorer.render_scene(scene_token, freq, image_width=imwidth, out_path=out_path)

    def render_scene_channel(
            self,
            scene_token: str,
            channel: str = "CAM_FRONT",
            freq: float = 10,
            imsize: Tuple[float, float] = (640, 360),
            out_path: Path = None,
            interactive: bool = True,
            verbose: bool = False,
    ) -> None:
        self.explorer.render_scene_channel(
            scene_token=scene_token,
            channel=channel,
            freq=freq,
            image_size=imsize,
            out_path=out_path,
            interactive=interactive,
            verbose=verbose,
        )

    def render_egoposes_on_map(self, log_location: str, scene_tokens: List = None, out_path: str = None) -> None:
        self.explorer.render_egoposes_on_map(log_location, scene_tokens, out_path=out_path)

    def render_sample_3d_interactive(
            self,
            sample_id: str,
            render_sample: bool = True
    ) -> None:
        """Render 3D visualization of the sample using plotly

        Args:
            sample_id: Unique sample identifier.
            render_sample: call self.render_sample (Render all LIDAR and camera sample_data in sample along with annotations.)

        """
        import pandas as pd
        import plotly.graph_objects as go

        sample = self.get('sample', sample_id)
        sample_data = self.get(
            'sample_data',
            sample['data']['LIDAR_TOP']
        )
        pc = LidarPointCloud.from_file(
            Path(os.path.join(str(self.data_path),
                              sample_data['filename']))
        )
        _, boxes, _ = self.get_sample_data(
            sample['data']['LIDAR_TOP'], flat_vehicle_coordinates=False
        )

        if render_sample:
            self.render_sample(sample_id)

        df_tmp = pd.DataFrame(pc.points[:3, :].T, columns=['x', 'y', 'z'])
        df_tmp['norm'] = np.sqrt(np.power(df_tmp[['x', 'y', 'z']].values, 2).sum(axis=1))
        scatter = go.Scatter3d(
            x=df_tmp['x'],
            y=df_tmp['y'],
            z=df_tmp['z'],
            mode='markers',
            marker=dict(
                size=1,
                color=df_tmp['norm'],
                opacity=0.8
            )
        )

        x_lines = []
        y_lines = []
        z_lines = []

        def f_lines_add_nones():
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

        ixs_box_0 = [0, 1, 2, 3, 0]
        ixs_box_1 = [4, 5, 6, 7, 4]

        for box in boxes:
            points = view_points(box.corners(), view=np.eye(3), normalize=False)
            x_lines.extend(points[0, ixs_box_0])
            y_lines.extend(points[1, ixs_box_0])
            z_lines.extend(points[2, ixs_box_0])
            f_lines_add_nones()
            x_lines.extend(points[0, ixs_box_1])
            y_lines.extend(points[1, ixs_box_1])
            z_lines.extend(points[2, ixs_box_1])
            f_lines_add_nones()
            for i in range(4):
                x_lines.extend(points[0, [ixs_box_0[i], ixs_box_1[i]]])
                y_lines.extend(points[1, [ixs_box_0[i], ixs_box_1[i]]])
                z_lines.extend(points[2, [ixs_box_0[i], ixs_box_1[i]]])
                f_lines_add_nones()

        lines = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            name='lines'
        )

        fig = go.Figure(data=[scatter, lines])
        fig.update_layout(scene_aspectmode='data')
        fig.show()


class SteelDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):

        self.split = split
        self.csv = csv
        self.mode = mode
        self.augment = augment

        self.uid = list(np.concatenate([np.load(DATA_DIR + '/split/%s' % f, allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(DATA_DIR + '/%s' % f) for f in csv])
        df.fillna('', inplace=True)
        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels'] != '').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId',
                            [u.split('/')[-1] + '_%d' % c for u in self.uid for c in [1, 2, 3, 4]])
        self.df = df

    def __str__(self):
        num1 = (self.df['Class'] == 1).sum()
        num2 = (self.df['Class'] == 2).sum()
        num3 = (self.df['Class'] == 3).sum()
        num4 = (self.df['Class'] == 4).sum()
        pos1 = ((self.df['Class'] == 1) & (self.df['Label'] == 1)).sum()
        pos2 = ((self.df['Class'] == 2) & (self.df['Label'] == 1)).sum()
        pos3 = ((self.df['Class'] == 3) & (self.df['Label'] == 1)).sum()
        pos4 = ((self.df['Class'] == 4) & (self.df['Label'] == 1)).sum()

        length = len(self)
        num = len(self) * 4
        pos = (self.df['Label'] == 1).sum()
        neg = num - pos

        # ---

        string = ''
        string += '\tmode    = %s\n' % self.mode
        string += '\tsplit   = %s\n' % self.split
        string += '\tcsv     = %s\n' % str(self.csv)
        string += '\t\tlen   = %5d\n' % len(self)
        if self.mode == 'train':
            string += '\t\tnum   = %5d\n' % num
            string += '\t\tneg   = %5d  %0.3f\n' % (neg, neg / num)
            string += '\t\tpos   = %5d  %0.3f\n' % (pos, pos / num)
            string += '\t\tpos1  = %5d  %0.3f  %0.3f\n' % (pos1, pos1 / length, pos1 / pos)
            string += '\t\tpos2  = %5d  %0.3f  %0.3f\n' % (pos2, pos2 / length, pos2 / pos)
            string += '\t\tpos3  = %5d  %0.3f  %0.3f\n' % (pos3, pos3 / length, pos3 / pos)
            string += '\t\tpos4  = %5d  %0.3f  %0.3f\n' % (pos4, pos4 / length, pos4 / pos)
        return string

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        # print(index)
        folder, image_id = self.uid[index].split('/')

        rle = [
            self.df.loc[self.df['ImageId_ClassId'] == image_id + '_1', 'EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId'] == image_id + '_2', 'EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId'] == image_id + '_3', 'EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId'] == image_id + '_4', 'EncodedPixels'].values[0],
        ]
        image = cv2.imread(DATA_DIR + '/%s/%s' % (folder, image_id), cv2.IMREAD_COLOR)
        mask = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])

        infor = Struct(
            index=index,
            folder=folder,
            image_id=image_id,
        )

        if self.augment is None:
            return image, mask, infor
        else:
            return self.augment(image, mask, infor)


'''
test_dataset : 
	mode    = train
	split   = ['valid0_500.npy']
	csv     = ['train.csv']
		len   =   500
		neg   =   212  0.424
		pos   =   288  0.576
		pos1  =    35  0.070  0.122
		pos2  =     5  0.010  0.017
		pos3  =   213  0.426  0.740
		pos4  =    35  0.070  0.122
		

train_dataset : 
	mode    = train
	split   = ['train0_12068.npy']
	csv     = ['train.csv']
		len   = 12068
		neg   =  5261  0.436
		pos   =  6807  0.564
		pos1  =   862  0.071  0.127
		pos2  =   242  0.020  0.036
		pos3  =  4937  0.409  0.725
		pos4  =   766  0.063  0.113

		
'''


def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        infor.append(batch[b][2])

    input = np.stack(input).astype(np.float32) / 255
    input = input.transpose(0, 3, 1, 2)
    truth = np.stack(truth)
    truth = (truth > 0.5).astype(np.float32)

    input = torch.from_numpy(input).float()
    truth = torch.from_numpy(truth).float()

    return input, truth, infor


class FourBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['Label'].values)
        label = label.reshape(-1, 4)
        label = np.hstack([label.sum(1, keepdims=True) == 0, label]).T

        self.neg_index = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        # assume we know neg is majority class
        num_neg = len(self.neg_index)
        self.length = 4 * num_neg

    def __iter__(self):
        neg = self.neg_index.copy()
        random.shuffle(neg)
        num_neg = len(self.neg_index)

        pos1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)

        l = np.stack([neg, pos1, pos2, pos3, pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


class FixedSampler(Sampler):

    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index
        self.length = len(index)

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return self.length


##############################################################
#
# class BalanceClassSampler(Sampler):
#
#     def __init__(self, dataset, length=None):
#         self.dataset = dataset
#
#         if length is None:
#             length = len(self.dataset)
#
#         self.length = length
#
#
#
#     def __iter__(self):
#
#         df = self.dataset.df
#         df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
#         df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
#
#
#         label = df['Label'].values*df['Class'].values
#         unique, count = np.unique(label, return_counts=True)
#         L = len(label)//5
#
#
#
#
#         pos_index = np.where(self.dataset.label==1)[0]
#         neg_index = np.where(self.dataset.label==0)[0]
#         half = self.length//2 + 1
#
#
#         neg  = np.random.choice(label==0, [L,6], replace=True)
#         pos1 = np.random.choice(label==1, L, replace=True)
#         pos2 = np.random.choice(label==2, L, replace=True)
#         pos3 = np.random.choice(label==3, L, replace=True)
#         pos4 = np.random.choice(label==4, L, replace=True)
#
#
#         l = np.stack([neg.reshape,pos1,pos2,pos3,pos3,pos4]).T
#         l = l.reshape(-1)
#         l = l[:self.length]
#         return iter(l)
#
#     def __len__(self):
#         return self.length


##############################################################

def image_to_input(image, rbg_mean, rbg_std):  # , rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = image.astype(np.float32)
    input = input[..., ::-1] / 255
    input = input.transpose(0, 3, 1, 2)
    input[:, 0] = (input[:, 0] - rbg_mean[0]) / rbg_std[0]
    input[:, 1] = (input[:, 1] - rbg_mean[1]) / rbg_std[1]
    input[:, 2] = (input[:, 2] - rbg_mean[2]) / rbg_std[2]
    return input


def input_to_image(input, rbg_mean, rbg_std):  # , rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = input.data.cpu().numpy()
    input[:, 0] = (input[:, 0] * rbg_std[0] + rbg_mean[0])
    input[:, 1] = (input[:, 1] * rbg_std[1] + rbg_mean[1])
    input[:, 2] = (input[:, 2] * rbg_std[2] + rbg_mean[2])
    input = input.transpose(0, 2, 3, 1)
    input = input[..., ::-1]
    image = (input * 255).astype(np.uint8)
    return image


##############################################################

def do_random_crop(image, mask, w, h):
    height, width = image.shape[:2]
    x, y = 0, 0
    if width > w:
        x = np.random.choice(width - w)
    if height > h:
        y = np.random.choice(height - h)
    image = image[y:y + h, x:x + w]
    mask = mask[:, y:y + h, x:x + w]
    return image, mask


def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x, y = 0, 0
    if width > w:
        x = np.random.choice(width - w)
    if height > h:
        y = np.random.choice(height - h)
    image = image[y:y + h, x:x + w]
    mask = mask[:, y:y + h, x:x + w]

    # ---
    if (w, h) != (width, height):
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

        mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        mask = mask.transpose(2, 0, 1)

    return image, mask


def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask = mask[:, :, ::-1]
    return image, mask


def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask = mask[:, ::-1, :]
    return image, mask


def do_random_scale_rotate(image, mask, w, h):
    H, W = image.shape[:2]

    # dangle = np.random.uniform(-2.5, 2.5)
    # dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-5, 5)
    dscale = np.random.uniform(-0.15, 0.15, 2)
    dshift = np.random.uniform(0, 1, 2)
    cos = np.cos(dangle / 180 * PI)
    sin = np.sin(dangle / 180 * PI)
    sx, sy = 1 + dscale  # 1,1 #
    tx, ty = dshift

    src = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float32)
    src = src * [sx, sy]
    x = (src * [cos, -sin]).sum(1)
    y = (src * [sin, cos]).sum(1)
    x = x - x.min()
    y = y - y.min()
    x = x + (W - x.max()) * tx
    y = y + (H - y.max()) * ty

    if 0:
        overlay = image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i], y[i]]), int_tuple([x[(i + 1) % 4], y[(i + 1) % 4]]), (0, 0, 255), 5)
        image_show('overlay', overlay)
        cv2.waitKey(0)

    src = np.column_stack([x, y])
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s, d)

    image = cv2.warpPerspective(image, transform, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    mask = mask.transpose(1, 2, 0)
    mask = cv2.warpPerspective(mask, transform, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = mask.transpose(2, 0, 1)
    mask = (mask > 0.5).astype(np.float32)

    return image, mask


def do_random_crop_rotate_rescale(image, mask, w, h):
    H, W = image.shape[:2]

    # dangle = np.random.uniform(-2.5, 2.5)
    # dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-8, 8)
    dshift = np.random.uniform(-0.1, 0.1, 2)

    dscale_x = np.random.uniform(-0.00075, 0.00075)
    dscale_y = np.random.uniform(-0.25, 0.25)

    cos = np.cos(dangle / 180 * PI)
    sin = np.sin(dangle / 180 * PI)
    sx, sy = 1 + dscale_x, 1 + dscale_y  # 1,1 #
    tx, ty = dshift * min(H, W)

    src = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float32)
    src = src * [sx, sy]
    x = (src * [cos, -sin]).sum(1) + W / 2
    y = (src * [sin, cos]).sum(1) + H / 2
    # x = x-x.min()
    # y = y-y.min()
    # x = x + (W-x.max())*tx
    # y = y + (H-y.max())*ty

    if 0:
        overlay = image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i], y[i]]), int_tuple([x[(i + 1) % 4], y[(i + 1) % 4]]), (0, 0, 255), 5)
        image_show('overlay', overlay)
        cv2.waitKey(0)

    src = np.column_stack([x, y])
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s, d)

    image = cv2.warpPerspective(image, transform, (W, H),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    mask = mask.transpose(1, 2, 0)
    mask = cv2.warpPerspective(mask, transform, (W, H),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = mask.transpose(2, 0, 1)

    return image, mask


def do_random_log_contast(image):
    gain = np.random.uniform(0.70, 1.30, 1)
    inverse = np.random.choice(2, 1)

    image = image.astype(np.float32) / 255
    if inverse == 0:
        image = gain * np.log(image + 1)
    else:
        image = gain * (2 ** image - 1)

    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def do_noise(image, mask, noise=8):
    H, W = image.shape[:2]
    image = image + np.random.uniform(-1, 1, (H, W, 1)) * noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, mask


def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.

    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
    """

    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)

    tm = np.eye(4, dtype=np.float32)
    translation = shape / 2 + offset / voxel_size

    tm = tm * np.array(np.hstack((1 / voxel_size, [1])))
    tm[:3, 3] = np.transpose(translation)
    return tm


def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3, 4]:
        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]


def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")

    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p


def create_voxel_pointcloud(points, shape, voxel_size=(0.5, 0.5, 1), z_offset=0):
    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1, 0)
    points_voxel_coords = np.int0(points_voxel_coords)

    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))

    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)

    # Note X and Y are flipped:
    bev[coord[:, 1], coord[:, 0], coord[:, 2]] = count

    return bev


def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev / max_intensity).clip(0, 1)

##############################################################

def run_check_train_dataset():
    dataset = LyftDataset(data_path=config.data_dir, json_path=config.train_data)
    records = [(dataset.get('sample', record['first_sample_token'])['timestamp'], record) for record in
               dataset.scene]

    entries = []

    for start_time, record in sorted(records):
        start_time = dataset.get('sample', record['first_sample_token'])['timestamp'] / 1000000

        token = record['token']
        name = record['name']
        date = datetime.utcfromtimestamp(start_time)
        host = "-".join(record['name'].split("-")[:2])
        first_sample_token = record["first_sample_token"]

        entries.append((host, name, date, token, first_sample_token))

    df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])

    host_count_df = df.groupby("host")['scene_token'].count()
    print(host_count_df)

    # Let's split the data by car to get a validation set.
    validation_hosts = ["host-a007", "host-a008"]

    validation_df = df[df["host"].isin(validation_hosts)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]

    print(len(train_df), len(validation_df), "train/validation split scene counts")

    sample_token = train_df.first_sample_token.values[0]
    sample = dataset.get("sample", sample_token)

    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = dataset.get("sample_data", sample_lidar_token)
    lidar_filepath = dataset.get_sample_data_path(sample_lidar_token)

    ego_pose = dataset.get("ego_pose", lidar_data["ego_pose_token"])
    calibrated_sensor = dataset.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    # Homogeneous transformation matrix from car frame to world frame.
    global_from_car = transform_matrix(ego_pose['translation'],
                                       Quaternion(ego_pose['rotation']), inverse=False)

    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
    car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                       inverse=False)

    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

    # The lidar pointcloud is defined in the sensor's reference frame.
    # We want it in the car's reference frame, so we transform each point
    lidar_pointcloud.transform(car_from_sensor)

    voxel_size = (0.4, 0.4, 1.5)
    z_offset = -2.0
    bev_shape = (336, 336, 3)

    bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)

    # So that the values in the voxels range from 0,1 we set a maximum intensity.
    bev = normalize_voxel_intensities(bev)

    plt.figure(figsize=(16, 8))
    plt.imshow(bev)
    plt.show()

    # the dataset consists of several scences, which are 25-45 second clips of image of LiDAR data from a self-driving car.

    dataset = SteelDataset(
        mode='train',
        csv=['train.csv', ],
        split=['train0_12068.npy', ],
        augment=None,  #
    )
    print(dataset)
    # exit(0)

    for n in range(0, len(dataset)):
        i = n  # i = np.random.choice(len(dataset))

        image, mask, infor = dataset[i]
        overlay = np.vstack([m for m in mask])

        # ----
        print('%05d : %s' % (i, infor.image_id))
        image_show('image', image, 0.5)
        image_show_norm('mask', overlay, 0, 1, 0.5)
        cv2.waitKey(0)


def run_check_test_dataset():
    dataset = SteelDataset(
        mode='test',
        csv=['sample_submission.csv', ],
        split=['test_1801.npy', ],
        augment=None,  #
    )
    print(dataset)
    # exit(0)

    for n in range(0, len(dataset)):
        i = n  # i = np.random.choice(len(dataset))

        image, mask, infor = dataset[i]
        overlay = np.vstack([m for m in mask])

        # ----
        print('%05d : %s' % (i, infor.image_id))
        image_show('image', image, 0.5)
        image_show_norm('mask', overlay, 0, 1, 0.5)
        cv2.waitKey(0)


def run_check_data_loader():
    dataset = SteelDataset(
        mode='train',
        csv=['train.csv', ],
        split=['train0_12068.npy', ],
        augment=None,  #
    )
    print(dataset)
    loader = DataLoader(
        dataset,
        # sampler     = BalanceClassSampler(dataset),
        # sampler     = SequentialSampler(dataset),
        sampler=RandomSampler(dataset),
        batch_size=32,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate
    )

    for t, (input, truth, infor) in enumerate(loader):

        print('----t=%d---' % t)
        print('')
        print(infor)
        print('input', input.shape)
        print('truth', truth.shape)
        print('')

        if 1:
            batch_size = len(infor)
            input = input.data.cpu().numpy()
            input = (input * 255).astype(np.uint8)
            input = input.transpose(0, 2, 3, 1)
            # input = 255-(input*255).astype(np.uint8)

            truth = truth.data.cpu().numpy()
            for b in range(batch_size):
                print(infor[b].image_id)

                image = input[b]
                mask = truth[b]
                overlay = np.vstack([m for m in mask])

                image_show('image', image, 0.5)
                image_show_norm('mask', overlay, 0, 1, 0.5)
                cv2.waitKey(0)


def run_check_augment():
    def augment(image, mask, infor):
        # image, mask = do_random_scale_rotate(image, mask)
        # image = do_random_log_contast(image)

        # if np.random.rand()<0.5:
        #     image, mask = do_flip_ud(image, mask)

        # image, mask = do_noise(image, mask, noise=8)
        # image, mask = do_random_crop_rescale(image,mask,1600-(256-224),224)
        image, mask = do_random_crop_rotate_rescale(image, mask, 1600 - (256 - 224), 224)

        # image, mask = do_random_scale_rotate(image, mask, 224*2, 224)
        return image, mask, infor

    dataset = SteelDataset(
        mode='train',
        csv=['train.csv', ],
        split=['train0_12068.npy', ],
        augment=None,  # None
    )
    print(dataset)

    for t in range(len(dataset)):
        image, mask, infor = dataset[t]

        overlay = image.copy()
        overlay = draw_contour_overlay(overlay, mask[0], (0, 0, 255), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[1], (0, 255, 0), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[2], (255, 0, 0), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[3], (0, 255, 255), thickness=2)

        print('----t=%d---' % t)
        print('')
        print('infor\n', infor)
        print(image.shape)
        print(mask.shape)
        print('')

        # image_show('original_mask',mask,  resize=0.25)
        image_show('original_image', image, resize=0.5)
        image_show('original_overlay', overlay, resize=0.5)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, mask1, infor1 = augment(image.copy(), mask.copy(), infor)

                overlay1 = image1.copy()
                overlay1 = draw_contour_overlay(overlay1, mask1[0], (0, 0, 255), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[1], (0, 255, 0), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[2], (255, 0, 0), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[3], (0, 255, 255), thickness=2)

                # image_show_norm('mask',mask1,  resize=0.25)
                image_show('image1', image1, resize=0.5)
                image_show('overlay1', overlay1, resize=0.5)
                cv2.waitKey(1)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_dataset()
    # run_check_test_dataset()

    # run_check_data_loader()
    #r un_check_augment()
