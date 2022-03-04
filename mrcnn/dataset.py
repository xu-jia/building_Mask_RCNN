from mrcnn import utils
import json
import os
import numpy as np
import random
import rasterio
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from shapely.geometry import Polygon,mapping
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from rasterio.windows import Window
from pycocotools.coco import COCO
import skimage.io
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from mrcnn.model import log
from mrcnn import visualize

class Building_Dataset(utils.Dataset):
    """
    generate custom dataset
    """
    def __init__(self, data_dir, loading_option = False, data_paths = None, seed=None, val_size=None, height=None, width=None):
        self.data_dir = data_dir
        if loading_option:
            assert None not in [data_paths, seed, val_size, height, width],"parameters for instances loading must be provided."
            # call the instance_loading function to generate and saving images and coco-like annotations to the data_dir
            # if only change the annotation file, set saving = False
            instance_loading(data_dir,data_paths,height,width,seed,val_size,saving = False)
        super(Building_Dataset, self).__init__()

    def load_dataset(self, class_ids=None, subset=None):
        assert subset in ["train", "val"]
        # The images and coco-like annotations exist already in the data_dir
        image_dir = os.path.join(self.data_dir, subset)
        coco = COCO(os.path.join(self.data_dir, "annotations", "instances_" + subset + "_bati.json"))
        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())
        # All images or a subset?
        # if class_ids:
        #     image_ids = []
        #     for id in class_ids:
        #         image_ids.extend(list(coco.getImgIds(catIds=[id])))
        #         # Remove duplicates
        #         image_ids = list(set(image_ids))
        # else:
            # All images
        image_ids = list(coco.imgs.keys())
        # Add classes
        for i in class_ids:
            self.add_class("coco_file", i, coco.loadCats(i)[0]["name"])
        # Add images
        for i in image_ids:
            self.add_image("coco_file", image_id=i,
                    path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["win"]["width"],
                    height=coco.imgs[i]["win"]["height"],
                    col_off = coco.imgs[i]["win"]["col_off"],
                    row_off = coco.imgs[i]["win"]["row_off"],
                    n_ob = coco.imgs[i]["n_ob"],
                    annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id']
            if class_id is not None:
                m = self.annToMask(annotation, image_info["height"],image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)
                # print("app")

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(Building_Dataset, self).load_mask(image_id)
        # The following two functions are from pycocotools with a few changes.

    def image_reference(self, image_id):
        """Return a link to the image."""
        info = self.image_info[image_id]
        if info["source"] == "coco_file":
            return image_id
        else:
            super(Building_Dataset, self).image_reference(image_id)
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def instance_loading(data_dir,data_paths,height,width,seed,val_size,saving = False):
    """
    @param data_dir: directory of data
    @param data_paths: a list of paths that contains the pairs of rasters and their according vectors
    @param height: patch images height
    @param width: patch images width
    @param seed: random seed to generate the dateset
    @param val_size: the validation size, set a value of (0,1)
    @param saving: option of saving patch images
    @return:
    """
    train_image_dir = os.path.join(data_dir, "train")
    val_image_dir = os.path.join(data_dir, "val")

    if saving:
        if not os.path.exists(train_image_dir):
            os.makedirs(train_image_dir)
        else:
            for f in os.listdir(train_image_dir):
                os.remove(os.path.join(train_image_dir, f))
        if not os.path.exists(val_image_dir):
            os.makedirs(val_image_dir)
        else:
            for f in os.listdir(val_image_dir):
                os.remove(os.path.join(val_image_dir, f))
    # avoid to use the category id 0 since it is for BG
    coco_train = dict()
    coco_train['images'] = []
    coco_train['type'] = 'instances'
    coco_train['annotations'] = []
    coco_train['categories'] = []
    # change to automatic later
    coco_train['categories'].append({"supercategory": "bati", "id": 1, "name": "gable"})
    coco_train['categories'].append({"supercategory": "bati", "id": 2, "name": "hip"})
    coco_train['categories'].append({"supercategory": "bati", "id": 3, "name": "flat"})
    coco_train['categories'].append({"supercategory": "bati", "id": 4, "name": "half-hip"})
    coco_train['categories'].append({"supercategory": "bati", "id": 5, "name": "pyramid"})
    coco_train['categories'].append({"supercategory": "bati", "id": 6, "name": "L-hip"})
    coco_train['categories'].append({"supercategory": "bati", "id": 7, "name": "L-gable"})
    coco_train['categories'].append({"supercategory": "bati", "id": 8, "name": "mansard"})
    coco_train['categories'].append({"supercategory": "bati", "id": 9, "name": "unknown"})

    coco_val = dict()
    coco_val['images'] = []
    coco_val['type'] = 'instances'
    coco_val['annotations'] = []
    coco_val['categories'] = []
    # change to automatic later
    coco_val['categories'].append({"supercategory": "bati", "id": 1, "name": "gable"})
    coco_val['categories'].append({"supercategory": "bati", "id": 2, "name": "hip"})
    coco_val['categories'].append({"supercategory": "bati", "id": 3, "name": "flat"})
    coco_val['categories'].append({"supercategory": "bati", "id": 4, "name": "half-hip"})
    coco_val['categories'].append({"supercategory": "bati", "id": 5, "name": "pyramid"})
    coco_val['categories'].append({"supercategory": "bati", "id": 6, "name": "L-hip"})
    coco_val['categories'].append({"supercategory": "bati", "id": 7, "name": "L-gable"})
    coco_val['categories'].append({"supercategory": "bati", "id": 8, "name": "mansard"})
    coco_val['categories'].append({"supercategory": "bati", "id": 9, "name": "unknown"})


    # dic_form = {1.0:1,2.0:2,3.0:3,4.0:4,5.0:5,6.0:6,7.0:7}
    annot_id = 1 # count of unique annotation id

    for rst_path,vec_path,in data_paths:
        prefix = rst_path[:-4]
        vec_path = os.path.join(data_dir, vec_path)
        rst_path = os.path.join(data_dir, rst_path)
        # process the geopandas dataframe
        gdf = gpd.read_file(vec_path)
        gdf.dropna(axis=0, subset=["geometry"], inplace=True)
        gdf = gdf[["forme", "geometry"]]
        gdf = gdf[gdf.geometry != None]
        gdf = gdf.fillna(0)
        invalid_geom = ~gdf['geometry'].is_valid
        gdf = gdf.loc[gdf['geometry'].is_valid, :]
        def to_categ(x):
            if x in range(1,9):
                return int(x)
            else:
                return 9
        gdf["forme"] = gdf["forme"].apply(to_categ)
        # open the raster
        with rasterio.open(rst_path) as src:
            n_h = src.height // height  # number of horizontal grid
            n_v = src.width // width  # number of vertical grid
            num_img = n_h * n_v
            # randomly generate the ImageIds
            ImageIds = np.arange(num_img)
            random.Random(seed).shuffle(ImageIds)
            train_set = ImageIds[0:-int(num_img * val_size)]
            val_set = ImageIds[-int(num_img * val_size)::]
            # save to training
            for grid_id in train_set:
                n, m = divmod(grid_id, n_v)
                win = Window.from_slices((n * height, (n + 1) * height), (m * width, (m + 1) * width))
                file_name = prefix+"_im_" + str(grid_id) + ".png"
                # when option saving is true, read the windowed image and save it.
                if saving:
                    if os.path.exists(os.path.join(train_image_dir,file_name)):
                        continue
                    else:
                        arr = src.read([1,2,3], window=win)
                        arr = np.moveaxis(arr, [0, 1, 2], [-1, -3, -2])
                        skimage.io.imsave(os.path.join(train_image_dir,file_name), arr)
                segs = geo2polyannot(gdf, src=src, win=win)
                res = dict()
                res["file_name"] = file_name
                res["n_ob"] = segs.shape[0]
                res["win"] = {"col_off": int(win.col_off), "row_off": int(win.row_off), "width": int(win.width), "height": int(win.height)}
                res["id"] = prefix+str(grid_id)
                coco_train['images'].append(res)
                print("image:", grid_id, "number of objects:",segs.shape[0])
                for _,seg in segs.iterrows():
                    res = dict()
                    res["segmentation"] = np.array(seg.geometry.exterior.coords).reshape(1,-1).tolist()
                    res["area"] = seg.geometry.area
                    res["iscrowd"] = 0
                    res["image_id"] = prefix+str(grid_id)
                    x, y, max_x, max_y = seg.geometry.bounds
                    w = max_x - x
                    h = max_y - y
                    bbox = (x, y, w, h)
                    res["bbox"] = bbox
                    res["category_id"] = seg.forme
                    res["id"] = annot_id
                    annot_id += 1
                    coco_train["annotations"].append(res)

            # save to validation
            for grid_id in val_set:
                n, m = divmod(grid_id, n_v)
                win = Window.from_slices((n * height, (n + 1) * height), (m * width, (m + 1) * width))
                file_name = prefix+"_im_" + str(grid_id) + ".png"
                # when option saving is true, read the windowed image and save it.
                if saving:
                    arr = src.read([1,2,3], window=win)
                    arr = np.moveaxis(arr, [0, 1, 2], [-1, -3, -2])
                    skimage.io.imsave(os.path.join(val_image_dir,file_name), arr)
                segs = geo2polyannot(gdf, src=src, win=win)
                res = dict()
                res["file_name"] = file_name
                res["n_ob"] = segs.shape[0]
                res["win"] = {"col_off": int(win.col_off), "row_off": int(win.row_off), "width": int(win.width), "height": int(win.height)}
                res["id"] = prefix+str(grid_id)
                coco_val['images'].append(res)
                print("image:", grid_id, "number of objects:",segs.shape[0])
                for _,seg in segs.iterrows():
                    res = dict()
                    res["segmentation"] = np.array(seg.geometry.exterior.coords).reshape(1,-1).tolist()
                    res["area"] = seg.geometry.area
                    res["iscrowd"] = 0
                    res["image_id"] = prefix+str(grid_id)
                    x, y, max_x, max_y = seg.geometry.bounds
                    w = max_x - x
                    h = max_y - y
                    bbox = (x, y, w, h)
                    res["bbox"] = bbox
                    res["category_id"] = seg.forme
                    res["id"] = annot_id
                    annot_id += 1
                    coco_val["annotations"].append(res)
    json_file = os.path.join(data_dir,"annotations","instances_train.json")
    json.dump(coco_train, open(json_file, 'w'))
    json_file = os.path.join(data_dir, "annotations", "instances_val.json")
    json.dump(coco_val, open(json_file, 'w'))

"""
convert spatial georeferenced geometries to polygons with respect to images pixels or vice versa
geo2polyannot:
    - used in the instance_loading function to generate patch images for training process of Machine learning
<->
polyannot2geo:
    - vectorization of masks predicted with Machine learning model
"""
def geo2polyannot(gdf,src=None,win=None):
    """
    @param gdf: shapefiles that contain the coordinates of each object
    @param src: original raster data
    @param win: window of patch image clipping
    @return:
    """
    if not src:
        raise ValueError("A source raster must be provided")
    if win is not None:
        bounds = src.window_bounds(win)
        width = win.width
        height = win.height
    else:
        bounds = src.bounds
        width = src.width
        height = src.height
    # calculate the affine
    xl,yb,xr,yt = bounds
    affine_obj = [width / (xr - xl), 0.0, 0.0, height / (yb - yt), -width * xl / (xr - xl),
                  height + height * yb / (yt - yb)]
    # gdf = gdf.cx[xl:xr, yb:yt]
    # gdf = gdf[gdf.geometry.within(Polygon(box(*bounds))) == True]
    gdf = gdf[gdf["geometry"].intersection(Polygon(box(*bounds))).is_empty == False]
    return gpd.GeoDataFrame(gdf, geometry=gdf["geometry"].affine_transform(affine_obj))

def polyannot2geo(masks,class_ids,src=None,win=None):
    """
    @param masks: masks of objects
    @param class_ids: the class_ids of each object
    @param src: original raster data
    @param win: window of patch image clipping
    @return:
    """
    if not src:
        raise ValueError("The source raster information is missing.")
    if win is not None:
        bounds = src.window_bounds(win)
        width = win.width
        height = win.height
    else:
        bounds = src.bounds
        width = src.width
        height = src.height
    # calculate the affine
    xl, yb, xr, yt = bounds
    affine_obj = [(xr - xl)/width, 0.0, 0.0,(yb - yt)/height,xl, yt]
    # create a dataframe with class ids
    df = pd.DataFrame({"class_id": class_ids})
    # case 1: masks predicted from object detection model: matrix of boolean width x height x n_objects
    if isinstance(masks, np.ndarray):
        N = masks.shape[2] # number of objects
        polys = []
        for i in range(N):
            mask = masks[:, :, i]
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
            polys.append(Polygon(verts))
        gdf = gpd.GeoDataFrame(df, geometry=polys)
    # case 2: Boundary regularized masks with a post-processing step (already converted to Polygons)
    elif isinstance(masks, list):
        N = len(masks)
        df = pd.DataFrame(masks)
        gdf = gpd.GeoDataFrame(df, geometry=masks)
    else:
        print("Either boolean masks matrix or regularized polygons must be provided.")
    return gpd.GeoDataFrame(gdf,  crs = src.crs, geometry=gdf["geometry"].affine_transform(affine_obj))

def test_data():
    # Training dataset
    data_paths = [["rst1.tif", "vec1.shp"], ["rst2.tif", "vec2.shp"], ["rst3.tif", "vec3.shp"]]
    dataset_train = Building_Dataset(data_dir="../data", loading_option=True, data_paths=data_paths, seed=42,
                                     val_size=0.2, height=512, width=512)
    dataset_train.load_dataset(subset="train")
    dataset_train.prepare()
    # Load random image and mask.
    image_id = random.choice(dataset_train.image_ids)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)
    # Display image and additional stats
    print("image_id ", image_id, dataset_train.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)
def create_data():
    data_paths = [["0835-6520.tif", "vec.shp"]]
    dataset_train = Building_Dataset(data_dir="../data", loading_option=True, data_paths=data_paths, seed=42,
                                     val_size=0.1, height=512, width=512)

if __name__ == '__main__':
    create_data()
    # test_data()
    # dataset_train = Building_Dataset(data_dir="../data")
    # dataset_train.load_dataset(subset="train")
    # dataset_train.prepare()
    # image_id = random.choice(dataset_train.image_ids)
    # image = dataset_train.load_image(image_id)
    # mask, class_ids = dataset_train.load_mask(image_id)
    # # Compute Bounding box
    # bbox = utils.extract_bboxes(mask)
    # # Display image and additional stats
    # print("image_id ", image_id, dataset_train.image_reference(image_id))
    # log("image", image)
    # log("mask", mask)
    # log("class_ids", class_ids)
    # log("bbox", bbox)
    # # Display image and instances
    # visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)
    # info = dataset_train.image_info[image_id]
    # win = Window(info["col_off"], info["row_off"], info["width"], info["height"])
    # with rasterio.open(os.path.join("../data/",os.path.split(info["path"])[-1][:4]+".tif")) as src:
    #     new_gdf = polyannot2geo(mask, class_ids, src=src, win=win)
    # new_gdf.to_file("../data/masks.shp")







