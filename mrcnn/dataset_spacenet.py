"""
We pre-train the Mask RCNN model with spacenet building dataset.
This module provides a set of functions that generate the input dataset from spacenet data: RGBPan + annotations
"""
from osgeo import gdal
import geoio
import subprocess
import numpy as np
from mrcnn import utils
import os
import random
import json
from mrcnn import visualize
from mrcnn.model import log
from PIL import Image, ImageDraw

def convert_gtif_to_8bit(src_raster_path, dst_raster_path = None,percentiles=[0, 100],saving = False):

    srcRaster = gdal.Open(src_raster_path)
    outputPixType = 'Byte'
    outputFormat = 'GTiff'

    cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat, '-co', '"PHOTOMETRIC=rgb"']

    if saving:
        for bandId in range(srcRaster.RasterCount):
            bandId = bandId + 1
            band = srcRaster.GetRasterBand(bandId)

            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(),
                             percentiles[0])
            bmax = np.percentile(band_arr_tmp.flatten(),
                             percentiles[1])
            cmd.append('-scale_{}'.format(bandId))
            cmd.append('{}'.format(bmin))
            cmd.append('{}'.format(bmax))
            cmd.append('{}'.format(0))
            cmd.append('{}'.format(255))

        cmd.append(src_raster_path)
        cmd.append(dst_raster_path)
        print(cmd)
        subprocess.call(cmd)
    #later, need a change to gdal without saving
    else:
        arr = np.zeros((650,650,srcRaster.RasterCount))
        for bandId in range(srcRaster.RasterCount):
            band = srcRaster.GetRasterBand(bandId + 1)

            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(),
                                 percentiles[0])
            bmax = np.percentile(band_arr_tmp.flatten(),
                                 percentiles[1])
            b_im = band.ReadAsArray()
            if bmax==0:
                continue
            else:
                arr[:,:,bandId] = 255*(b_im - bmin) / (bmax - bmin)
        return arr.astype(np.uint8)

def fill_between(polygon):
    """
    Returns: a bool array
    """
    img = Image.new('1', (650, 650), False)
    ImageDraw.Draw(img).polygon(polygon, outline=True, fill=True)
    mask = np.array(img)
    return mask
class SpaceNetDataset(utils.Dataset):
    """Generates the spacenet dataset, including images and according annotations
    the 16bits pixels values are converted to 8bita on the fly.
    """

    def load_dataset(self, dataset_dir, seed, val_size, subset):
        assert subset in ["train", "val"]
        # Add classes
        self.add_class("SpaceNetDataset", 1, "building")

        # define data locations for images and annotations
        images_dir = os.path.join(dataset_dir, "train")
        annotations_dir = os.path.join(dataset_dir, "train_annotations")

        file_list = []
        for filename in os.listdir(images_dir):
            file_list.append(filename)
        random.Random(seed).shuffle(file_list)
        num_img = len(file_list)
        if subset == "train":
            Id_set = file_list[0:-int(num_img * val_size)]
        else:
            Id_set = file_list[-int(num_img * val_size)::]

        # Iterate through all files in the subet folder
        # add class, images and annotaions

        for filename in Id_set:
            image_id = filename[:-4].split("_")[-3]+"_"+filename[:-4].split("_")[-2]+"_"+filename[:-4].split("_")[-1]
            im_path = os.path.join(images_dir,filename)
            ann_path = os.path.join(annotations_dir, "buildings_AOI_" + image_id + ".geojson")
            self.add_image('SpaceNetDataset', image_id=image_id, path=im_path, annotation=ann_path,prefix = image_id)
#
    def load_image(self, image_id):
        info = self.image_info[image_id]
        return convert_gtif_to_8bit(info["path"])
#
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "SpaceNetDataset":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks from geojson annotation files
        """
        masks = np.zeros((650, 650))
        info = self.image_info[image_id]
        RGBTIFResmi = geoio.GeoImage(info["path"])
        with open(info["annotation"]) as f:
            data = json.load(f)
            allBuildings = data['features']

            for building in allBuildings:
                veri = building['geometry']['coordinates'][0]

                tip = str(building['geometry']['type'])
                coordinates = list()
                if tip == ('Point'):
                    continue

                elif tip == ('MultiPolygon'):

                    if isinstance(veri, float): continue

                    kucukBinalar = (building['geometry']['coordinates'])
                    for b in range(len(kucukBinalar)):
                        veri = kucukBinalar[b][0]
                        for i in veri:
                            xPixel, yPixel = RGBTIFResmi.proj_to_raster(i[0], i[1])
                            xPixel = 649 if xPixel > 649 else xPixel
                            yPixel = 649 if yPixel > 649 else yPixel
                            coordinates.append((xPixel, yPixel))
                else:
                    if isinstance(veri, float): continue

                    for i in veri:
                        xPixel, yPixel = RGBTIFResmi.proj_to_raster(i[0], i[1])
                        xPixel = 649 if xPixel > 649 else xPixel
                        yPixel = 649 if yPixel > 649 else yPixel
                        coordinates.append((xPixel, yPixel))

                maske = fill_between(coordinates)
                masks = np.dstack((masks, maske))

        if masks.shape != (650, 650):
            masks = masks[:, :, 1:]
            class_ids = np.asarray([1] * masks.shape[2])
        else:
            class_ids = np.ones((1))
            masks = masks.reshape((650, 650, 1))
        return masks.astype(np.bool), class_ids.astype(np.int32)



if __name__ == '__main__':

    dataset_dir = os.path.abspath("../../spacenetdata/")
    # Training dataset
    dataset_train = SpaceNetDataset()
    dataset_train.load_dataset(dataset_dir=dataset_dir, seed=42, val_size=0.2, subset="train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SpaceNetDataset()
    dataset_val.load_dataset(dataset_dir=dataset_dir, seed=42, val_size=0.2, subset="val")
    dataset_val.prepare()

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




