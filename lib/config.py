import os
import sys
from easydict import EasyDict

# path for CityRefer dataset
'''
CONF = EasyDict()
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/content/drive/MyDrive/LISA_Grounding"  # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.SCAN = os.path.join(CONF.PATH.DATA, "sensaturban")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")
'''

CONF = EasyDict()
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/content/drive/MyDrive/CityAnchor_Release" # change this
CONF.PATH.DATA = "/content/drive/MyDrive/CityAnchor_Release_Data/data_cityrefer" # change this
CONF.PATH.SCAN = os.path.join(CONF.PATH.DATA, "sensaturban")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCAN_SCANS = os.path.join(CONF.PATH.SCAN, "scans")
CONF.PATH.SCAN_META = os.path.join(CONF.PATH.SCAN, "meta_data")
CONF.PATH.SCAN_DATA = os.path.join(CONF.PATH.SCAN, "pointgroup_data/balance_split/random-50_crop-250")  # change this

# landmark data
CONF.PATH.LANDMARK_DATA = os.path.join(CONF.PATH.SCAN, "landmark_feat_llama_128") # change this

# path for CityRefer dataset
CONF.PATH.BASE_CITYANCHOR = "/content/drive/MyDrive/CityAnchor_Release_Data/data_cityanchor"
CONF.PATH.FEAT_CITYANCHOR = os.path.join(CONF.PATH.BASE_CITYANCHOR, "feat")
CONF.PATH.BOX_CITYANCHOR = os.path.join(CONF.PATH.BASE_CITYANCHOR, "bbox")
CONF.PATH.MAP_CITYANCHOR = os.path.join(CONF.PATH.BASE_CITYANCHOR, "map")
