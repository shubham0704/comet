import os
import gdown
from zipfile import ZipFile
parent_dir = "/mnt/data/shubham/"
zip_file = "OBJ3D.zip"
zip_loc = os.path.join(parent_dir, zip_file)
gdown.download(id="1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm", output=zip_loc)
# final file path -> parent_dir/OBJ3D
with ZipFile(zip_loc, 'r') as f:
    f.extractall(parent_dir)

