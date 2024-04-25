import gdown
from zipfile import ZipFile
zip_loc = "/mnt/data/shubham/OBJ3D.zip"
gdown.download(id="1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm", output=zip_loc)

with ZipFile(zip_loc, 'r') as f:
    f.extractall(zip_loc[:-3])

