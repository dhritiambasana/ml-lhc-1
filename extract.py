# /home/mini/Downloads
from zipfile import ZipFile

# loading the temp.zip and creating a zip object
with ZipFile("/run/media/mini/Data/marwadi-uni/sem-5/ml/lhc-1/data/3.zip", 'r') as zObject:

    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path="/run/media/mini/Data/marwadi-uni/sem-5/ml/lhc-1/data")