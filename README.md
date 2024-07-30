# EBGM


## How to run the code

1. Create virtual environment to be used
```
python3 -m venv venv
```
2. Activate the virtual environment
```
source venv/bin/activate
```
3. Run the model locally. NOTE: The first argument is the python file containing all functions to be uses, the second argument is the training dataset, and the third argument is the testing image
```
python src/main.py "src/rough/grayscale/" "src/00002qr010_940928.tif"
```
4. Build the image/container in docker
```
docker build -t patrick:0001 .
```
5. Run the docker container
```
docker run patrick:0001
```
6. Kill container
```
# EBGM
