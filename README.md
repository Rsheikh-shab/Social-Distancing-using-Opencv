# Social-Distancing-using-Opencv
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

### 1.Clone the repo

```
git clone https://github.com/Rsheikh-shab/Social-Distancing-using-Opencv.git
```

### 2. Install dependencies (reccommended in virtual environment)

``` 
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download YOLOV3 pretrained weights

``` 
mkdir yolo-model
cd yolo-model
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

### 4. Run the code

``` 
python yolov3_distance.py
```

### Sample result

![output](output.gif)

### Citation

``` 
@article{yolov3, 
  title={YOLOv3: An Incremental Improvement}, 
  author={Redmon, Joseph and Farhadi, Ali}, 
  journal = {arXiv}, 
  year={2018}
}
```