The dataset available at (https://openreview.net/forum?id=mV4EKzUVI96) was utilized, specifically focusing on the dog category. This dataset comprises video data of dogs, captured in frames ranging from 1 to 15. Each frame is annotated with keypoints and bounding boxes.

Overall pipleline
![그림2](https://github.com/seohyun8825/DogCFLD/assets/153355118/0175b521-945d-4fd6-bb8b-99592a6745ec)

For training, the images were cropped using the bounding boxes, and then every two consecutive frames (from 1 to 15) were paired to create a paired dataset. The results were as follows:
![dog qualitative](https://github.com/seohyun8825/DogCFLD/assets/153355118/7ce8a4ab-1eaa-49ed-8c3a-051c9ba8706f)

Checkpoints for training AP-36k dataset will be updated later

*The model code is based on CFLD official code(https://github.com/YanzuoLu/CFLD)
