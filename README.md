The dataset available at (https://openreview.net/forum?id=mV4EKzUVI96) was utilized, specifically focusing on the dog category. This dataset comprises video data of dogs, captured in frames ranging from 1 to 15. Each frame is annotated with keypoints and bounding boxes.

Overall pipleline
![그림2](https://github.com/seohyun8825/DogCFLD/assets/153355118/0175b521-945d-4fd6-bb8b-99592a6745ec)

For training, the images were cropped using the bounding boxes, and then every two consecutive frames (from 1 to 15) were paired to create a paired dataset. The results were as follows:
![dog qualitative](https://github.com/seohyun8825/DogCFLD/assets/153355118/7ce8a4ab-1eaa-49ed-8c3a-051c9ba8706f)

(Checkpoints for training AP-36k dataset will be updated later)


The model demonstrated good results across various dog breeds and is a generalized model. To better reflect the characteristics of a specific breed, we performed transfer learning using four photos of my dog, Happy, on top of a previously trained checkpoint. The model was overfitted for about 30 minutes to see if it better captured the specific characteristics.

The Happy dataset was created using DeepLabCut to extract key points from Happy's images in JSON format and used them as the dataset. Unlike the previous approach where pose images were created by connecting all key points with lines, this time we marked only the joints with dots to prevent the pose images from being too dominant. The Happy dataset, along with annotations, is located in the github_happy folder. To overfit this dataset, you may use happyoverfit.py for training and happyoverfit_test.py for test and inference.

The results showed that the fine-tuned model better captured Happy's unique features compared to the generalized model. This experiment illustrates the effectiveness of transfer learning in refining a model to better suit specific cases with limited data.

The result was as follows
![happy](https://github.com/seohyun8825/DogCFLD/assets/153355118/b9ef68d6-2024-4973-a70a-d466d46dbd68)


*The model code is based on CFLD official code(https://github.com/YanzuoLu/CFLD)
