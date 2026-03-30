# To run the container

Under the root directory run:
```
docker run --gpus '"device=0,1,2,3,4,5,6,7,8,9"'     --shm-size=8g     -v $(pwd):/app/UAV-Language-Guided-Navigation     -v /export/openhome/vaziri/datasets:/app/datasets  -v /export/openhome/vaziri/datasets/AVDN/train_images:/app/UAV-Language-Guided-Navigation/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images   --rm -it     armanshadow/ualgn:latest

```

# To reformat the lagacy dataset to the turned based version
Under 'AnsweringAgent/src/data' run:
```
python format_avdn_dataset.py
```

# To generate paraphrases and add it to the dataset
Under 'AnsweringAgent/src/data' run:
```
python comprehensive_avdn_pipeline.py
```

# To run the training pipeline
Under the root directory run:
```
torchrun --nproc_per_node=10 AnsweringAgent/src/train.py --batch-size=8 --grad-steps=2
```