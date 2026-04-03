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

# To fix short answer problem
Under 'AnsweringAgent/src/data' run:
```
python fix_short_answers.py
``` 
# To fix structure problem (each turn must have exactly 2 positive and 1 negative paraphrase)
Under 'AnsweringAgent/src/data' run:
```
python fix_structure_issues.py
``` 

# To mine hard and diverse negatives for the train dataset 
Under 'AnsweringAgent/src/data' run:
```
python add_hard_negatives.py   --split train   --gpu-id 0   --k-nn 1000   --min-answer-length 30   --cosine-threshold 0.18   --min-visual-similarity 0.25   --diverse-ratio 0.5   --fallback-phrase-reuse-limit 1   --sliding-window-size 5000   --batch-size 64
``` 
# To mine hard and diverse negatives for the val_seen dataset 
Under 'AnsweringAgent/src/data' run:
```
python add_hard_negatives.py   --split val_seen   --gpu-id 0   --k-nn 80   --min-answer-length 30   --cosine-threshold 0.18   --min-visual-similarity 0.25   --diverse-ratio 0.5   --fallback-phrase-reuse-limit 1   --sliding-window-size 5000   --batch-size 64
``` 

# To mine hard and diverse negatives for the val_unseen dataset 
Under 'AnsweringAgent/src/data' run:
```
python add_hard_negatives.py   --split val_unseen   --gpu-id 0   --k-nn 60   --min-answer-length 30   --cosine-threshold 0.18   --min-visual-similarity 0.25   --diverse-ratio 0.5   --fallback-phrase-reuse-limit 1   --sliding-window-size 5000   --batch-size 64
``` 

# To run the training pipeline
Under the root directory run:
```
torchrun --nproc_per_node=10 AnsweringAgent/src/train.py --batch-size=8 --grad-steps=2
```