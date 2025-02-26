## How to use

### Create Environment and Install Packages

```shell
conda create -n face-dev python=3.9
```

```shell
conda activate face-dev
```

```shell
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Add new persons to datasets

1. **Create a folder with the folder name being the name of the person**

   ```
   datasets/
   ├── backup
   ├── data
   ├── face_features
   └── new_persons
       ├── name-person1
       └── name-person2
   ```

2. **Add the person's photo in the folder**

   ```
   datasets/
   ├── backup
   ├── data
   ├── face_features
   └── new_persons
       ├── name-person1
       │   └── image1.jpg
       │   └── image2.jpg
       └── name-person2
           └── image1.jpg
           └── image2.jpg
   ```

3. **Run to add new persons**

   ```shell
   python augmentation.py
   ```

4. **Run to recognize**

   ```shell
   python recognize.py
   ```