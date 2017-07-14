# osvr-review
This repository contains the code used to execute a review study of Zhao et al's ["Facial Expression Intensity Estimation Using Ordinal Information"](http://ieeexplore.ieee.org/document/7780746) (doi:10.1109/CVPR.2016.377), which introduces a new method of estimating facial emotional intensity estimation called Ordinal Support Vector Regression (OSVR). We assume that you've already read the paper and downloaded and tried out the [MATLAB code](https://github.com/rort1989/OSVR) provided with it. We also assume that you'll be running experiments on the [CK+ image dataset](http://www.consortium.ri.cmu.edu/ckagree/) (or another dataset that follow's CK+'s filetree structure) for the purposes of this README, although the Usage section has some tips on modifying the code to accept other datasets. 

## Installation
Starting with a clean, fully-updated copy of Ubuntu 16.04 Server

```bash
# Install the Miniconda2 Python distribution. You may also use the full Anaconda2
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash ./Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/miniconda2
echo "PATH=\$HOME/miniconda2/bin:\$PATH" >> ~/.profile
source ~/.profile

# See note below for explanation of conda environments
conda create --name falign python=2 numpy==1.8.2 six==1.10.0 pillow==3.4.2 scikit-image==0.10.1 scipy==0.14.0 opencv==2.4.9 ipython
conda create --name labeler python=2 scipy==0.14.0 matplotlib==1.3.1 pillow ipython

# Download the code
git clone https://github.com/abyrne55/osvr-review.git
```

### Why did we have to create two separate Python environments?
Unfortunately, there's an unresolvable version conflict between the dependencies of `face-align.py` and those of `labeler.py`. Specifically, `labeler.py` requires matplotlib 1.3.1, which requires freetype 2.4.10, while several `face-align.py` dependencies require at least freetype 2.5.2. Because of these conflicts, you must run the two scripts in separate conda environments. 

## Usage
**Note:** The code is currently set up to work with the directory structure provided by the CK+ dataset. You'll have to modify `load_frame_path()`, `load_frame()`, and `load_eye_landmarks()` in `face-align.py` accordingly if you're using a different dataset. You'll also have to modify the pixel values in `labeler.py`'s `LabeledFrame` class if your input images are different sizes than those of CK+. 

### Preparing the input to the MATLAB script
First, edit the `CKPLUSPATH` variable in `face-align.py` to match the path to your copy of the CK+ dataset (be sure *not* to include a trailing slash). For example, if you set `CKPLUSPATH = "/home/ubuntu/ckplus"`, then that filetree should look like:
```
/home/ubuntu/ckplus
├── Emotion
│   ├── S005
│   ├── S010
│   ├── S011
│   ...
│   ├── S895
│   └── S999
│
├── Landmarks
│   ├── S005
│   ├── S010
│   ├── S011
│   ...
│   ├── S895
│   └── S999
│
└── cohn-kanade-images
    ├── S005
    ├── S010
    ├── S011
    ...
    ├── S895
    └── S999
```

Now enter the `falign` environment you set up during installation and load the script into an ipython console
```bash
cd osvr-review/frame-normaliser
source activate falign
ipython -i face-align.py
```

You'll usually start by generating a list of "usable frames", or frames that depict a certain emotion. As an example, let's generate a training dataset of all but the last 20 neutral (0), angry (1), and disgusted (3) frames (you can find the integer codes for other emotions in the CK+ readme), and a testing dataset of the last 20 frames of each of those emotions. Note that these operations are IO-bound and may take a while.
```python
uf_train = get_usable_frames(0)[:-20] + get_usable_frames(1)[:-20] + get_usable_frames(3)[:-20]
uf_test = get_usable_frames(0)[-20:] + get_usable_frames(1)[-20:] + get_usable_frames(3)[-20:]
```
*Tip: use the `cpaste` command in ipython to paste several lines of code at once*

`uf_train` and `uf_test` both contain lists of "frame descriptors", which are just python lists containing the string `subject_id`, integer `seq_num`, and integer `frame_id` of the frame in question. For example, the file "S005_001_00000002.png" would have a frame descriptor that looked like `['S005', 1, 2]`. A handful of random frame descriptors are included at the end of `face-align.py` for your convience and should be present in your ipython environment.
```python
repr(f512)
# Out: ['S005', 1, 2]
repr(f1011)
# Out: ['S010', 1, 1]
```

Now that you have lists of usable frame descriptors, you can generate the testing and training matrices required to use the paper's MATLAB script. You'll need to specify a frame as the "master frame" that every other frame will be aligned to. It makes little difference which frame is used as the "master," as long as you use the same frame consistently across experiments. We'll use the `f512` frame descriptor from above as our master. Note that these operations are CPU-bound and may also take a while.
```python
mx_train = get_train_matrix(uf_train, f512)
mx_test = get_test_matrix(uf_test, f512)
```

Once the matrices are generated, it is time to export them to MATLAB format so that they can be loaded into the paper's script. We'll use `scipy`'s IO package for this, which has already been imported as `sio` in this script.

```python
data = {'train_data_seq': mx_train[0], 'train_label_seq': mx_train[1], 'test_data': mx_test[0], 'test_label': mx_test[1]}
sio.savemat('input_to_MATLAB.mat', data)
```

You can now use `input_to_MATLAB.mat` with the `main.m` script provided by the paper to execute experiments with OSVR. Before exiting your ipython environment though, you'll want to save the paths to the images that you used to generate your testing dataset. This is so that you can input them into `labeler.py` after running the MATLAB script.
```python
pt_test = []
for desc, _ in uf_test:
    pt_test.append(load_frame_path(desc[0], desc[1], desc[2]))

# We can use pickle to save pt_test to a file, but any method of saving a variable will do
import pickle
pickle.dump(pt_test, open('pt_test.p', 'wb'))

exit
```

### Using the output from the MATLAB script
After executing the `main.m` script in MATLAB, save the original `test_label` and the resulting `dec_values` matrices to a `.mat` file. Find the `MATLAB_FILENAME` variable in the `labeler.py` script and edit it to match the full path of the `.mat` file you just saved. Finally, `cd` back to where you cloned this repo and enter the labeler environment
```bash
cd osvr-review/image-labeler
source activate labeler
ipython -i labeler.py
```

Load up the list of image paths that you saved from the last section
```python
import pickle
pt_test = pickle.load(open('pt_test.p', 'rb'))
```

Now would be a good time to pause and skim through the code of `labeler.py`. You'll notice that the "meat" of this code is the `LabeledFrame` class, which can be used to perform most of the image processing needed to overlay the MATLAB script's results on each frame. The other functions in `labeler.py` are mostly just tests of `LabeledFrame` that can be used as examples when modifying this code to suit other datasets. Luckily, the `test_77out()` function should work for our example without any modifications.
```python
# Make sure this directory actually exists!
output_dir = "./animations"

test_77out(pt_test, output_dir)
# Out: "Labeling /home/ubun..."
```

The resulting animated GIF will be saved in the `output_dir` specified. The GIF will probably become quite large when working with more than 100 or so testing frames, so it might be wise to convert it to a more space-efficient format like MP4 using one of the many free online converters available.

```bash
# Tip: After you exit() ipython and return to your shell, you may return to your default python environment
source deactivate
```

### Additional usage
There are a few additional functions present in the code that were not demonstrated in this README but may be useful to you. You're encouraged to read through the PyDoc comments in `face-align.py` and `labeler.py` for a better understanding of the code.
