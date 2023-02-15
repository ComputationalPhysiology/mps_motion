(cli)=
# Command line interface

The package comes with a command line interface that run in two different ways. One way is to use the script, e.g
```
mps-motion analyze data.nd2
```
The other way is to also specify the python interpreter that you are using
```
python -m mps_motion analyze data.nd2
```
The latter might be good if you have multiple versions of `mps-motion` installed on different versions of python. However, for most cases, the first approach would be sufficient

The command line interface currently comes with two command; `analyze` and `gui`. The `analyze` command is used for performing an analysis of a file or folder by running the motion analysis and storing the results in a folder. The `gui` command is for starting an interactive gui, see See {ref}`gui` for more info.

## Analyzing a file or folder
To analyze a single file you can run the command
```
mps-motion analyze data.nd2
```

If you have multiple files in a folder, then you could also pass in the folder as an argument. Say that you have a folder called `data` with the following content
```
data
├── data1.nn2
├── data2.nd2
```
Then you could run the command
```
mps-motion analyze data
```
and the script will analyze the different datasets one by one

## Custom output directory
The several ways in which you can customize the way the script works. To see all the options, execute the command
```
mps-motion analyze --help
```
For most cases the default settings should be OK, but if you for example want to change the output directory you could e.g do
```
mps-motion analyze data.nd2 --outdir=my-outdir
```

## Selected region of time span
Sometimes you want to focus on a particular region in the video. For example, say that you are tracking a tissue, and you want to only focus on a part of the tissue. This is also useful if you have objects surrounding the tissue that you don't want to focus on. You can specify the start and end indices using the `--start-x`, `--end-x`, `--start-y`, and `--end-y` flags. For example the command
```
mps-motion analyze data.nd2 --start-x=200 --end-x=800 --start-y=100 --end-y=300
```
will only analyze the motion in the region [200, 800] x [100, 300]. It is sometimes difficult to know in advance what are the dimensions of the images, and you can use the `mps-motion info` command for this, i.e
```
mps-motion info data.nd2
```
You might also be in a situation where you want to discard some of the time stamps and choose a different start and end time. For this you can use the `--start-t` and `--end-t` flags, for example
```
mps-motion analyze data.nd2 --start-t=200 --end-t=2200
```
This will only analyze the signal from time 200ms to 2200 ms. Note however that the values for the time is in the same unit as the time stamp in the data (i.e the values are not indices but actual time values in milliseconds).

## Creating movies
It is also possible to create movies with the displacement or velocity vectors on top.
To create a movie of the displacement you can use the command
```
mps-motion analyze data/data.nd2 --video-disp
```
This will first run the normal analysis script, and then at the end create a movie of the displacement vectors.

The number of vectors are scalar of the vectors can also be adjusted, using the options `--video-disp-step` and ` --video-disp-scale`. Lower values of ` --video-disp-step` will make the vectors more dense. The default value if 24 which means that the space between two vectors are 24 pixels. Higher values of ` --video-disp-scale` will make the vectors longer. Here, the value represent the actual pixel movement. However for small deformation it might be beneficial to increase the scale. For example, we could do
```
mps-motion analyze data/data.npy --video-disp --video-disp-step=48 --video-disp-scale=12
```
to make the vectors longer and more sparse.

A similar command exists for the velocity, where you swap out the name `disp` for `vel` (for velocity). For example, to create a movie of the velocity you could execute the command
```
mps-motion analyze data/data.npy --video-vel --video-vel-step=36 --video-vel-scale=2
```
Note that the size of the velocity vectors are typically greater than the displacement, so you might want to choose a smaller value of the scale.
