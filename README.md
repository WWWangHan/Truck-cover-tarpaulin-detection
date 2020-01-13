# Truck-cover-tarpulin-detection

> In order to protect the environment and ensure the safety of road traffic, we try to check whether the passing trucks cover the tarpaulin at the exit of the coal mine and the road toll station.

![img_02345.jpg](https://github.com/WWWangHan/Truck-cover-tarpaulin-detection/blob/master/readme_img/img_02345.jpg)


![img_02348.jpg](https://github.com/WWWangHan/Truck-cover-tarpaulin-detection/blob/master/readme_img/img_02348.jpg)

### training
> Running environment `ubuntu 16.04` with `pytorch 1.3.1` and `torchvision 0.4.2`.

> To start the training procedure, just run `python finetuning_models.py`, after one epoch, you will see something like this:

![one_epoch_result](https://github.com/WWWangHan/Truck-cover-tarpaulin-detection/blob/master/readme_img/one_epoch_result.png)

### evaluate
> After training, the best model weights will be storaged at `tarp_detect/model/`, then you can test this program with the following commands:

```python
python server.py
python realtime_output_demo.py
```

> Then you will get something like this:

![test_order_result](https://github.com/WWWangHan/Truck-cover-tarpaulin-detection/blob/master/readme_img/test_order_result.png)

> Open another terminal and run some orders like this, you can see:

![copy_file](https://github.com/WWWangHan/Truck-cover-tarpaulin-detection/blob/master/readme_img/copy_file.jpg)

![test_output](https://github.com/WWWangHan/Truck-cover-tarpaulin-detection/blob/master/readme_img/test_output.png)

### mention
> This is a classification task, we call it detection just mean we wanna know whether the passing trucks cover the tarpaulin. 
> Due to the limited deployment environment, we fixed the gpu id for training to 1. Meanwhile, some of the running parameters are loaded from the `cfg/config.ini` so that the user can modify them easily.

