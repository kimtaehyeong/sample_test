# Test code for jetson nano 2GB temperature check <br><br>
You can do it in the following order.
<br><br>
cpu/gpu load (warming)<br>

```sudo docker pull nvcr.io/nvidia/l4t-ml:r32.6.1-py3```<br>
```sudo docker run -it --rm --runtime nvidia --network host -v /home:/location/in/container nvcr.io/nvidia/l4t-ml:r32.6.1-py3```<br>
```git clone https://github.com/kimtaehyeong/sample_test.git```<br>
```cd sample_test``` <br>
```python3 mnist_sample_1.py```<br><br>

cpu / gpu monitoring<br>
```sudo -H pip3 install -U jetson-stats```<br>
```sudo jtop```
