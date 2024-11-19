
# Webapp 

```bash
pip install streamlit
pip install numpy
pip install Pillow
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install matplotlib
```
We have created our web application using streamlit. It takes in input of image and does segmentation using our model. 

To run the application

```bash
streamlit run Segmentation_App.py
```
Also add the ``model_checkpoint.pth`` in the root directory. You can get the these files from the ``drive url``.