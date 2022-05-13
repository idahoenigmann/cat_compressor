# cat_compressor
This projects goal is to compress images using a neural network. By the "fallen over 
hourglass" shape of the neural network, we hope to be able to produce new images by
feeding some (random?) data into the middle of the neural network. Btw everything will
be done using images of cats.

### getting this project to run on your machine
... is quite difficult.

Anyhow, you will want to get all the data and the pretrained model from 
[here](https://drive.google.com/drive/folders/1K58Dt07-jXyqFFZDlCrxCmVxdoDTbXZe?usp=sharing). 
Move the cat images to your ~/.keras/datasets directory. There is a 
somewhat pretrained nn weight and biases that you should move into the 
cat_compressor directory.

Get the following python modules:

* tensorflow and keras
* pillow
* pickle
* scipy
* sklearn

Run python cat_creator.py.

Open index.html with a web browser.