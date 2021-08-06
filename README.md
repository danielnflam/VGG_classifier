# VGG_classifier

VGG-based classifier trained on CXR images (RICORD-1c and RSNA Pneumonia Challenge).  The aim of the network was to classify CXRs into 2 classes (COVID-19 or non-COVID-19).  The network structure is based on that introduced by Rajaraman et al. https://doi.org/10.3390/diagnostics11050840

Pre-trained network has the following specs: LR=1e-3, batch size =4, Number of freeze layers = 28, num classes = 2, epochs = 13 complete epochs of training data.

### Training and testing

Train using train.ipynb script

Test using test.ipynb script
