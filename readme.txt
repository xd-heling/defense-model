
Operating environment£º

tensorflow-gpu	1.14.0
Keras	2.3.1
numpy	1.16.0
opencv-python	3.4.1.15
scikit-image	0.16.1




Main function:

1.the priori method

image_pyr_read(path): get the image pyramid for the priori method.

image_priori(): this is a method that can detect adversarial examples and estimate the strength of perturbations.

2.the reconstruction model 

my_imread_pyr(path): get the image pyramid for the reconstruction model.

reconstruction_model(name): repair adversarial examples with strong perturbations.

model_fit(name): training model.

