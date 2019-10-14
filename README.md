# MaximumMarginGANs
Code for paper: [Support Vector Machines, Wasserstein's distance and gradient-penalty GANs maximize a margin](xxxxxxx)

**Discussion at https://ajolicoeur.wordpress.com/MaximumMarginGANs.**

This basically the same code as https://github.com/AlexiaJM/relativistic-f-divergences, but with more options.

**Sample PyTorch code to use L1, L2, Linfinity gradient penalties with hinge or LS:**

```python

penalty = 20 # 10 is the more usual choice
u.resize_(batch_size, 1, 1, 1)
u.uniform_(0, 1)
x_both = x.data*u + x_fake.data*(1-u) # interpolation between real and fake samples
x_both = x_both.cuda()
x_both = Variable(x_both, requires_grad=True)
y0 = D(x_both)
grad = torch.autograd.grad(outputs=y0, inputs=x_both, grad_outputs=grad_outputs, retain_graph=True, 
create_graph=True, only_inputs=True)[0]
x_both.requires_grad_(False)
grad = grad.view(current_batch_size,-1)
grad_abs = torch.abs(grad) # Absolute value of gradient
			
if l1_margin: # Linfinity gradient norm penalty
  grad_norm , _ = torch.max(grad_abs,1)
elif l1_margin_smoothmax: # Smooth Maximum of absolute gradient penalty
  grad_norm = torch.sum(grad_abs*torch.exp(grad_abs))/torch.sum(torch.exp(grad_abs))
elif linf_margin: # L1 gradient norm penalty
  grad_norm = grad.norm(1,1) 
else: # L2 gradient norm penalty
  grad_norm = grad.norm(2,1)

if penalty_type == 'LS':
  constraint = (grad_norm-1).pow(2)
elif penalty_type == 'hinge':
  constraint = torch.nn.ReLU()(grad_norm - 1)

constraint = constraint.mean()
grad_penalty = penalty*constraint
grad_penalty.backward(retain_graph=True)
```

**Needed**

* Python 3.6
* Pytorch (Latest from source)
* Tensorflow (Latest from source, needed to get FID)
* Cat Dataset (http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd)

**To do beforehand**

* Change all folders locations in startup_tmp.sh, fid_script.sh, experiments.sh, and GAN.py
* Make sure that there are existing folders at the locations you used
* Open and run each necessary lines of setting_up_script.sh in same folder as preprocess_cat_dataset.py (It will automatically download the cat datasets, if this doesn't work well download it from http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd)

**To run models**
* HingeGAN Linfinity grad norm penalty with max(0, ||grad||-1):
   * python GAN.py --loss_D 3 --image_size 32 --CIFAR10 True --grad_penalty True --l1_margin --penalty-type 'hinge'
* WGAN Linfinity grad norm penalty with max(0, ||grad||-1):
   * python GAN.py --loss_D 4 --image_size 32 --CIFAR10 True --grad_penalty True --l1_margin --penalty-type 'hinge'
* WGAN L2 grad norm penalty with (||grad||-1)^2 (i.e., WGAN-GP):
   * python GAN.py --loss_D 4 --image_size 32 --CIFAR10 True --grad_penalty True
  
**To replicate the paper**
  * Open experiments.sh and run the lines you want

## Citation

If you find this code useful please cite us in your work:
```
@article{jolicoeur2018relativistic,
  title={The relativistic discriminator: a key element missing from standard GAN},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:1807.00734},
  year={2018}
}
```
