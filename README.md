## Face Photo-Sketch Synthesis via Intra-Domain Enhancement

torch Implementation of ["Face Photo-Sketch Synthesis via Intra-Domain Enhancement"](https://github.com/shenhaiyoualn/idesketch)

## Abstract

Face photo-sketch synthesis plays an increasingly important role in the field of law enforcement and entertainment. However, most of the existing methods only focus on how to cross the modality differences between the photo domain and the sketch domain, while neglecting the image quality enhancement of the generated faces. In this paper, we propose a novel intra-domain enhancement (IDE) method for the face photo-sketch synthesis task. Our method is composed of two steps to cope with the inter-domain generation gap and the intra-domain quality defect respectively, i.e. inter-domain face generation and intra-domain face enhancement. In the inter-domain face generation stage, we apply a probabilistic graphical model which is capable of synthesizing the coarse face sketch from photos with low quality. In the intra-domain face enhancement stage, a multi-Layer feature Aggregation (MLA) based Generative Adversarial Networks (GAN) is designed to enhance the facial details in the sketches and further refine the synthesized image quality. Our first phase helps bridge the modality gap between photo and sketch and ensures the shape consistency between them, and the second phase aims to improve the detail quality and resolution of the synthesized sketches. Extensive experimental results on public face sketch databases illustrate the effectiveness of our proposed approach.

### More synthesis results are available [Online](https://github.com/shenhaiyoualn/fine-sketch)

### The framework of our proposed Intra-Domain  Enhancement (IDE) method
<div align="center">
	<img src="imgs/IDE.PNG" width="80%" height="10%"/>
</div>
</a>

### The architecture of the Generator Network

<div align="center">
	<img src="imgs/G.PNG" width="80%" height="20%"/>
</div>
</a>

### The architecture of the Discriminator Network

<div align="center">
	<img src="imgs/D.PNG" width="300"/>
</div>
</a>



## Results
### The proposed IDE-based approach compares the effect of face sketchsynthesis  in  each  step  on  the  CUFSF  and  CUFS  datasets.  (a)  and  (d)  Inputphotos.  (b)  and  (d)  Results  of  the  inter-domain  face  generation.  (c)  and  (f)Results  of  the  IDE-base  method.  The  dataset  in  the  upper-left  corner  is  theCUFSF  dataset,  and  the  dataset  in  the  upper-right  corner  is  the  XM2VTSdataset,  AR  and  CUHK  datasets  are  in  the  lower-left  corner  and  the  lower-right corner, respectively

<div align="center">
	<img src="imgs/result1.PNG" width="80%" height="50%"/>
</div>
</a>

### Comparison on public face sketch dataset. The result of the first two columns are on the CUFSF dataset, and the following three datasets are in turn the comparison of XM2VTS,CUHK and AR dataset in different face sketch synthesis methods.

<div align="center">
	<img src="imgs/result2.PNG" width="80%" height="50%"/>
</div>
</a>


### Example pairs of input forensic images on a variety of real-world forensic photos for face sketch synthesis using our proposed IDE-based method. 

<div align="center">
	<img src="imgs/result3.PNG" width="60%"/>
</div>
</a>

### Comparison of celebrity photos retrieved from the Internet. (a) Input photos. (b) Results based on CUFSF. (c) Results based on CUFS.
<div align="center">
	<img src="imgs/result4.PNG" width="60%"/>
</div>
</a>


