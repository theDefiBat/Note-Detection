# Note-Detection
determining  currency denomination and detecting where its counterfeit or not using ML

# To Run
` cd app `
` python main.py `

# Problem Statement 
 To identify the denomination of a note using Machine Learning based on an image.
 To design the present currency template in ATM’s it requires 6-9 months.
 This is an attempt is to use Machine Learning(ML) to reduce this time.

# Introduction\Proposed solution
  CNNs are known to work better with Images as it works on a window(group of pixels)
  Used LeNET architecture which uses CNN and is simple to train on a CPU. 
  This model will classify the image given into one of the class.
  MLP(Multi-layer Perceptron) converts input into 1-D array, which causes loss of spatial correlation between the pixels
  counterfeit detection using auto-encoder technique
 
# results 

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/29819481/67925053-3a3b8e00-fbd8-11e9-829a-f8fac1c980b6.jpg">
</p>

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/29819481/67925054-3a3b8e00-fbd8-11e9-9d6c-efcffd855011.jpg">
</p>

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/29819481/67925055-3ad42480-fbd8-11e9-95b1-8e2882164361.jpg">
</p>

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/29819481/67925056-3ad42480-fbd8-11e9-977b-bb5a24e0290f.jpg">
</p>
