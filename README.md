# Autonomous-Pi
Drives an autonomous raspberry pi robot using a convolutional neural network.

Watch the car drive!

https://www.youtube.com/watch?v=a3ui9-E9Szk

The CNN predicts 400 different drive states from images, which are being captured at a rate of 30 frames per second. The pi streams images to the main pc and receives drive commands in return.

The CNN architecture comes from a paper published by NVIDIA for their self driving car:

https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
