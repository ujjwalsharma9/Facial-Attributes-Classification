# Facial-Attributes-Classification

In this project, Facial Attribute Classification method is used in which first all faces are detected in an image using suitable face detection algorithm. Then these faces are cropped out of the original image and attribute classification models are applied to each one of them to predict the attributes of each face. During this process the correlation of the attributes is also taken care of. For example: The face which was detected as belonging to “Male” category is not checked further for “Wearing Lipstick” category. These attributes are displayed with an index to identify which person in that photo the attributes belong to.

The models have been trained individually using CNN on datasets such as CelebA and LFWA. Following attributes are selected based on accuracy and importance.

1) Gender: Male
2) Age Group: Child, Youth, Middle Aged, Senior
3) Race: Asian, White, Black, Indian
4) Expression: Mouth_Slightly_Open, Smiling, Frowning
5) Hair: Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair, Bald
6) Wearing: Wearing_Hat, Wearing_Lipstick, Wearing_Necktie, Eyeglasses
7) Others: 5_o_Clock_Shadow, Bangs, Heavy_Makeup, Mustache, No_Beard, Pale_Skin, Rosy_Cheeks, Sideburns


![test on harry potter image](https://github.com/ujjwalsharma9/Facial-Attributes-Classification/blob/master/imageTest/hp_feat.jpg)
