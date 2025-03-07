# Segment River Images
We have finetuned the Segment Anything Model 2 \(SAM2\) model to create segmentation maps \(masks\) of images containing rivers. You have the option of using finetuning.ipynb to finetune the model on your own images of rivers if you have corresponding masks. You also have the option of loading the pretrained model, which can be found on Box, and immediately begin predicting masks for new images of rivers using predict_new.ipynb.

## Using the finetuning Notebook
**This is optional!** You may load in the pretrained model that we created and skip directly to predicting masks for new images.

You must first update the rootPath to the location of this repository on your machine. This can be found by running the following command:

```
!pwd
```

You will then need to update json_dir, which is defined in the 4th code chunk immediately following the definition of read_batch. We recommend placing all of your jsons in a folder in your current working directory and replacing `json_dir = rootPath + 'jsons'` with `json_dir = rootPath + {path to your jsons}`. Each json must have a path to an image of a river labeled "image" and a path to the corresponding mask labeled "map".

You will also need to ensure you have downloaded the SAM2 checkpoints, and may need to update the path to load them in. In our example, we use the tiny model checkpoint.

The rest of the defined code will train the model using the data you have provided it with. The final code chunk will save the model that you trained into your current working directory. You may change the name of the model configuration, but ensure that it ends with the ".pt" extension. You are now ready to predict masks for new images!

## Using the predict_new notebook
Begin by updating the rootPath \(run `!pwd` to find this\) and the modelPath, which will just be the name of the saved model configuration if your model configuration is stored in the current working directory. The model that we trained is currently set as the value of modelPath.

You will also need to ensure you have downloaded the SAM2 checkpoints, and may need to update the path to load them in. In our example, we use the tiny model checkpoint.

All that is left to do is edit `img = cv2.imread(rootPath + "/8999_-66.80780964334836_45.25465155192838_z18.png")`. If your images are saved in your current working directory, simply update the quoted image name. If your images are saved in some other location, delete "rootPath +" and input the full path as the quoted image name. The model will then predict and print the mask of the river!