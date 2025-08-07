peft_label_template ='''
## Task Description
You are a multimodal sentiment classification expert for Twitter posts. 
First, thoroughly understand the content of this image.
Then your task is to infer the label of the post based on the information provided.

## Post Information
This Twitter post includes the text: {text} and the image.

## Response Format
"Prediction": "positive/neutral/negative"
'''
