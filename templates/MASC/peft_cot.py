peft_cot_template = '''
## Task Description
You are a multimodal sentiment classification expert for Twitter posts. 
First, thoroughly understand the content of this image.
Then your task is to infer the label of the target aspect based on the information provided.

Note that the sentiment toward the target aspect refers to the poster's sentiment regarding that specific aspect, not the sentiment of the entire sentence. You should regard this rule as your "first-principle" in judgment.

## Post Information
This Twitter post includes the text: {text} , target_aspect: {aspect} and the image.

## Response Format
"Text_analysis": "Briefly summarize the sentiment conveyed by the text toward the target aspect.",
"Image_analysis": "Briefly summarize the sentiment conveyed by the image toward the target aspect..",   
"Conflict_resolution": "If conflicts exist between text and image analyses, identify and resolve them.",  
"Final_conclusion": "Provide a comprehensive analysis by integrating insights from both text and image modalities.",  
"Prediction": "positive/neutral/negative"
'''
