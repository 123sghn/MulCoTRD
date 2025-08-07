inference_cot_template = '''
## Task Description
You are a multimodal sentiment classification expert for Twitter posts. 
First, thoroughly understand the content of this image.
Then your task is to infer the label of the post based on the information provided.

## Sentiment Definitions
Each post is labeled in three-way classification scheme: ["positive", "neutral", "negative"].
The definitions of these three are given below:
1.Positive: Represents emotions or attitudes that are  happy, pleasant, optimistic or humorous.
2.Neutral: Represents emotions with no clear tendency, neither positive nor negative. It is often used to objectively describe facts, information without distinct emotional color.
3.Negative: Represents emotions or attitudes that are negative, angry, disappointed, fear, or irony.

## Post Information
This Twitter post includes the text: {text} and the image.

## Response Format
"Text_analysis": "Briefly summarize the sentiment conveyed by the text.",
"Image_analysis": "Briefly summarize the sentiment conveyed by the image.",   
"Conflict_resolution": "If conflicts exist between text and image analyses, identify and resolve them.",  
"Final_conclusion": "Provide a comprehensive analysis by integrating insights from both text and image modalities.",  
"Prediction": "positive/neutral/negative"
'''