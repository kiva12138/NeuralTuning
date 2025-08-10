CUDA = [0, 1, 2, 3]

LLAMATokenizerPath = r'/mnt/ssd/SH/Models/LLAMA/llama-2-7b-hf'

ClipPath = r'/mnt/ssd/SH/Models/CLIPHF/clip-vit-large-patch14'
ClipTokens = {
    r'/mnt/ssd/SH/Models/CLIPHF/clip-vit-base-patch16': 197,
    r'/mnt/ssd/SH/Models/CLIPHF/clip-vit-large-patch14': 257,
    r'/mnt/ssd/SH/Models/CLIPHF/clip-vit-large-patch14-336': 577
}
ClipImageSize = {
    r'/mnt/ssd/SH/Models/CLIPHF/clip-vit-base-patch16': 224,
    r'/mnt/ssd/SH/Models/CLIPHF/clip-vit-large-patch14': 224,
    r'/mnt/ssd/SH/Models/CLIPHF/clip-vit-large-patch14-336': 336
}
ClipGenFeaturePath = r'/mnt/ssd/SH/Datasets/Haru/CLIPGENFeatureUN'
ClipGenFeatureSize = 512

ImagePath = r'/mnt/ssd/SH/Datasets/Haru/MergedImages/'
DatasetPath = r'/mnt/ssd/SH/Datasets/Haru/'

ModelQuestionTemplate = "Write a response that appropriately completes the request. ##Request: {}\n##Response: "

CaptionPrompts = [
    "Describe the scene in this image."
    "What is in the image?",
    "Describe the scene in this image.",
    "What can you see in this picture?",
    "Provide a caption that captures the essence of this image.",
    "Imagine you are describing this image to someone who can't see it. What would you say?",
    "What story does this image tell?",
    "Describe the setting of this image.",
    "How would you describe the colors in this picture?",
    "Describe the action taking place in this image.",
    "Write a caption that highlights the contrast in this image.",
    "What is the main subject of this image?",
    "Write a caption that emphasizes the scale of this image.",
    "Describe the texture or patterns in this image.",
    "What is the focal point of this image?",
    "Write a caption that captures the movement in this picture.",
    "What objects can you identify in this picture?",
    "Write a caption that reflects the tranquility of this image.",
    "Describe the composition of this image.",
    "How would you describe the architecture in this picture?",
    "Write a caption that highlights the beauty of this image.",
    "Describe the symbolism in this image.",
    "Write a caption that conveys the energy of this image.",
    "Write a caption that reflects the simplicity of this image.",
]

GenerationPrompts = [
    "Generate an image based on the following description. {}",
    "Create a picture according to the text. {}",
    "Produce an image based on the following text. {}",
    "Please generate an image from the description provided. {}",
    "Generate a picture based on the following description. {}",
    "Create an image based on the text below. {}",
    "Produce a visual representation of the following text. {}",
    "Please generate an image based on the text provided. {}",
    "Generate an image that corresponds to the following description. {}",
    "Create a visual representation based on the text below. {}",
    "Produce an image according to the following description. {}",
    "Generate a picture based on the provided text. {}",
    "Please generate an image based on the description given. {}",
    "Create an image based on the following text description. {}",
    "Produce an image based on the description provided. {}",
    "Generate an image according to the following text. {}",
    "Create a visual representation according to the text. {}",
    "Produce an image based on the following description. {}",
    "Please generate an image according to the description. {}",
    "Generate a picture based on the text provided. {}",
    "Create an image based on the following description. {}",
    "Produce a visual representation based on the text. {}",
    "Generate an image according to the provided text. {}",
    "Please generate an image based on the description below. {}",
    "Generate a picture based on the following text. {}",
    "Create an image according to the text provided. {}",
    "Produce an image based on the following description. {}",
    "Please generate an image based on the text below. {}",
    "Generate an image according to the description provided. {}",
    "Create a visual representation of the text below. {}",
]

GenerationAnswerPrompts = [
    "<GLB>",
    "Yes, the generated image is <GLB>.",
    "Indeed, the image generated is <GLB>.",
    "Affirmative, the generated image is <GLB>.",
    "Absolutely, the image generated is <GLB>.",
    "Correct, the generated image is <GLB>.",
    "Right, the image generated is <GLB>.",
    "Exactly, the generated image is <GLB>.",
    "Indubitably, the image generated is <GLB>.",
    "Undoubtedly, the generated image is <GLB>.",
    "Without a doubt, the image generated is <GLB>.",
    "Certainly, the generated image is <GLB>.",
    "Surely, the image generated is <GLB>.",
    "Definitely, the generated image is <GLB>.",
    "Indeed, the generated image is <GLB>.",
    "For sure, the image generated is <GLB>.",
    "Absolutely, the generated image is <GLB>.",
    "Certainly, the image generated is <GLB>.",
    "Absolutely, the generated image is <GLB>.",
    "Without question, the image generated is <GLB>.",
    "Absolutely, the image generated is <GLB>.",
    "Certainly, the generated image is <GLB>.",
    "Without a doubt, the image generated is <GLB>.",
    "Yes, indeed, the generated image is <GLB>.",
    "Certainly, the image generated is <GLB>.",
    "Correct, the generated image is <GLB>.",
    "Yes, indeed, the image generated is <GLB>.",
    "Correct, the image generated is <GLB>.",
    "Right, the generated image is <GLB>.",
    "Without a doubt, the generated image is <GLB>.",
    "Absolutely, the generated image is <GLB>.",
    "The generated image is <GLB>.",
    "Here is the image generated: <GLB>.",
    "Behold, the image generated: <GLB>.",
    "This is the generated image: <GLB>.",
    "The resulting image is <GLB>.",
    "The image created is <GLB>.",
    "The image that was generated is <GLB>.",
    "Displaying the generated image: <GLB>.",
    "The generated image can be seen here: <GLB>.",
    "Witness the generated image: <GLB>.",
    "Revealing the generated image: <GLB>.",
    "The image produced is <GLB>.",
    "Showing the generated image: <GLB>.",
    "This is the image created: <GLB>.",
    "The generated image is displayed here: <GLB>.",
    "Presenting the generated image: <GLB>.",
    "Here is the generated image: <GLB>.",
    "The generated image is shown here: <GLB>.",
]

SegmentPrompts = [
    "Please segment {} from the image.",
    "Could you isolate {} from this picture?",
    "Can you extract {} from this scene?",
    "Let's focus on just {} in this image.",
    "Please highlight and separate {} from the rest of the image.",
    "Isolate {} to make it stand out.",
    "Could you extract and separate {} from the rest?",
    "Please segment and isolate {} from this image.",
    "I need {} to be segmented and highlighted, please.",
    "Could you segment {} to make it more prominent?",
    "Highlight and extract {} for a clearer view.",
    "Segment out {} to make it more visually appealing.",
    "I'd like to see just {} without any distractions.",
    "Please segment {} for a more detailed examination.",
    "Separate {} from the rest of the image for clarity.",
    "Can you extract and separate {} from this image?",
    "Please highlight and separate {} for a clearer view.",
    "I need {} to be segmented and isolated, please.",
    "Please highlight and isolate {} for a more focused view.",
    "I need {} to be segmented and highlighted, please.",
    "Could you extract {} to make it more visually striking?",
    "Separate {} from the rest of the image for better analysis.",
    "Let's highlight and isolate {} for a more focused view.",
    "I'd like to see just {} without any surrounding elements.",
    "I need {} to be segmented and highlighted, please.",
]

SegmentAnswerPrompts = [
    "The segmentation reveals {}.",
    "The segmented object is {}.",
    "The object highlighted in the segmentation is {}.",
    "This is the segmentation: {}.",
    "The object segmented is {}.",
    "The segmentation contains {}.",
    "The highlighted area in the segmentation is {}.",
    "The segmented object appears to be {}.",
    "The segmented area seems to be {}.",
]
