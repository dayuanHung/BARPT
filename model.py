import torch
from torchvision.transforms import transforms
from furhat_remote_api import FurhatRemoteAPI
from PIL import Image
from torch import nn

class MLP(nn.Module):
    def __init__(self, features_in=20, features_out=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features_in, 128),
            nn.ReLU(),
            #nn.Linear(512, 128),
            #nn.ReLU(),
            nn.Linear(128, features_out)
        )

    def forward(self, input):
        return self.net(input)

####Notice!! This part hsan't finished yet! I am not sure how this part work.####
def process_interaction_with_features(predicted_emotion):
    # Define your interaction logic based on predicted_emotion here
    if predicted_emotion == 0:
        print("Detected emotion: Happy")
        # Perform actions for happy emotion
    elif predicted_emotion == 1:
        print("Detected emotion: Sad")
        # Perform actions for sad emotion
    elif predicted_emotion == 2:
        print("Detected emotion: Angry")
        # Perform actions for angry emotion
    else:
        print("Detected emotion: Other")
        # Perform default actions or handle other emotions

# Initialize connection to Furhat's API
furhat = FurhatRemoteAPI("localhost")

# Load the pre-trained model weights
model_path = 'best_model_lr0.0005.pth'
model = MLP(20,7)  # Replace with your model type and define your model architecture as needed
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Using the model within the interaction loop
while True:
    # Capture a frame from Furhat's webcam
    frame = furhat.capture_frame_from_webcam()  # This function might not be named exactly like this
    
    # Convert frame to a PIL image
    pil_image = Image.fromarray(frame)
    
    # Image preprocessing
    input_image = transform(pil_image).unsqueeze(0)
    
    # Perform inference using the model
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs, 1)
    
    # 'predicted' represents the predicted emotion; proceed with further processing or interaction logic
    process_interaction_with_features(predicted)

# Close the connection or release resources when done
#furhat.disconnect()

