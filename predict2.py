import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
import cv2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        # Conv layers. expect the shape to be [B, C, H, W]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # flatting all dimensions (including batch) for feeding a 1d array for the linear layers 
        x = torch.flatten(x, 0)
        #x = x.view(x.size(0), -1)

        # linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def predict(frame):

    CSV_path = "bom.csv"
    model_path = './pcb_components4.pth'
    confidence_threshold = 0.95

    wanted_comps = ["bosa", "tu_dien", "jack_nguon", "thieu_tu_dien", "thieu_jack_nguon", "thieu_nut_nguon", "thieu_reset", "thieu_wps", "nut_nguon", "reset",  "wps"]
    labels_map = {
        0: wanted_comps[0],
        1: wanted_comps[1],
        2: wanted_comps[2],
        3: wanted_comps[3],
        4: wanted_comps[4],
        5: wanted_comps[5],
        6: wanted_comps[6],
        7: wanted_comps[7],
        8: wanted_comps[8],
        9: wanted_comps[9],
        10: wanted_comps[10]
    }

    device = f"{torch.cuda.get_device_name()}" if torch.cuda.is_available() else "cpu"

    transform_img = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomResizedCrop(size=(32,32), scale=(0.8,1)),
                            transforms.ToTensor(),
                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        # Loading model
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load(model_path))
        # Only for prediction mode
        model.eval()
    
        csvFile = pd.read_csv(CSV_path, names=["point1_x", "point1_y", "point2_x", "point2_y"])

        # if csvFile is None or pcbImage is None:
        #     sys.exit("Couldn't load image or csv file!")

        # Cropping background - TODO: the values shouldn't be hard coded but imported from image2schematic
        #   x, y      x    y
        # region = [[245,140], [1405,880]]
        # pcbImage = pcbImage[region[0][1] : region[1][1] , region[0][0] : region[1][0]]
        
        # A copy of pcbImage so I could draw on it without messing with detection
        show_pcb_image = frame.copy()

        # validComponentsCounter = 0
        for i, row in csvFile.iterrows():
            point1_x = row['point1_x']
            point1_y = row['point1_y']
            point2_x = row['point2_x']
            point2_y = row['point2_y']

            # cropping component from image
            component = frame[point1_y: point2_y, point1_x: point2_x]
            # For inspectiong every detection 
            #cv2.imshow("result", component)
            
            # Transforming to fit model requirements
            component = transform_img(component).to(device)

            prediction = model(component)

            # Getting prediction and confidence score
            probs = torch.nn.functional.softmax(prediction, dim=-1)
            conf, classes = torch.max(probs, -1)
            if conf < confidence_threshold: continue

            # another way
            #predicted_class = np.argmax(prediction.cpu())
            

            #finalPrediction = labels_map[classes.item()]
            #print(finalPrediction, conf)
            #cv2.waitKey(0)

            # validComponentsCounter += 1

            red = [0,0,255]
            green = [0,255,0]
            # For Random color
            #color = (list(np.random.choice(range(256), size=3)))
            #color =[int(color[0]), int(color[1]), int(color[2])]
            if classes.item() > 2 and classes.item() < 8:
                cv2.rectangle(show_pcb_image, (point1_x, point1_y), (point2_x, point2_y), red, 2)
            else:
                cv2.rectangle(show_pcb_image, (point1_x, point1_y), (point2_x, point2_y), green, 2)

            cv2.putText(show_pcb_image, str(labels_map[classes.item()]) + " "+ str(round(conf.item(), 2)), (point1_x,point2_y+20), cv2.FONT_HERSHEY_DUPLEX, 0.5, [0,255,255], 1)

        #cv2.imshow("result", show_pcb_image)
        # print(f"number of valid Components: {validComponentsCounter}")

    return show_pcb_image


