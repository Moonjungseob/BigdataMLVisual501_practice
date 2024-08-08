from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# 경로 설정 및 모델 로드
model_weight_save_path = "testdjangoProject/resnet50_epoch_10.pth"
num_= 3

# ResNet-50 모델 정의 및 로드
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

# 모델 가중치 로드
checkpoint = torch.load(model_weight_save_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class ImageClassificationView(APIView):

    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']

            # 이미지 변환
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # 이미지 처리
            image = Image.open(image).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # 예측
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs, 1)
                predicted_class_index = predicted.item()
                confidence = probabilities[predicted_class_index].item()

                # 클래스 레이블 정의
                # class_labels = {0: '먼치킨', 1: '샴', 2: '스핑크스'}
                # 2조
                class_labels = {0: '업소용냉장고', 1: 'cpu', 2: '드럼세탁기', 3: '가정용냉장고', 4: '그래픽카드', 5: '메인보드'
                    , 6: '전자레인지', 7: '파워', 8: '렘', 9: '스탠드에어컨', 10: 'TV', 11: '벽걸이에어컨', 12: '통돌이세탁기'}

                # 정확도가 50% 미만인 경우 "기타"로 분류
                max_confidence, predicted = torch.max(probabilities, 0)
                if max_confidence < 0.5:
                    predicted_class_label = "기타"
                else:
                    predicted_class_label = class_labels.get(predicted.item(), "기타")

                # 모든 클래스에 대한 확률 반환
                class_confidences = {class_labels[i]: round(probabilities[i].item(), 4) for i in range(num_classes)}

            # 응답 데이터
            response_data = {
                'predicted_class_index': predicted_class_index,
                'predicted_class_label': predicted_class_label,
                'confidence': confidence,
                'class_confidences': class_confidences
            }

            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
