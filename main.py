from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import *
from demo import *

data_train = pd.read_csv("train_angle.csv")
data_test = pd.read_csv("test_angle.csv")

X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']

model = SVC(kernel='rbf', decision_function_shape='ovo',probability=True)
model.fit(X, Y)

# Test phase : build test dataset then evaluate
predictions = evaluate(data_test, model, show=True)

#Create a confusion matrix
cm = confusion_matrix(data_test['target'], predictions)

# Display the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#predict_video(model,'video_demo/yoga poses demo 2/yoga poses demo 2/goddess.mp4',show=True)
#predict('DATASET/TEST/goddess/00000092.png',model,show=True)
correct_feedback(model,'downdog_warrior2.mp4','teacher_yoga/angle_teacher_yoga.csv')

cv2.destroyAllWindows()
