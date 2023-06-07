from keras.models import load_model
import numpy as np
import cv2

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def preprocess_image(image_path, desired_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size, desired_size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), desired_size / 30), -4, 128)

    return img


# Load mô hình đã train từ trước
loaded_model = load_model('model.h5')
img_path = r'test1.png'
im_size = 320
x_test = np.empty((1, im_size, im_size, 3), dtype=np.uint8)

x_test[0, :, :, :] = preprocess_image(
    f'{img_path}',
    desired_size=im_size
)

preds = loaded_model.predict(x_test)

# Hiển thị kết quả dự đoán
print('Kết quả dự đoán:', preds)
max_index = np.argmax(preds)

print("gia tri lon nhat",max_index)


