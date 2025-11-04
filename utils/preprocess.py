from rembg import remove
from PIL import Image
import cv2
def preprocess_image(image_path):
    img = Image.open(image_path)
    no_bg_img = remove(img)
    img_cv = cv2.cvtColor(np.array(no_bg_img), cv2.COLOR_RGBA2BGR)
    high_res_img = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    product_features = clip_model.predict(high_res_img) 
    return high_res_img, product_features