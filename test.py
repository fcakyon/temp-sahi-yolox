from mmdet.apis import init_detector, inference_detector
from sahi.model import MmdetDetectionModel
from sahi.predict import get_sliced_prediction, predict
from sahi.utils.cv import read_image_as_pil


config_file = 'yolox_tiny_8x8_300e_coco.py'
checkpoint_file = 'yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'


detection_model = MmdetDetectionModel(
	model_path = checkpoint_file,
	config_path = config_file,
	confidence_threshold = 0.2,
	device = 'cpu'
)

# img = read_image_as_pil("images/2706.jpg")

# result = get_sliced_prediction(
#     img,
#     detection_model,
#     slice_height = 512,
#     slice_width = 512,
#     overlap_height_ratio = 0.2,
#     overlap_width_ratio = 0.2
# )

model_type = "mmdet"
model_device = "cpu" # or 'cuda:0'
model_confidence_threshold = 0.2

slice_height = 512
slice_width = 512
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "images/"

predict(
    model_type=model_type,
    model_path=checkpoint_file,
    model_config_path=config_file,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)